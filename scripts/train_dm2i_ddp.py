import os,sys,json,time,math,datetime,hashlib
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import torch,numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from interaction_k12 import InteractionK12,objective
from interaction_k12.data import SceneDataset,collate,model_inputs,labels
from interaction_k12.evaluation import evaluate

def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
class Batches:
 def __init__(self,n,epoch,rank):self.n=n;self.epoch=epoch;self.rank=rank
 def __len__(self):return math.ceil(self.n/4)
 def __iter__(self):
  order=torch.randperm(self.n,generator=torch.Generator().manual_seed(self.epoch)).tolist()
  for i in range(0,self.n,4):
   b=order[i:i+4];m=(len(b)+1)//2;yield b[:m] if self.rank==0 else b[m:]
def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);parser.add_argument('--config',default=str(REPO/'configs/d_m2i.json'));parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 R=Path(args.cache).parent;cache=Path(args.cache)
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);torch.manual_seed(0)
 dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45));device=f'cuda:{rank}'
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path(args.config).read_text());model=InteractionK12(cfg['model']).to(device)
 ck=None
 if args.resume:
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
 else:
  assert not (out/'last.pt').exists(),'Choose a fresh output or --resume'
 wrapped=DDP(model,device_ids=[rank],find_unused_parameters=True)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
 if ck:opt.load_state_dict(ck['optimizer'])
 train=SceneDataset(cache/'train.jsonl');val=SceneDataset(cache/'val.jsonl');assert len(train)==61975 and len(val)==7776
 assert not {r['split_group_id'] for r in train.rows}&{r['split_group_id'] for r in val.rows}
 bench=False;history=json.loads((out/'history.json').read_text()) if ck else []
 best=min((x['validation']['selection_pool8_group_minADE'] for x in history),default=float('inf'))
 steps=ck['steps'] if ck else 0;initial_epoch=ck['epoch'] if ck else 0;start=time.time();prior_elapsed=history[-1]['elapsed_s'] if history else 0.
 identity=dict(config=cfg,variant='D',seed=0,train_groups=len(train),val_groups=len(val),backbone_checkpoint_hash=train.rows[0]['backbone_hash'],extractor_version='interaction-scene-v1',aggregation='group-balanced',parallelism='DDP global batch 4',test_used=False)
 if rank==0:save(out/'identity.json',identity)
 for epoch in range(initial_epoch+1,31):
  loader=DataLoader(train,batch_sampler=Batches(len(train),epoch,rank),num_workers=2,persistent_workers=True,pin_memory=True,prefetch_factor=4,collate_fn=collate)
  epoch_values=[];timing=[];wrapped.train();batch_start=time.perf_counter()
  for bi,batch in enumerate(loader):
   x=model_inputs(batch,device);l=labels(batch,device);result=wrapped(x)
   loss=objective(result,x['native'],l['gt'],l['future_valid'],l['target_mask'],l['current_valid'],cfg['training']['lambda_disp'],cfg['training']['lambda_endpoint'])
   valid=l['future_valid']&l['target_mask'][...,None];supervised=valid.any(-1)&l['target_mask'];pairs=valid&torch.cat([l['current_valid'][...,None],valid[:,:,:-1]],-1)
   cnt=torch.tensor([supervised.any(-1).sum(),(supervised & pairs.any(-1)).any(-1).sum()],device=device,dtype=torch.float32);total=cnt.clone();dist.all_reduce(total)
   # Exact global grouped means, including the last uneven 3-sample batch.
   objective_global=(loss['pool']+loss['new'])*2*cnt[0]/total[0].clamp_min(1)+cfg['training']['lambda_disp']*loss['disp']*2*cnt[1]/total[1].clamp_min(1)
   assert cfg['training']['lambda_endpoint']==0
   opt.zero_grad(set_to_none=True);objective_global.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip']);assert torch.isfinite(norm)
   opt.step();steps+=1
   terms=torch.stack([loss['pool'].detach()*cnt[0],loss['new'].detach()*cnt[0],loss['disp'].detach()*cnt[1]]);dist.all_reduce(terms);terms/=torch.stack([total[0],total[0],total[1]]).clamp_min(1)
   vals=dict(pool=float(terms[0]),new=float(terms[1]),disp=float(terms[2]),gradient_norm=float(norm));vals['loss']=vals['pool']+vals['new']+cfg['training']['lambda_disp']*vals['disp'];epoch_values.append(vals)
   torch.cuda.synchronize();timing.append(time.perf_counter()-batch_start);batch_start=time.perf_counter()
   if bench and bi==99:
    if rank==0:save(R/'ddp_benchmark.json',dict(passed=True,steps=100,mean_step_s=float(np.mean(timing[10:])),samples_per_s=4/float(np.mean(timing[10:])),global_batch=4,model_checkpoint_modified=False))
    dist.barrier();dist.destroy_process_group();return
   if not bench and rank==0 and steps%20==0:save(out/'progress.json',dict(phase='training',epoch=epoch,epochs=30,steps=steps,elapsed_s=prior_elapsed+time.time()-start,last_loss=vals,parallelism='DDP',gpus=[0,1],step_s_recent=float(np.mean(timing[-20:]))))
  # Original evaluator, same sample order, batching and metrics. Both GPUs remain used for validation.
  dist.barrier()
  if rank==0:
   save(out/'progress.json',dict(phase='validation',epoch=epoch,steps=steps,gpus=[0,1]))
   vl=DataLoader(val,batch_size=4,num_workers=4,persistent_workers=True,pin_memory=True,collate_fn=collate)
   metrics=evaluate(torch.nn.DataParallel(model,device_ids=[0,1]),vl,'cuda:0');key=metrics['selection_pool8_group_minADE'];assert key is not None
   entry=dict(epoch=epoch,steps=steps,train={k:float(np.mean([v[k] for v in epoch_values])) for k in epoch_values[0]},validation=metrics,elapsed_s=prior_elapsed+time.time()-start,parallelism='DDP')
   history.append(entry);save(out/'history.json',history)
   checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),identity=identity,epoch=epoch,steps=steps,selection_value=key)
   torch.save(checkpoint,out/'last.tmp');(out/'last.tmp').replace(out/'last.pt')
   if key<best:best=key;torch.save(checkpoint,out/'best.tmp');(out/'best.tmp').replace(out/'best.pt');save(out/'selected_validation.json',entry)
   save(out/'progress.json',dict(phase='complete' if epoch==30 else 'training',epoch=epoch,steps=steps,best_validation_pool8_minADE=best,parallelism='DDP',elapsed_s=prior_elapsed+time.time()-start));print(epoch,key,flush=True)
  dist.barrier()
 if rank==0:save(out/'progress.json',dict(phase='complete',training_started=True,parallelism='DDP'))
 dist.destroy_process_group()
if __name__=='__main__':main()
