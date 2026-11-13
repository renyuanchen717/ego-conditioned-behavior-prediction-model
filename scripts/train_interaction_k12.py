import argparse,hashlib,json,random,subprocess,time,os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from interaction_k12 import InteractionK12,objective
from interaction_k12.data import SceneDataset,collate,model_inputs,labels
from interaction_k12.evaluation import evaluate

def save_json(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/interaction_k12.yaml')
    p.add_argument('--variant',choices=['B','C','D','E'],default='D');p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--overfit-groups',type=int,default=0);p.add_argument('--overfit-steps',type=int,default=0);p.add_argument('--lambda-disp',type=float)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    cfg=json.loads(Path(a.config).read_text());cfg['model']['use_plan']=a.variant!='B';cfg['model']['post_interaction']=a.variant!='C'
    if a.variant=='E':cfg['training']['lambda_disp']=0.
    if a.lambda_disp is not None:cfg['training']['lambda_disp']=a.lambda_disp
    cfg['training']['epochs']=a.epochs or cfg['training']['epochs'];cfg['training']['batch_size']=a.batch_size or cfg['training']['batch_size']
    root=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise ValueError('Fresh output directory required; no overwrite')
    audit=json.loads((root/'audit.json').read_text())
    if audit['phase']!='complete' or audit['errors']:raise ValueError('Cache audit has not passed')
    train=SceneDataset(root/'train.jsonl');val=SceneDataset(root/'val.jsonl')
    for key in ['scene_id','split_group_id']:
        if {r[key] for r in train.rows}&{r[key] for r in val.rows}:raise ValueError('Scene leakage')
    hashes={r['backbone_hash'] for r in train.rows+val.rows}
    if len(hashes)!=1:raise ValueError('Mixed backbone checkpoints')
    model=InteractionK12(cfg['model']).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(a.device)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
    # Hash actual working-tree model/trainer sources: a git commit alone does not identify uncommitted new code.
    repo=Path(__file__).resolve().parents[1]
    source_hashes={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src/interaction_k12','scripts'] for f in (repo/folder).glob('*.py')}
    identity={'config':cfg,'normalization':{'position_m':30,'edge_position_m':50,'velocity_mps':15,'future_seconds':8},'git_commit':commit,'source_sha256':source_hashes,'seed':a.seed,'variant':a.variant,'backbone_checkpoint_hash':next(iter(hashes)),'extractor_version':'interaction-scene-v1','split_checksum':{s:hashlib.sha256((root/(s+'.jsonl')).read_bytes()).hexdigest() for s in ['train','val']},'trainable_parameters':sum(p.numel() for p in model.parameters()),'data_scope':'real pilot' if len(train)<10000 else 'full','train_groups':len(train),'val_groups':len(val),'test_used':False,'aggregation':'group-balanced','batch_size':cfg['training']['batch_size'],'pid':os.getpid()}
    save_json(out/'identity.json',identity);best=float('inf');steps=0;history=[];start=time.time()
    def step(batch):
        model.train();x=model_inputs(batch,a.device);l=labels(batch,a.device)
        result=model(x);loss=objective(result,x['native'],l['gt'],l['future_valid'],l['target_mask'],l['current_valid'],cfg['training']['lambda_disp'],cfg['training']['lambda_endpoint'])
        if not loss['groups']:return None
        opt.zero_grad();loss['loss'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        opt.step();return {k:float(loss[k].detach()) for k in ('loss','pool','new','disp')}|{'gradient_norm':float(norm)}
    if a.overfit_groups:
        loader=DataLoader(Subset(train,range(a.overfit_groups)),batch_size=a.overfit_groups,collate_fn=collate)
        batch=next(iter(loader));before=evaluate(model,loader,a.device)
        for s in range(a.overfit_steps or 200):
            value=step(batch);history.append(value)
            if s%25==0:save_json(out/'progress.json',{'phase':'tiny_overfit','step':s,'loss':value,'elapsed_s':time.time()-start})
        after=evaluate(model,loader,a.device);save_json(out/'overfit.json',{'before':before,'after':after,'steps':len(history),'history_first':history[:3],'history_last':history[-3:],'heldout':False})
        torch.save({'model':model.state_dict(),'identity':identity},out/'last.pt');save_json(out/'progress.json',{'phase':'complete','scope':'tiny_overfit','steps':len(history)});print('Tiny overfit complete',flush=True);return
    tl=DataLoader(train,batch_size=cfg['training']['batch_size'],shuffle=True,collate_fn=collate,num_workers=0)
    vl=DataLoader(val,batch_size=cfg['training']['batch_size'],shuffle=False,collate_fn=collate,num_workers=0)
    for epoch in range(1,cfg['training']['epochs']+1):
        epoch_losses=[]
        for batch in tl:
            result=step(batch);steps+=1
            if result:epoch_losses.append(result)
            if steps%5==0:save_json(out/'progress.json',{'phase':'training','epoch':epoch,'epochs':cfg['training']['epochs'],'steps':steps,'elapsed_s':time.time()-start,'last_loss':result})
        metrics=evaluate(model,vl,a.device);key=metrics['selection_pool8_group_minADE']
        if key is None:raise ValueError('No valid validation targets')
        entry={'epoch':epoch,'steps':steps,'train':{k:float(np.mean([l[k] for l in epoch_losses])) for k in epoch_losses[0]},'validation':metrics,'elapsed_s':time.time()-start,'peak_memory_gb':torch.cuda.max_memory_allocated(a.device)/1e9 if a.device.startswith('cuda') else None}
        history.append(entry);save_json(out/'history.json',history)
        ck={'model':model.state_dict(),'optimizer':opt.state_dict(),'identity':identity,'epoch':epoch,'steps':steps,'selection_value':key}
        tmp=out/'last.tmp';torch.save(ck,tmp);tmp.replace(out/'last.pt')
        if key<best:best=key;torch.save(ck,out/'best.pt');save_json(out/'selected_validation.json',entry)
        save_json(out/'progress.json',{'phase':'complete' if epoch==cfg['training']['epochs'] else 'training','epoch':epoch,'steps':steps,'best_validation_pool8_minADE':best,'elapsed_s':time.time()-start})
        print(a.variant,epoch,steps,key,flush=True)
if __name__=='__main__':main()
