import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from interaction_k12 import InteractionK12
from interaction_k12.data import SceneDataset,collate
from interaction_k12.evaluation import evaluate
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=4)
a=p.parse_args();torch.set_num_threads(2);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
model=InteractionK12(ck['identity']['config']['model']).to(a.device);model.load_state_dict(ck['model'])
data=SceneDataset(a.manifest)
assert {r['backbone_hash'] for r in data.rows}=={ck['identity']['backbone_checkpoint_hash']}
metrics=evaluate(model,DataLoader(data,batch_size=a.batch_size,collate_fn=collate),a.device)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics['metrics']['micro'],indent=2))
