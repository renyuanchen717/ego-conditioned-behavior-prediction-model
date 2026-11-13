"""Export inference-only Direct D weights; omit optimizer and private dataset paths."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(source,map_location='cpu',weights_only=True);old=ck['identity']
 assert old['variant']=='D' and old['config']['model']['use_plan'] and old['config']['model']['post_interaction']
 keys=['config','seed','variant','backbone_checkpoint_hash','extractor_version','train_groups','val_groups','aggregation','split_checksum','trainable_parameters']
 public={'model':ck['model'],'epoch':ck['epoch'],'steps':ck['steps'],'selection_value':ck['selection_value'],'identity':{k:old[k] for k in keys}}
 dest=out/'model.pt';torch.save(public,dest);back=torch.load(dest,map_location='cpu',weights_only=True)
 assert set(back['model'])==set(ck['model']) and all(torch.equal(v,back['model'][k]) for k,v in ck['model'].items())
 manifest={'model':'Direct D','format':'inference-only PyTorch checkpoint; no optimizer or NDPNet weights','checkpoint':'model.pt','sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source_checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'epoch':ck['epoch'],'seed':old['seed'],'steps':ck['steps'],'backbone_checkpoint_sha256':old['backbone_checkpoint_hash'],'tensors_verified_identical':True}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()