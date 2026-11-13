"""Verified loader for the released Direct D proposal generator."""
import hashlib,json
from pathlib import Path
import torch
from .model import InteractionK12

def load_direct_d(folder,device='cpu'):
 folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text());path=folder/manifest['checkpoint']
 if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Direct D checkpoint checksum mismatch')
 ck=torch.load(path,map_location='cpu',weights_only=True);identity=ck['identity']
 if identity['variant']!='D' or not identity['config']['model']['use_plan'] or not identity['config']['model']['post_interaction']:raise ValueError('Expected full Direct D architecture')
 model=InteractionK12(identity['config']['model']).to(device)
 model.load_state_dict(ck['model'],strict=True);model.eval();model.requires_grad_(False)
 return model