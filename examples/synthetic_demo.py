"""Synthetic smoke example; no dataset or NDPNet weights required."""
import argparse
import torch
from recorded_k12 import RecordedK12
p=argparse.ArgumentParser();p.add_argument('--checkpoints',default='checkpoints/locked');a=p.parse_args()
torch.manual_seed(7);torch.set_num_threads(2)
t=torch.arange(1,17)*.5
x=torch.zeros(3,6,16,2);x[...,0]=t*4
x[...,1]=torch.arange(6)[None,:,None]*.15
q=torch.full((3,6),1/6);typ=torch.tensor([1,2,3])
cv=torch.stack([t*5+8,torch.zeros_like(t)],-1)[None].repeat(3,1,1)
plan=cv.clone();plan[:,:,0]-=.2*t.square();plan[:,:,1]+=.04*t.square()
model=RecordedK12(a.checkpoints)
with torch.inference_mode():pool=model(x,q,typ,cv,plan)
assert pool.shape==(3,12,16,2) and torch.equal(pool[:,:6],x)
print('Output:',tuple(pool.shape),'native modes exactly preserved:',torch.equal(pool[:,:6],x))
print('Max conditional change vs CV:',(pool-model(x,q,typ,cv,cv)).abs().max().item())
