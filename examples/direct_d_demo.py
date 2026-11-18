"""Synthetic Direct D inference: execution/shape check, not an accuracy example."""
from pathlib import Path
import torch
from interaction_k12 import load_direct_d

def synthetic_inputs():
 torch.manual_seed(19);b,a,m=1,4,5
 x=dict(native=torch.randn(b,a,6,80,2),native_prob=torch.randn(b,a,6).softmax(-1),native_valid=torch.ones(b,a,dtype=torch.bool),agent_features=torch.randn(b,a,128),types=torch.tensor([[1,1,2,3]]),origin=torch.randn(b,a,2)*5,yaw=torch.randn(b,a),velocity=torch.randn(b,a,2),history_mask=torch.ones(b,a,11,dtype=torch.bool),ego_role=torch.tensor([[True,False,False,False]]),agent_present_mask=torch.ones(b,a,dtype=torch.bool),context_mask=torch.ones(b,a,dtype=torch.bool),target_mask=torch.tensor([[False,True,True,True]]),map_features=torch.randn(b,m,128),map_position=torch.randn(b,m,2)*8,map_yaw=torch.randn(b,m),map_mask=torch.ones(b,m,dtype=torch.bool),nearest_map=torch.tensor([[0,1,2,3]]),ego_plan=torch.randn(b,80,2).cumsum(1)*.1,ego_plan_valid_mask=torch.ones(b,80,dtype=torch.bool),ego_current=torch.zeros(b,2))
 x['origin'][:,0]=0;x['native_valid'][:,0]=False;x['native'][:,0]=0;x['native_prob'][:,0]=0
 return x

def main():
 torch.set_num_threads(2)
 model=load_direct_d(Path(__file__).resolve().parents[1]/'checkpoints/d_m2i')
 x=synthetic_inputs()
 with torch.inference_mode():out=model(x)
 assert out['pool'].shape==(1,4,12,80,2) and torch.isfinite(out['pool']).all()
 assert torch.equal(out['pool'][:,:,:6],x['native'])
 print('Direct D loaded; native modes preserved exactly. Pool shape:',tuple(out['pool'].shape))
 print('Only evaluate agents selected by target_mask. Synthetic inputs do not measure accuracy.')
if __name__=='__main__':main()