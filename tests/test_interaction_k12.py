import copy,io,unittest
import torch
from torch import nn
from interaction_k12 import InteractionK12,FrozenBackboneGuard,objective
from interaction_k12.frames import to_local,to_scene

def fixture():
    torch.manual_seed(19);b,a,m=2,4,5
    x=dict(native=torch.randn(b,a,6,80,2),native_prob=torch.randn(b,a,6).softmax(-1),native_valid=torch.ones(b,a,dtype=torch.bool),agent_features=torch.randn(b,a,128),types=torch.tensor([[1,2,3,1]]).expand(b,-1).clone(),origin=torch.randn(b,a,2)*5,yaw=torch.randn(b,a),velocity=torch.randn(b,a,2),history_mask=torch.ones(b,a,11,dtype=torch.bool),ego_role=torch.zeros(b,a,dtype=torch.bool),agent_present_mask=torch.ones(b,a,dtype=torch.bool),context_mask=torch.ones(b,a,dtype=torch.bool),target_mask=torch.ones(b,a,dtype=torch.bool),map_features=torch.randn(b,m,128),map_position=torch.randn(b,m,2)*8,map_yaw=torch.randn(b,m),map_mask=torch.ones(b,m,dtype=torch.bool),nearest_map=torch.tensor([[0,1,2,3]]).expand(b,-1).clone(),ego_plan=torch.randn(b,80,2).cumsum(1)*.1,ego_plan_valid_mask=torch.ones(b,80,dtype=torch.bool),ego_current=torch.zeros(b,2))
    x['ego_role'][:,0]=True;x['native_valid'][:,0]=False;x['target_mask'][:,0]=False
    x['native'][:,0]=0;x['native_prob'][:,0]=0
    x['context_mask'][:,-1]=False;x['agent_present_mask'][:,-1]=False;x['target_mask'][:,-1]=False
    return x

class InteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.x=fixture();self.config={'hidden_dim':32,'heads':4,'dropout':0.}
        self.model=InteractionK12(self.config).eval()
    def test_shapes_preservation_and_save_reload(self):
        out=self.model(self.x);self.assertEqual(out['pool'].shape,(2,4,12,80,2))
        self.assertTrue(torch.equal(out['pool'][:,:,:6],self.x['native']))
        buf=io.BytesIO();torch.save(self.model.state_dict(),buf);buf.seek(0)
        model=InteractionK12(self.config).eval();model.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(model(self.x)['new'],out['new'],atol=0,rtol=0)
    def test_padding_and_cross_group(self):
        old=self.model(self.x)['new'];x=copy.deepcopy(self.x)
        for k in ['native','native_prob','agent_features','origin','yaw','velocity']:x[k][:,-1]=torch.nan
        out=self.model(x)['new'];torch.testing.assert_close(out[:,:3],old[:,:3])
        x=copy.deepcopy(self.x);x['ego_plan'][1]+=20;x['agent_features'][1]*=10
        torch.testing.assert_close(self.model(x)['new'][0],old[0])
        x['context_mask'][:]=False;x['agent_present_mask'][:]=False
        self.assertTrue(torch.isfinite(self.model(x)['new']).all())
    def test_agent_and_mode_permutations(self):
        old=self.model(self.x)['new'];x=copy.deepcopy(self.x);perm=torch.tensor([2,0,3,1])
        agentkeys=['native','native_prob','native_valid','agent_features','types','origin','yaw','velocity','history_mask','ego_role','agent_present_mask','context_mask','target_mask','nearest_map']
        for k in agentkeys:x[k]=x[k][:,perm]
        # Output is in meters after a 30m unit normalization; allow FP32 reduction-order noise.
        torch.testing.assert_close(self.model(x)['new'],old[:,perm],atol=1e-5,rtol=2e-5)
        modes=torch.tensor([5,2,1,4,0,3]);x=copy.deepcopy(self.x)
        x['native']=x['native'][:,:,modes];x['native_prob']=x['native_prob'][:,:,modes]
        torch.testing.assert_close(self.model(x)['new'],old,atol=1e-5,rtol=2e-5)
    def test_frames(self):
        old=self.model(self.x)['new'];x=copy.deepcopy(self.x);angle=torch.tensor(.7);shift=torch.tensor([40.,-10.])
        for k in ['origin','map_position','ego_plan','ego_current']:x[k]=to_scene(x[k],shift,angle)
        x['velocity']=to_scene(x['velocity'],torch.zeros(2),angle)
        x['yaw']+=angle;x['map_yaw']+=angle
        # Native is already target-local and must NOT be rotated a second time.
        torch.testing.assert_close(self.model(x)['new'],old,atol=3e-6,rtol=3e-5)
        p=torch.randn(7,2);o=torch.randn(7,2);y=torch.randn(7)
        torch.testing.assert_close(to_scene(to_local(p,o,y),o,y),p)
    def test_conditioning_and_null(self):
        x=copy.deepcopy(self.x);x['ego_plan'].requires_grad_(True)
        out=self.model(x);out['new'][:,:3].square().mean().backward()
        self.assertGreater(float(x['ego_plan'].grad.abs().sum()),0)
        null=InteractionK12({**self.config,'use_plan':False}).eval()
        a=null(x)['new'];x['ego_plan']=x['ego_plan'].detach()+100
        torch.testing.assert_close(null(x)['new'],a,atol=0,rtol=0)
    def test_training_frozen_and_gt_exclusion(self):
        bb=FrozenBackboneGuard(nn.Sequential(nn.Linear(3,8),nn.BatchNorm1d(8),nn.Dropout(.5)))
        before={k:v.clone() for k,v in bb.state_dict().items()};bb.train(True);bb(torch.randn(4,3))
        self.assertFalse(bb.training);self.assertFalse(bb.backbone.training)
        before_new={k:v.clone() for k,v in self.model.named_parameters()};self.model.train()
        y=torch.randn(2,4,80,2);v=torch.ones(2,4,80,dtype=torch.bool);v[:,:,20:25]=False;y[~v]=torch.nan
        output=self.model(self.x);loss=objective(output,self.x['native'],y,v,self.x['target_mask'],self.x['agent_present_mask'])
        loss['loss'].backward();opt=torch.optim.AdamW(self.model.parameters(),lr=.001);opt.step()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
        self.assertTrue(any(not torch.equal(p,before_new[k]) for k,p in self.model.named_parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in bb.state_dict().items()))
        with self.assertRaises(ValueError):self.model({**self.x,'gt':y})
    def test_displacement_mask_and_empty(self):
        n=torch.zeros(1,1,6,80,2);p=torch.ones_like(n,requires_grad=True)
        gt=torch.zeros(1,1,80,2);v=torch.zeros(1,1,80,dtype=torch.bool);v[:,:,0]=True;v[:,:,2]=True
        target=torch.ones(1,1,dtype=torch.bool)
        out=objective({'new':p},n,gt,v,target,target)
        self.assertAlmostEqual(float(out['disp']),2**.5,places=5)
        # Native wins the pool, but the new-only term still provides a gradient.
        out['loss'].backward();self.assertGreater(float(p.grad.abs().sum()),0)
        gt[:]=torch.nan;v[:]=False
        empty=objective({'new':p},n,gt,v,target,target)
        self.assertEqual(float(empty['loss']),0);self.assertEqual(empty['groups'],0)
    def test_isolated_ego_and_masked_maps(self):
        from interaction_k12.data import AGENT_KEYS
        x=copy.deepcopy(self.x)
        for k in AGENT_KEYS:
            if k in x:x[k]=x[k][:,:1]
        x['map_mask'][:]=False
        for k in ['map_features','map_position','map_yaw']:x[k][:]=torch.nan
        result=self.model(x)
        self.assertEqual(result['new'].shape,(2,1,6,80,2))
        self.assertTrue(torch.isfinite(result['new']).all())

if __name__=='__main__':unittest.main()
