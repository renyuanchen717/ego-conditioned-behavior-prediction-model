import ast
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from recorded_k12 import RecordedK12
from recorded_k12.metrics import displacement_metrics

ROOT=Path(__file__).resolve().parents[1]

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(812)
        cls.model=RecordedK12(ROOT/'checkpoints/locked')
        n=7;t=torch.arange(1,17)*.5
        cls.x=torch.randn(n,6,16,2)*.2+t[None,None,:,None]*torch.tensor([4.,.1])
        cls.q=torch.randn(n,6).softmax(-1);cls.typ=torch.tensor([1,2,3,1,1,2,3])
        cls.cv=torch.stack([8+5*t,torch.zeros_like(t)],-1)[None].repeat(n,1,1)
        cls.plan=cls.cv.clone();cls.plan[:,:,0]-=.14*t.square();cls.plan[:,:,1]+=.06*t.square()

    def test_original_reference_parity(self):
        # Execute original component definitions in an isolated namespace, not the portable module.
        ns={'torch':torch,'nn':nn,'F':F}
        entries=[('cvdev_head.py',['features','Head']),('signed_plan_response.py',['cues','Response']),('conflict_order_response.py',['descriptors']),('separate_horizon12.py',['geometry5'])]
        for name,names in entries:
            source=(ROOT/'research_reference'/name).read_text()
            for node in ast.parse(source).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                    exec(ast.get_source_segment(source,node),ns)
        anchor=ns['Head']();anchor.load_state_dict(self.model.anchor.state_dict())
        heads=[ns['Response'](),ns['Response']()]
        for h,original in zip(heads,self.model.responses):h.load_state_dict(original.state_dict())
        x,q,t,b,r=self.x,self.q,self.typ,self.cv,self.plan
        with torch.inference_mode():
            ctx=ns['features'](x,q,t,b,b);ref=x+anchor(ctx);axes=ns['cues'](x,b,b)[1]
            d5=self.model.scales[0]*heads[0](ctx,ns['geometry5'](x,r)-ns['geometry5'](x,b),axes)
            d8=self.model.scales[1]*heads[1](ctx,ns['descriptors'](x,r)-ns['descriptors'](x,b),axes)
            times=torch.arange(1,17)*.5
            def smooth(u):return (u*u*(3-2*u))[None,None,:,None]
            h5=smooth((times/5).clamp(max=1));h8=smooth(((times-5)/3).clamp(min=0))
            expected=torch.cat([x,ref+h5*(1-h8)*d5[:,:,None]+h8*d8[:,:,None]],1)
            actual=self.model(x,q,t,b,r)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_native_and_cv_preservation(self):
        pool=self.model(self.x,self.q,self.typ,self.cv,self.cv)
        self.assertTrue(torch.equal(pool[:,:6],self.x))
        from recorded_k12.components import features
        ref=self.x+self.model.anchor(features(self.x,self.q,self.typ,self.cv,self.cv))
        torch.testing.assert_close(pool[:,6:],ref,atol=0,rtol=0)
        recorded=self.model(self.x,self.q,self.typ,self.cv,self.plan)
        self.assertGreater(float((recorded-pool).abs().max()),0)

    def test_dense_knots(self):
        dense=F.interpolate(self.x.permute(0,1,3,2).reshape(-1,2,16),size=80,mode='linear',align_corners=False).reshape(7,6,2,80).transpose(-1,-2)
        cv=self.cv.repeat_interleave(5,1);plan=self.plan.repeat_interleave(5,1)
        out=self.model.predict_10hz(dense,self.q,self.typ,cv,plan)
        sparse=self.model(dense[:,:,4::5],self.q,self.typ,cv[:,4::5],plan[:,4::5])
        self.assertTrue(torch.equal(out[:,:6],dense))
        torch.testing.assert_close(out[:,:,4::5],sparse,atol=1e-5,rtol=1e-5)

    def test_metric_masks(self):
        pool=torch.zeros(2,6,16,2);gt=torch.zeros(2,16,2);gt[...,0]=1
        mask=torch.ones(2,16,dtype=torch.bool);mask[0,9]=False;mask[1]=False
        gt[~mask]=torch.nan
        m=displacement_metrics(pool,gt,mask)
        self.assertEqual(m['5s_minADE'][0],1)
        self.assertTrue(torch.isnan(m['5s_minFDE']).all())
        self.assertTrue(torch.isnan(m['8s_minADE'][1]))

    def test_reject_bad_probabilities(self):
        with self.assertRaises(ValueError):self.model(self.x,self.q*2,self.typ,self.cv,self.plan)

if __name__=='__main__':unittest.main()
