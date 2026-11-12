import hashlib,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
BASE=Path('/path/to/experiment-root/InterHubNative_RouterAdapter')
sys.path.append(str(BASE))
from separate_horizon12 import Head,Response,features,geometry5,cues,descriptors
class PlanForecast:
    def __init__(self):
        folder=BASE/'locked_pair12'
        lock=json.loads((folder/'protocol.json').read_text())
        for file,digest in lock['weights'].items():
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest
        for file in ['separate_horizon12.py','signed_plan_response.py','conflict_order_response.py','cvdev_head.py']:
            assert hashlib.sha256((BASE/file).read_bytes()).hexdigest()==lock['implementation'][file]
        self.anchor=Head().cuda().eval()
        self.anchor.load_state_dict(torch.load(folder/'anchor.pt',map_location='cpu')['anchor'])
        self.nets=[]
        for name in ['five','eight']:
            ck=torch.load(folder/(name+'.pt'),map_location='cpu')
            net=Response().cuda().eval()
            net.load_state_dict(ck['state'])
            self.nets.append((net,ck['scale']))
    def __call__(self,f,plans,ctx,history):
        native=f['mu'].cuda()
        n=len(native);B=len(plans)
        yaw=f['yaw'].cuda()
        c,s=yaw.cos(),yaw.sin()
        rot=torch.stack([c,-s,s,c],1).reshape(n,2,2)
        org=f['origin'].cuda()
        agent=ctx['data']['agent']
        ego=agent.av_index
        pos=agent.position[ego,10,:2].cuda()
        past,now=history[-2:]
        dt=now.time_point.time_s-past.time_point.time_s
        assert dt>0
        velocity=pos.new_tensor([(now.center.x-past.center.x)/dt,(now.center.y-past.center.y)/dt])
        ts=torch.arange(1,17,device='cuda')*.5
        cv=pos+ts[:,None]*velocity
        b=torch.einsum('nti,nij->ntj',cv[None]-org[:,None],rot)
        plan=torch.einsum('bnti,nij->bntj',plans.cuda()[:,None,4::5,:2]-org[None,:,None],rot)
        x=native[:,:,4::5];q=f['log_w'].exp().cuda()
        context=features(x,q,torch.ones(n,dtype=torch.long,device='cuda'),b,b)
        ref=x+self.anchor(context)
        axes=cues(x,b,b)[1]
        def expand(a):return a.expand(B,*a.shape).flatten(0,1)
        xx=expand(x);r=plan.flatten(0,1)
        time5=(ts/5).clamp(max=1);time8=((ts-5)/3).clamp(min=0)
        h5=time5.square()*(3-2*time5);h8=time8.square()*(3-2*time8)
        def correction(k,fn):
            net,scale=self.nets[k]
            return (scale*net(expand(context),fn(xx,r)-expand(fn(x,b)),expand(axes))).reshape(B,n,6,2)
        d5=correction(0,geometry5);d8=correction(1,descriptors)
        residual=(ref-x)[None]+(h5*(1-h8))[:,None]*d5[...,None,:]+h8[:,None]*d8[...,None,:]
        # Preserve all learned 2 Hz knots and the native 10 Hz motion between them.
        knots=F.pad(residual.transpose(-1,-2),(1,0)).reshape(-1,2,17)
        dense=F.interpolate(knots,size=81,mode='linear',align_corners=True)[:,:,1:]
        cond=native[None]+dense.reshape(B,n,6,2,80).transpose(-1,-2)
        torch.testing.assert_close(cond[:,:,:,4::5],x[None]+residual,atol=1e-4,rtol=1e-5)
        return torch.cat([expand(native).reshape(B,*native.shape),cond],2)
