import torch
def to_local(x,origin,yaw):
    """Broadcastable row-vector transform; last axis is xy."""
    d=x-origin;c=yaw.cos();s=yaw.sin()
    return torch.stack([d[...,0]*c+d[...,1]*s,-d[...,0]*s+d[...,1]*c],-1)
def to_scene(x,origin,yaw):
    c=yaw.cos();s=yaw.sin()
    return torch.stack([x[...,0]*c-x[...,1]*s,x[...,0]*s+x[...,1]*c],-1)+origin
