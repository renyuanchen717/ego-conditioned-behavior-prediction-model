import torch
import torch.nn.functional as F

def normalized_scores(logits, temperature=1.):
    if temperature <= 0: raise ValueError('temperature must be positive')
    return F.log_softmax(F.logsigmoid(logits) / temperature, dim=-1)

def log_prob(mu, scale, log_w, target, valid):
    """One shared categorical mode per complete trajectory, masked observations."""
    safe_target = torch.where(valid[..., None], target, torch.zeros_like(target))
    terms = torch.log(2 * scale) + (safe_target.unsqueeze(-3) - mu).abs() / scale
    component = -torch.where(valid.unsqueeze(-2).unsqueeze(-1), terms, 0.).sum((-1,-2))
    return torch.logsumexp(log_w + component, dim=-1)

def sample(mu, scale, log_w, kind='continuous'):
    if kind not in ('continuous','mode_mean'): raise ValueError(kind)
    index = torch.distributions.Categorical(logits=log_w).sample()
    gather = index[...,None,None,None].expand(*index.shape,1,mu.shape[-2],mu.shape[-1])
    loc=mu.gather(-3,gather).squeeze(-3)
    b=scale.gather(-3,gather).squeeze(-3)
    return torch.distributions.Laplace(loc,b).sample() if kind=='continuous' else loc

def log_density_ratio(pred,prior,target,valid,joint=False):
    """Ratio of two explicit normalized models, not a causal likelihood ratio.

    For J, pred contains teacher-forced conditional parameters at this target;
    all target coordinates must be observed for the exact joint ratio.
    """
    ratio=log_prob(pred['mu'],pred['scale'],pred['log_w'],target,valid)-log_prob(prior['mu'],prior['scale'],prior['log_w'],target,valid)
    return ratio.sum(-1) if joint else ratio
