"""Candidate-aware conditional ranking. Frozen trajectories are never modified."""
import math
import torch
from torch import nn


def mlp(n, h=128):
    return nn.Sequential(nn.Linear(n, h), nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h), nn.GELU())


class ProbabilityHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.trajectory = mlp(160, hidden)
        self.relative = mlp(240, hidden)
        self.agent = mlp(128, hidden)
        self.scene = mlp(256, hidden)
        self.map_geometry = mlp(32, hidden)
        self.prior = mlp(2, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.output = nn.Linear(hidden, 1)
        # Begin close to uniform rather than assigning arbitrary high confidence.
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, pool):
        pool = pool[:, :, 6:]
        b, a, k, t, _ = pool.shape
        assert (k, t) == (6, 80)
        active = x['agent_present_mask'] & x['context_mask']
        features = torch.where(active[..., None], x['agent_features'], 0.)
        scene_agents = features.sum(1) / active.sum(1, keepdim=True).clamp_min(1)
        maps = torch.where(x['map_mask'][..., None], x['map_features'], 0.)
        scene_maps = maps.sum(1) / x['map_mask'].sum(1, keepdim=True).clamp_min(1)
        scene = self.scene(torch.cat([scene_agents, scene_maps], -1))
        # Ego plan is supplied in scene axes. Predictions are target-local.
        delta = x['ego_plan'][:, None] - x['origin'][:, :, None]
        c, s = x['yaw'].cos()[..., None], x['yaw'].sin()[..., None]
        ego = torch.stack([delta[..., 0]*c + delta[..., 1]*s,
                           -delta[..., 0]*s + delta[..., 1]*c], -1)
        relative = ego[:, :, None] - pool
        relation = torch.cat([relative / 30., relative.norm(dim=-1, keepdim=True) / 30.], -1)
        # Per-candidate nearest-map distance / heading at 16 future points.
        local = pool[..., 4::5, :]
        co, si = x['yaw'].cos()[..., None, None], x['yaw'].sin()[..., None, None]
        world = torch.stack([local[..., 0]*co-local[..., 1]*si,
                             local[..., 0]*si+local[..., 1]*co], -1) + x['origin'][:, :, None, None]
        map_pos = torch.where(x['map_mask'][..., None], x['map_position'], 0.)
        distance = torch.cdist(world.reshape(b, -1, 2), map_pos)
        distance = distance.masked_fill(~x['map_mask'][:, None], 1e4)
        nearest_distance, nearest = distance.min(-1)
        nearest_yaw = x['map_yaw'].gather(1, nearest).reshape(b, a, k, 16)
        movement = torch.diff(local, dim=-2, prepend=torch.zeros_like(local[..., :1, :]))
        heading = torch.atan2(movement[..., 1], movement[..., 0]) + x['yaw'][..., None, None]
        geometry = torch.stack([nearest_distance.reshape(b, a, k, 16).clamp_max(100.) / 30.,
                                (heading-nearest_yaw).cos()], -1)
        geometry = geometry * x['map_mask'].any(1)[:, None, None, None, None]
        prior = torch.zeros(b,a,k,device=pool.device,dtype=pool.dtype)
        native_flag = torch.zeros_like(prior)
        z = (self.trajectory((pool / 30.).flatten(-2)) + self.relative(relation.flatten(-2))
             + self.map_geometry(geometry.flatten(-2)) + self.prior(torch.stack([prior, native_flag], -1))
             + self.agent(features)[:, :, None] + scene[:, None, None])
        z = z.reshape(b*a, k, -1)
        attention, _ = self.attention(z, z, z, need_weights=False)
        logits = self.output(self.norm(z + attention)).reshape(b, a, k)
        return logits


def target_valid(batch):
    return batch['future_valid'].bool() & batch['target_mask'][..., None].bool()


def energy(pool, gt, valid, sigma=2.):
    # Mask before arithmetic: unavailable GT may contain NaNs.
    delta = torch.where(valid[:, :, None, :, None], pool-gt[:, :, None], 0.)
    return delta.square().sum(-1).sum(-1) / valid.sum(-1).clamp_min(1)[..., None] / (2*sigma*sigma)


def loss_per_scene(logits, pool, batch, sigma=2.):
    valid = target_valid(batch)
    mask = valid.any(-1)
    e = energy(pool[:, :, 6:], batch['gt'], valid, sigma)
    # Length-normalized mixture energy NLL, not a calibrated physical joint likelihood.
    losses = -torch.logsumexp(logits.log_softmax(-1)-e, -1)
    return (losses*mask).sum(-1)/mask.sum(-1).clamp_min(1), mask.any(-1)
