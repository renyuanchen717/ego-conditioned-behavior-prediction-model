import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .model import INPUT_KEYS

AGENT_KEYS=set(INPUT_KEYS)-{'map_features','map_position','map_yaw','map_mask','ego_plan','ego_plan_valid_mask','ego_current'}
AGENT_KEYS|={'gt','future_valid','current_valid','history','history_velocity'}
MAP_KEYS={'map_features','map_position','map_yaw','map_mask'}
def read_manifest(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
class SceneDataset(Dataset):
    def __init__(self,manifest):self.rows=read_manifest(manifest)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return torch.load(self.rows[i]['path'],map_location='cpu',weights_only=True)

def collate(samples):
    maxa=max(len(s['native']) for s in samples);maxm=max(max(1,len(s['map_features'])) for s in samples)
    result={}
    for key in samples[0]:
        if not torch.is_tensor(samples[0][key]):result[key]=[s[key] for s in samples];continue
        values=[]
        for s in samples:
            x=s[key]
            if key in AGENT_KEYS|MAP_KEYS:
                n=maxa if key in AGENT_KEYS else maxm
                v=x.new_zeros((n,*x.shape[1:]));v[:len(x)]=x;x=v
            values.append(x)
        result[key]=torch.stack(values)
    return result

def model_inputs(batch,device):return {k:batch[k].to(device) for k in INPUT_KEYS}
def labels(batch,device):return {k:batch[k].to(device) for k in ('gt','future_valid','current_valid','target_mask')}
