"""Observational heartbeat: no changes to planner inputs, outputs or RNG."""
import json, os, sys, time
from pathlib import Path
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
_out=next((x.split('=',1)[1] for x in sys.argv if x.startswith('output_dir=')),None)
_original=AbstractPlanner.compute_trajectory

def _compute(self, current_input):
    start=time.monotonic()
    result=_original(self,current_input)
    if _out:
        import torch
        p=Path(_out)/'heartbeat.json'
        record=dict(pid=os.getpid(),time=time.time(),iteration=current_input.iteration.index,
                    step_s=time.monotonic()-start,
                    gpu_allocated_mb=torch.cuda.memory_allocated()/2**20,
                    gpu_reserved_mb=torch.cuda.memory_reserved()/2**20)
        if record['gpu_reserved_mb']>4096 and record['gpu_allocated_mb']<1024:
            torch.cuda.empty_cache()
            record['cache_released']=True
            record['gpu_reserved_mb_after']=torch.cuda.memory_reserved()/2**20
        record['pdm_pool_reuse']=getattr(self,'_pool_reuse_report',None)
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record));tmp.replace(p)
    return result
AbstractPlanner.compute_trajectory=_compute
