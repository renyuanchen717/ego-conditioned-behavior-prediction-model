"""Official InterPlan runner; use original hydra scenario filters and licensed native DBs."""
import bootstrap,os,sys,collections,collections.abc,importlib
if not hasattr(collections,'Generator'):collections.Generator=collections.abc.Generator
from pathlib import Path
sys.path.insert(0,os.environ['INTERPLAN_ROOT'])
os.environ.setdefault('FLOW_INFERENCE_CONFIG',str(bootstrap.ROOT/'config/inference.yaml'))
os.environ.setdefault('NUPLAN_EXP_ROOT',str(bootstrap.OUTPUT))
# Same verified algebraic caching, no change to scores or candidate set.
os.environ.setdefault('FLOW_POOL_REUSE','1')
import occupancy_cache,batch_pdm,map_cache,evaluation_heartbeat
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
v=os.environ.get('DM2I_SCORING','worst12')
choices={'worst12':('dm2i','FlowDirectDM2IH8Planner'),'top1':('dm2ip','FlowDirectDM2IPH8Planner'),'worst2':('dm2ip2','FlowDirectDM2IP2H8Planner')}
module,cls=choices[v]
cfg=OmegaConf.load(bootstrap.ROOT/'config/planner.yaml')
cfg.flow_c0._target_=f'direct_{module}_h8_planner.{cls}'
cfg.flow_c0.variant='flow_'+module+'_'+v
ConfigStore.instance().store(group='planner',name='flow_c0',node=cfg)
import interplan.planning.script.run_simulation as runner
original=runner.run_simulation
def checked(cfg,**kwargs):
    assert 'NuPlanModifiedScenarioBuilder' in cfg.scenario_builder._target_
    out=Path(cfg.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'effective_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    return original(cfg,**kwargs)
runner.run_simulation=checked
runner.main()
