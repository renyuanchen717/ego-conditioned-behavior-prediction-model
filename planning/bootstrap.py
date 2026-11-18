import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
OUTPUT=Path(os.environ.get('DM2I_OUTPUT',str(REPO/'outputs/interplan')))
OUTPUT.mkdir(parents=True,exist_ok=True)
FLOW=os.environ['FLOW_PLANNER_ROOT']
for p in [ROOT,REPO/'src',ROOT/'vendor_pdm',FLOW,os.environ['NDPNET_ROOT']]:sys.path.insert(0,str(p))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
if 'bool' not in np.__dict__:np.bool=np.bool_
