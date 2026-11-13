"""Plot recorded training/validation histories (optional matplotlib dependency)."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True);a=p.parse_args()
fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
for name in ['B_seed0','C_seed0','D_seed0','E_seed0','D_seed1','D_seed2','D_disp5_seed0']:
 h=json.loads((Path(a.runs)/name/'history.json').read_text());epochs=[r['epoch'] for r in h]
 values=[[r['train']['loss'] for r in h],[r['validation']['selection_pool8_group_minADE'] for r in h],[r['validation']['metrics']['micro']['new6_8s_minADE']['mean'] for r in h]]
 for ax,v in zip(axes,values):ax.plot(epochs,v,label=name,linewidth=1.5)
for ax,title,y in zip(axes,['Training objective','Validation selection metric','Validation new-only error'],['Group-balanced loss','Group-balanced pool minADE@12 (m)','Target-micro new minADE@6 (m)']):
 ax.set_title(title);ax.set_xlabel('Epoch');ax.set_ylabel(y);ax.grid(alpha=.25)
axes[0].set_yscale('log');axes[-1].legend(fontsize=8)
fig.suptitle('Interaction Direct K=12 pilot: 122 train / 32 validation scenes; no test tuning')
Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)
