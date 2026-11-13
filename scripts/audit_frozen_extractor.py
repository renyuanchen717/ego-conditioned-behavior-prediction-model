"""Optional real-backbone immutability audit using an installed, audited external wrapper."""
import argparse,sys,json,hashlib
from pathlib import Path
import torch
from interaction_k12 import InteractionK12,objective
from interaction_k12.data import SceneDataset,collate,model_inputs,labels
p=argparse.ArgumentParser();p.add_argument('--wrapper-root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--context',required=True);p.add_argument('--scene-manifest',required=True);p.add_argument('--output',required=True)
a=p.parse_args();sys.path.insert(0,a.wrapper_root);torch.set_num_threads(2)
from ndp_mode_adapter.frozen_features import FrozenFeatures
from ndp_adapter.frozen_backbone import state_hash
backbone=FrozenFeatures(a.checkpoint,'cpu');before=state_hash(backbone)
backbone.train(True)
assert not any(m.training for m in backbone.modules())
assert not any(p.requires_grad for p in backbone.parameters())
context=torch.load(a.context,map_location='cpu',weights_only=False)
with torch.no_grad():
    _,pred=backbone.forward_history_only(context['data'])
    changed=context['data'].clone()
    for key in ('position','velocity','heading'):changed['agent'][key][:,11:]=9999
    _,other=backbone.forward_history_only(changed)
    torch.testing.assert_close(pred['mu'],other['mu'],atol=0,rtol=0)
batch=collate([SceneDataset(a.scene_manifest)[0]])
model=InteractionK12({'hidden_dim':32,'dropout':0.}).train();opt=torch.optim.AdamW(model.parameters(),lr=.001)
x=model_inputs(batch,'cpu');lab=labels(batch,'cpu')
loss=objective(model(x),x['native'],lab['gt'],lab['future_valid'],lab['target_mask'],lab['current_valid'])
loss['loss'].backward();opt.step()
after=state_hash(backbone);assert before==after
report={'real_backbone_parameters_and_buffers_unchanged':True,'state_sha256_before':before,'state_sha256_after':after,'backbone_checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'all_modules_eval':True,'all_parameters_frozen':True,'target_future_mutation_prediction_max_abs_delta':float((pred['mu']-other['mu']).abs().max()),'new_module_training_loss':float(loss['loss']),'device':'cpu'}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
