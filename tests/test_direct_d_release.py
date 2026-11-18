import json,sys,tempfile,unittest,shutil
from pathlib import Path
import torch
from interaction_k12 import load_direct_d,InteractionK12
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
from direct_d_demo import synthetic_inputs

class ReleasedDirectD(unittest.TestCase):
 def test_released_weights_and_forward(self):
  torch.set_num_threads(2);folder=ROOT/'checkpoints/d_m2i';model=load_direct_d(folder)
  ck=torch.load(folder/'model.pt',weights_only=True);self.assertNotIn('optimizer',ck)
  reference=InteractionK12(ck['identity']['config']['model']).eval();reference.load_state_dict(ck['model'])
  self.assertEqual(ck['epoch'],26);self.assertFalse(any(p.requires_grad for p in model.parameters()))
  x=synthetic_inputs()
  with torch.inference_mode():a=model(x);b=reference(x)
  self.assertTrue(torch.equal(a['pool'],b['pool']));self.assertTrue(torch.equal(a['pool'][:,:,:6],x['native']))
 def test_checksum_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp);shutil.copyfile(ROOT/'checkpoints/d_m2i/manifest.json',path/'manifest.json');(path/'model.pt').write_bytes(b'corrupted')
   with self.assertRaisesRegex(ValueError,'checksum'):load_direct_d(path)
if __name__=='__main__':unittest.main()