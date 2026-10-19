"""Numerical contracts that protect the experimental comparison."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from lie_ssl.config import Config,VARIANTS
from lie_ssl.model import GroupLatentAutoencoder,circle_coordinates,principal_angle
from lie_ssl.reference_model import GroupLatentAutoencoder as Reference
from lie_ssl.losses import compute_method_losses
from lie_ssl.rotation_metrics import rotation_metrics
from lie_ssl.probe import relative_target,metrics_from_rows
from lie_ssl.runtime import compatible,save,load,seed_everything


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        seed_everything(42)
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def config(self,variant='baseline'):
        return Config(data_root='/unused',output_root='/unused',variant=variant)

    def test_baseline_reference_outputs_gradients(self):
        torch.manual_seed(42)
        old = Reference(24).to(self.device)
        old.normalize_generator_()
        new = GroupLatentAutoencoder(self.config()).to(self.device)
        new.load_state_dict(old.state_dict())
        first,second = torch.rand(2,1,28,28,device=self.device),torch.rand(2,1,28,28,device=self.device)
        for key in ('z','A','observed_recon','pose_coefficient'):
            torch.testing.assert_close(old(first)[key],new(first)[key],rtol=0,atol=0)
        losses = []
        for model in (old,new):
            loss = compute_method_losses(model,first,second)['total']
            loss.backward()
            losses.append(loss)
        torch.testing.assert_close(*losses,rtol=0,atol=0)
        for a,b in zip(old.parameters(),new.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_variants_finite_gradients_and_norms(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = self.config(variant)
                model = GroupLatentAutoencoder(config).to(self.device)
                optimizer = torch.optim.Adam(model.parameters(),lr=.001)
                for _ in range(2):
                    losses = compute_method_losses(model,torch.rand(2,1,28,28,device=self.device),
                                                    torch.rand(2,1,28,28,device=self.device))
                    optimizer.zero_grad()
                    losses['total'].backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()
                    model.normalize_generator_()
                expected = math.sqrt(24) if config.circle or config.structured else 1.
                self.assertAlmostEqual(float(model.action.B.detach().norm()),expected,places=5)

    def test_blocks_periodic_orthogonal(self):
        model = GroupLatentAutoencoder(self.config('block_skew')).to(self.device)
        B = model.action.B
        torch.testing.assert_close(B+B.T,torch.zeros_like(B))
        for j in range(0,24,2):
            self.assertEqual(float(torch.linalg.matrix_norm(B[j:j+2,j:j+2],ord=2)),1.)
        a = torch.tensor([[-math.pi],[0],[math.pi],[8.]],device=self.device)
        action,_ = model.action.matrix(a)
        torch.testing.assert_close(action,torch.matrix_exp(a.unsqueeze(-1)*B),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(action.transpose(-1,-2)@action,torch.eye(24,device=self.device).expand_as(action))
        torch.testing.assert_close(torch.matrix_exp(2*math.pi*B),torch.eye(24,device=self.device),rtol=1e-5,atol=1e-5)

    def test_circle_range_seam_and_zero(self):
        raw = torch.tensor([[0.,0.],[1.,0.],[-1.,-0.],[0.,1.],[0.,-1.]],requires_grad=True)
        unit = circle_coordinates(raw)
        torch.testing.assert_close(unit.norm(dim=-1),torch.ones(5))
        a = principal_angle(unit)
        self.assertTrue(bool(((a > -math.pi)&(a <= math.pi)).all()))
        a.sum().backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        model = GroupLatentAutoencoder(self.config('block_skew_circle'))
        plus,_ = model.action.matrix(torch.tensor([[-1.,1e-7]]))
        minus,_ = model.action.matrix(torch.tensor([[-1.,-1e-7]]))
        torch.testing.assert_close(plus,minus,rtol=1e-5,atol=1e-5)

    def test_r2_and_wrap(self):
        angle = np.arange(32)*2*np.pi/32
        target = np.stack((np.cos(angle),np.sin(angle)),axis=-1)
        self.assertAlmostEqual(rotation_metrics(target,target)['rotation_r2'],1.)
        self.assertAlmostEqual(rotation_metrics(target,np.broadcast_to(target.mean(0),target.shape))['rotation_r2'],0.)
        self.assertLess(rotation_metrics(target,-target)['rotation_r2'],0.)
        result = rotation_metrics(np.full((7,2),.1),np.zeros((7,2)))
        self.assertIsNone(result['rotation_r2'])
        seam = np.deg2rad([179.,-179.])
        target = np.stack((np.cos(seam),np.sin(seam)),axis=-1)
        self.assertAlmostEqual(rotation_metrics(target,target[::-1])['mae'],2.,places=5)
        forward = relative_target(torch.tensor([179.]),torch.tensor([-179.]))
        reverse = relative_target(torch.tensor([-179.]),torch.tensor([179.]))
        torch.testing.assert_close(forward[:,0],reverse[:,0])
        torch.testing.assert_close(forward[:,1],-reverse[:,1])

    def test_export_recompute(self):
        rows = [dict(target_cos=float(np.cos(t)),target_sin=float(np.sin(t)),
                     predicted_cos=.7*float(np.cos(t)),predicted_sin=.8*float(np.sin(t))) for t in np.arange(8)]
        result = metrics_from_rows('rotation',rows)
        expected = rotation_metrics([[r['target_cos'],r['target_sin']] for r in rows],
                                    [[r['predicted_cos'],r['predicted_sin']] for r in rows])
        self.assertEqual(result,expected)

    def test_resume_contract_and_optimizer(self):
        config = self.config('dense_circle')
        model = GroupLatentAutoencoder(config)
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        batch = torch.rand(2,1,28,28)
        loss = model(batch)['observed_recon'].sum()
        loss.backward()
        optimizer.step()
        model.normalize_generator_()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'
            save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config.to_dict(),data={},epoch=1),path)
            state = load(path)
            compatible(config,state,{})
            with self.assertRaises(ValueError):
                compatible(self.config('baseline'),state,{})
            replica = GroupLatentAutoencoder(config)
            replica.load_state_dict(state['model'])
            opt2 = torch.optim.Adam(replica.parameters(),lr=.001)
            opt2.load_state_dict(state['optimizer'])
            for m,o in ((model,optimizer),(replica,opt2)):
                o.zero_grad()
                m(batch)['observed_recon'].sum().backward()
                o.step()
                m.normalize_generator_()
            for a,b in zip(model.parameters(),replica.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__':
    unittest.main()
