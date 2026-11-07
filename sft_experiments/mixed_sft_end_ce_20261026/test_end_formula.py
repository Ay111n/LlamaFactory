import unittest
import torch
import torch.nn.functional as F
from llamafactory.train.sft.termination_end_ce import end_boundary_losses
from llamafactory.train.sft.termination_gold import gold_boundary_losses

class EndFormulaTests(unittest.TestCase):
    def test_formula_detach_old_end_equivalence(self):
        torch.manual_seed(3407)
        for positions,kinds,alpha in [([[0,2]],[[0,1]],.5),([[2]],[[1]],1.)]:
            z=torch.randn(1,4,7,requires_grad=True)
            labels=torch.tensor([[-100,4,5,2]])
            one,two,_=end_boundary_losses(z,labels,positions,kinds,2)
            h=F.cross_entropy(z[0,2:3],torch.tensor([2]));w=2-(-h).exp().detach()
            torch.testing.assert_close(one,alpha*h)
            torch.testing.assert_close(two,alpha*w*h)
            g1=torch.autograd.grad(one,z,retain_graph=True)[0]
            g2=torch.autograd.grad(two,z,retain_graph=True)[0]
            torch.testing.assert_close(g2,w*g1)
            self.assertLess(g1[0,2,2],0);self.assertGreater(g1[0,2,4],0)
            self.assertTrue((g1[0,:2]==0).all())
            old,_,_=gold_boundary_losses(z,labels,[[2]],[[1]],2)
            for key,new in [('gold',one),('dynamic_gold',two),('dynamic_ce',two)]:
                torch.testing.assert_close(new,alpha*old[key])
                torch.testing.assert_close(torch.autograd.grad(new,z,retain_graph=True)[0],alpha*torch.autograd.grad(old[key],z,retain_graph=True)[0])

    def test_edges_alignment_and_no_think(self):
        z=torch.tensor([[[10000.,-10000.,0.,1.]]*4096],requires_grad=True)
        y=torch.full((1,4096),-100);y[0,-1]=2
        one,two,_=end_boundary_losses(z,y,[[4094]],[[1]],2)
        (one+two).backward();self.assertTrue(torch.isfinite(z.grad).all())
        one,two,_=end_boundary_losses(z,y,[[]],[[]],2)
        self.assertEqual(float(one+two),0)
        for p,k in [([[4095]],[[1]]),([[4094,4094]],[[0,1]]),([[0]],[[1]])]:
            with self.assertRaises(ValueError):end_boundary_losses(z,y,p,k,2)
        wrong=y.clone();wrong[0,-1]=1
        with self.assertRaises(ValueError):end_boundary_losses(z,y,[[4094]],[[1]],2,wrong)
        # No-think auxiliary gradient is exactly zero.
        z.grad=None;(one+two).backward();self.assertEqual(float(z.grad.abs().sum()),0)
if __name__=='__main__':unittest.main(verbosity=2)
