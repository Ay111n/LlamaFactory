from experiment_paths import experiment_root,benchmark_root,data_root,model_root,paths,require_path,script_path,stage_log,evaluation_analysis_root
import importlib.util
import math
import os
from pathlib import Path
import unittest

import torch

ROOT = require_path('framework_root')
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'src/llamafactory/train/sft'/f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
losses = load('termination_loss')
boundaries = load('termination_boundaries')


class TerminationTests(unittest.TestCase):
    def test_single_boundary_formula_and_detach(self):
        for y in (0, 1):
            z = torch.tensor([[[0., 2., 1., -1.]]], requires_grad=True)
            if not y:
                # Add a separate true end; examine just the continue gradient.
                z = z.detach().repeat(1,2,1).requires_grad_()
                pos, ys = [[0,1]], [[0,1]]
            else:
                pos, ys = [[0]], [[1]]
            one, two, _ = losses.boundary_losses(z,pos,ys,2)
            g1 = torch.autograd.grad(one,z,retain_graph=True)[0]
            g2 = torch.autograd.grad(two,z)[0]
            d = (1-2*y)*(-1.)
            self.assertAlmostEqual(float(g2[0,0,2]/g1[0,0,2]), 1+torch.sigmoid(torch.tensor(d)).item(),places=6)
            self.assertGreater(float(g1[0,0,2])*(1-2*y),0)

    def test_multiboundary_manual_sum(self):
        torch.manual_seed(12)
        z = torch.randn(2,7,5,requires_grad=True)
        p,y = [[0,2,5],[3]], [[0,0,1],[1]]
        one,two,_=losses.boundary_losses(z,p,y,2)
        manual=z.sum()*0
        for b in range(2):
            vals=[]
            for t,label in zip(p[b],y[b]):
                v=z[b,t].float()
                d=(1-2*label)*(v[2]-torch.cat((v[:2],v[3:])).max())
                vals.append((1+d.sigmoid().detach())*torch.nn.functional.softplus(d))
            manual += .5*(torch.stack(vals[:-1]).mean()+vals[-1]) if len(vals)>1 else vals[0]
        torch.testing.assert_close(two,manual)
        torch.testing.assert_close(torch.autograd.grad(two,z,retain_graph=True)[0],torch.autograd.grad(manual,z)[0])

    def test_accumulation_and_matching(self):
        torch.manual_seed(8)
        w=torch.randn(4,5,requires_grad=True)
        x=torch.randn(3,6,4)
        p,y=[[0,4],[],[2]],[[0,1],[],[1]]
        a,b,_=losses.boundary_losses(x@w,p,y,2)
        g1=torch.autograd.grad(a/2,w,retain_graph=True)[0]
        g2=torch.autograd.grad(b/2,w)[0]
        summed=[torch.zeros_like(w),torch.zeros_like(w)]
        for i in range(3):
            a,b,_=losses.boundary_losses(x[i:i+1]@w,[p[i]],[y[i]],2)
            summed[0]+=torch.autograd.grad(a/2,w,retain_graph=True)[0]
            summed[1]+=torch.autograd.grad(b/2,w)[0]
        torch.testing.assert_close(g1,summed[0]);torch.testing.assert_close(g2,summed[1])
        c,n1,n2=losses.match_scale([g1],[g2])
        c2,_,_=losses.match_scale([summed[0]],[summed[1]])
        self.assertAlmostEqual(c,c2,places=6)
        self.assertAlmostEqual(losses.grad_norm([c*g1]),n2,places=5)
        ce=torch.randn_like(w)
        corrected=ce+g1+(c-1)*g1
        torch.testing.assert_close(corrected,ce+c*g1)
        before=corrected.clone()
        parameter=torch.nn.Parameter(w.detach().clone());parameter.grad=corrected
        torch.nn.utils.clip_grad_norm_([parameter],.1)
        self.assertLessEqual(float(parameter.grad.norm()),.100001)

    def test_extremes_padding_zero_and_duplicates(self):
        z=torch.tensor([[[10000.,-10000.,0.,1.]]*4096],requires_grad=True)
        a,b,_=losses.boundary_losses(z,[[4095]],[[1]],2)
        self.assertTrue(torch.isfinite(a+b));(a+b).backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        z.grad=None
        a,b,s=losses.boundary_losses(z,[[]],[[]],2)
        (a+b).backward();self.assertEqual(float(z.grad.abs().sum()),0)
        with self.assertRaises(ValueError):losses.boundary_losses(z,[[0,0]],[[0,1]],2)
        self.assertEqual(losses.match_scale([],[])[0],1)
        with self.assertRaises(FloatingPointError):losses.match_scale([torch.zeros(1)],[torch.ones(1)])
        tied=torch.ones(1,2,5,requires_grad=True)
        a,b,_=losses.boundary_losses(tied,[[0,1]],[[0,1]],2)
        (a+b).backward();self.assertTrue(torch.isfinite(tied.grad).all())

    def test_rule_boundaries_and_offsets(self):
        import spacy
        nlp=spacy.blank('en');nlp.add_pipe('sentencizer')
        text='Dr. Smith uses 3.14. Done!\n```python\nx = 2.0\n```\nNext?'
        found=boundaries.find_boundaries(text,nlp)
        self.assertTrue(any(text[:p].endswith('Done!') for p,_ in found))
        self.assertFalse(any(text[:p].endswith('Dr.') for p,_ in found))
        self.assertFalse(any(text[:p].endswith('2.0\n') for p,_ in found))
        rendered='<think>A.\nB.</think>'
        # Character tokens make the logit/next-token shift explicit.
        ids=list(range(len(rendered))); ids[12]=99
        offsets=[(i,i+1) for i in range(len(rendered))]
        # </think> is a single token at character 12.
        ids=ids[:12]+[99];offsets=offsets[:12]+[(12,len(rendered))]
        p,y,a=boundaries.map_boundaries(rendered,ids,offsets,7,12,nlp,99)
        self.assertEqual((p[-1],y[-1]),(11,1))
        self.assertEqual(len(p),len(set(p)))


if __name__=='__main__':unittest.main(verbosity=2)
