import unittest,math
import torch
import torch.nn.functional as F
from transformers import Qwen3Config,Qwen3ForCausalLM
from component_probe import gold_boundary_losses,masks,component_grad,gram,statistics,collect_window

class AttributionTests(unittest.TestCase):
    def test_original_loss_masked_vjp(self):
        torch.manual_seed(31)
        z=torch.randn(2,4,8,requires_grad=True)
        ids=torch.tensor([[3,4,5,2],[3,4,2,0]])
        labels=ids.clone();pos=[[0,2],[1]];kinds=[[0,1],[1]]
        losses,_,_=gold_boundary_losses(z,labels,pos,kinds,2,ids)
        cm,sm=masks(z,pos,kinds)
        dc=z[0,0,2]-z[0,0,4]
        ds=[torch.logsumexp(torch.cat([v[:2],v[3:]]),0)-v[2] for v in [z[0,2],z[1,1]]]
        wc=1+dc.sigmoid().detach();ws=[1+d.sigmoid().detach() for d in ds]
        for key in ['gold','dynamic_gold','dynamic_ce']:
            c=F.softplus(dc) if key=='gold' else wc*F.softplus(dc) if key=='dynamic_gold' else wc*F.cross_entropy(z[0,0].unsqueeze(0),torch.tensor([4]))
            end=[F.cross_entropy(v.unsqueeze(0),torch.tensor([2])) for v in [z[0,2],z[1,1]]]
            if key!='gold':end=[w*v for w,v in zip(ws,end)]
            continue_loss=.5*c/2;stop_loss=(.5*end[0]+end[1])/2
            gc=component_grad(losses[key]/2,z,[z],cm)[0]
            gs=component_grad(losses[key]/2,z,[z],sm)[0]
            ga=component_grad(losses[key]/2,z,[z])[0]
            torch.testing.assert_close(gc,torch.autograd.grad(continue_loss,z,retain_graph=True)[0],rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(gs,torch.autograd.grad(stop_loss,z,retain_graph=True)[0],rtol=2e-5,atol=2e-7)
            torch.testing.assert_close(gc+gs,ga)
        # The mask hooks must have been removed after each VJP.
        expected=2*component_grad(losses['gold']/2,z,[z])[0]
        torch.testing.assert_close(torch.autograd.grad(losses['gold'],z)[0],expected)

    def test_projection_and_cancellation(self):
        vectors={'gold_c':[torch.tensor([1.,0.])],'gold_s':[torch.tensor([-.5,0.])],'gold_a':[torch.tensor([.5,0.])],'ce':[torch.tensor([0.,1.])]}
        keys,g=gram(vectors,[2],chunk=1);s=statistics(keys,g,'gold','gold_s',None,False)
        self.assertEqual(s['continue_projection_share_sum'],2.)
        self.assertEqual(s['stop_projection_share_sum'],-1.)
        self.assertAlmostEqual(s['cancellation_factor'],1/3)
        self.assertEqual(s['cos_continue_stop'],-1.)
        self.assertEqual(s['sum_vs_original_relative_l2_error'],0.)
        self.assertEqual(s['cos_aux_plain_stop_ce'],-1.)

    def test_model_accumulation_and_readonly(self):
        torch.manual_seed(3407);torch.set_num_threads(2)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=8,hidden_size=8,intermediate_size=12,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=8,attention_dropout=0.,tie_word_embeddings=False))
        records=[([3,4,5,2,6],[0,2],[0,1]),([3,4,5,6],[],[]),([3,4,2,6,7],[1],[1])]
        micro=[]
        for ids,pos,targets in records:
            micro.append({'input_ids':torch.tensor([ids]),'labels':torch.tensor([[-100]+ids[1:]]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),'term_positions':[pos],'term_targets':[targets]})
        full={'input_ids':torch.tensor([r[0]+[0]*(5-len(r[0])) for r in records]),'labels':torch.tensor([[-100]+r[0][1:]+[-100]*(5-len(r[0])) for r in records]),'attention_mask':torch.tensor([[1]*len(r[0])+[0]*(5-len(r[0])) for r in records]),'term_positions':[r[1] for r in records],'term_targets':[r[2] for r in records]}
        params=list(model.parameters());weights=[p.detach().clone() for p in params]
        for p in params:p.grad=torch.ones_like(p)
        grads=[p.grad.clone() for p in params];rng=torch.get_rng_state().clone()
        objectives=['gold','dynamic_gold','dynamic_ce']
        one,_=collect_window(model,[full],params,objectives,torch.device('cpu'),3407,True,2)
        many,meta=collect_window(model,micro,params,objectives,torch.device('cpu'),3407,True,2)
        reference_only,_=collect_window(model,micro,params,['gold'],torch.device('cpu'),3407,True,2,extra_totals=['dynamic_gold'])
        self.assertNotIn('dynamic_gold_c',reference_only)
        for a,b in zip(reference_only['dynamic_gold_a'],many['dynamic_gold_a']):
            if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(meta['window_think'],2);self.assertEqual(meta['counts']['end_only'],1)
        for key in one:
            for a,b in zip(one[key],many[key]):
                if a is not None:torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)
        for obj in objectives:
            for c,s,a in zip(many[obj+'_c'],many[obj+'_s'],many[obj+'_a']):
                if a is not None:torch.testing.assert_close(c+s,a,rtol=2e-4,atol=2e-6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for p,w,g in zip(params,weights,grads):
            torch.testing.assert_close(p,w,rtol=0,atol=0);torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

if __name__=='__main__':unittest.main(verbosity=2)
