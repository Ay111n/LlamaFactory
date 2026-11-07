"""Full-vocabulary gold-constrained reasoning targets and bounded-memory KL.

Reference computations are frozen FP32. No top-k or vocabulary truncation.
KL backward recomputes each target chunk instead of retaining all targets.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def projected_log_probs(rp, r0, gold, beta=0.5, cap=2.0, iterations=32):
    gold = gold.reshape(-1, 1)
    score = beta * (rp - r0).clamp(-cap, cap)
    weights = (rp - rp.max(-1, keepdim=True).values).exp()
    wg, sg = weights.gather(1, gold), score.gather(1, gold)
    if not bool((wg > 0).all()):
        raise FloatingPointError("Underflow in gold reference weight")
    lo, hi = sg.clone(), score.max(-1, keepdim=True).values.clone()
    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        f = wg * (mid - sg) - (weights * (score - mid).clamp_min(0)).sum(-1, keepdim=True)
        lo, hi = torch.where(f < 0, mid, lo), torch.where(f >= 0, mid, hi)
    correction = torch.minimum(score, hi).scatter(1, gold, hi)
    return F.log_softmax(rp + correction, dim=-1)


class ReferenceTargets:
    def __init__(self, model, beta=0.5, cap=2.0, chunk=64):
        self.model, self.beta, self.cap, self.chunk = model, beta, cap, chunk
        self.device = next(model.parameters()).device
        self.forward_calls = 0
        self.target_calls = 0

    @torch.no_grad()
    def hidden(self, ids, positions):
        with torch.autocast("cuda", enabled=False):
            x = torch.tensor([ids], device=self.device, dtype=torch.long)
            result = self.model.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
            idx = torch.tensor(positions, device=self.device) - 1
            h = result.last_hidden_state[0].index_select(0, idx).detach()
        self.forward_calls += 1
        return h

    @torch.no_grad()
    def prepare(self, plus_ids, zero_ids, plus_positions, zero_positions, gold):
        return {
            "plus": self.hidden(plus_ids, plus_positions),
            "zero": self.hidden(zero_ids, zero_positions),
            "gold": torch.tensor(gold, device=self.device),
        }

    @torch.no_grad()
    def target(self, data, start, end, mode):
        with torch.autocast("cuda", enabled=False):
            r0 = F.log_softmax(self.model.lm_head(data["zero"][start:end]).float(), -1)
            if mode == "anchor":
                return r0
            rp = F.log_softmax(self.model.lm_head(data["plus"][start:end]).float(), -1)
            q = projected_log_probs(rp, r0, data["gold"][start:end], self.beta, self.cap)
        self.target_calls += 1
        return q


class ChunkedTargetKL(torch.autograd.Function):
    """Exact KL(q||softmax(logits)); q is frozen and recomputed in backward.

    logits are already selected answer positions [answer_tokens, vocabulary].
    Only this answer-sized logit view/copy is retained, not two teacher targets.
    """
    @staticmethod
    def forward(ctx, logits, provider, data, mode, chunk):
        ctx.save_for_backward(logits)
        ctx.provider, ctx.data, ctx.mode, ctx.chunk = provider, data, mode, chunk
        total = logits.new_zeros((), dtype=torch.float32)
        n = logits.shape[0]
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            lq = provider.target(data, start, end, mode).to(logits.device)
            ls = F.log_softmax(logits[start:end].float(), -1)
            total += (lq.exp() * (lq - ls)).sum()
        return total / n

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        n = logits.shape[0]
        with torch.no_grad():
            for start in range(0, n, ctx.chunk):
                end = min(n, start + ctx.chunk)
                lq = ctx.provider.target(ctx.data, start, end, ctx.mode).to(logits.device)
                probabilities = F.softmax(logits[start:end].float(), -1)
                q = lq.exp()
                # Retain the tiny FP32 normalization residual, matching dense
                # autograd's derivative of the actual computed KL exactly.
                gradient[start:end] = ((probabilities * q.sum(-1, keepdim=True) - q) * (grad_output / n)).to(logits.dtype)
        return gradient, None, None, None, None


def answer_kl(logits, provider, data, mode, chunk=64):
    return ChunkedTargetKL.apply(logits, provider, data, mode, chunk)
