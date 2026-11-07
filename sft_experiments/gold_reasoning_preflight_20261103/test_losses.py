"""Independent dense-autograd and sorted-root checks before actual training."""
import json
import torch
import torch.nn.functional as F
from losses import projected_log_probs, answer_kl


class DenseProvider:
    def __init__(self, target):
        self.target_logp = target

    def target(self, data, start, end, mode):
        return self.target_logp[start:end]


def sorted_projection(rp, r0, gold):
    # Independent active-set calculation in float64 (no bisection).
    p = rp.double().exp()
    score = 0.5 * (rp.double() - r0.double()).clamp(-2, 2)
    result = []
    for i in range(len(p)):
        g = int(gold[i])
        candidates = sorted((float(score[i, j]), float(p[i, j])) for j in range(p.shape[1]) if j != g)
        candidates.reverse()
        mass, moment, tau = float(p[i, g]), float(p[i, g] * score[i, g]), float(score[i, g])
        for value, weight in candidates:
            if value <= tau:
                break
            mass += weight
            moment += weight * value
            tau = moment / mass
        w = score[i].clamp_max(tau)
        w[g] = tau
        result.append(F.log_softmax(rp[i].double() + w, -1))
    return torch.stack(result)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    reports = []
    for vocab, n in ((7, 3), (31, 19), (1009, 11)):
        rp, r0 = [F.log_softmax(torch.randn(n, vocab) * 2, -1) for _ in range(2)]
        gold = torch.randint(vocab, (n,))
        q = projected_log_probs(rp, r0, gold)
        independent = sorted_projection(rp, r0, gold)
        projection_error = float((q.exp().double() - independent.exp()).abs().max())
        idx = gold[:, None]
        assert float((rp.gather(1, idx) - q.gather(1, idx)).max()) < 3e-6
        assert float(((q - q.gather(1, idx)) - (rp - rp.gather(1, idx))).max()) < 5e-6
        assert projection_error < 2e-6
        assert torch.allclose(projected_log_probs(rp, r0, gold, beta=0), rp, atol=2e-6)
        for chunk in (1, 8, 64):
            x = torch.randn(n, vocab, requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            loss = answer_kl(x, DenseProvider(q), None, "projected", chunk)
            expected = (q.exp() * (q - F.log_softmax(y, -1))).sum(-1).mean()
            (loss * 0.137).backward()
            (expected * 0.137).backward()
            error = float((x.grad - y.grad).abs().max())
            assert torch.allclose(loss, expected, atol=2e-6, rtol=2e-6)
            assert torch.allclose(x.grad, y.grad, atol=2e-7, rtol=2e-5)
            reports.append({"vocab": vocab, "n": n, "chunk": chunk, "loss_error": float((loss - expected).abs()), "gradient_error": error, "projection_probability_error": projection_error})
    result = {"passed": True, "tests": reports, "optimizer_updates": 0, "full_vocabulary": True}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
