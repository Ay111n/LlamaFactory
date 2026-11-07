"""Auditable termination-margin objectives; no changes to the CE labels."""

import math

import torch
import torch.nn.functional as F


def boundary_losses(logits, positions, targets, end_id):
    """Positions index logits (not target tokens). Return per-sample losses."""
    ordinary, dynamic, stats = [], [], {"think": 0, "continue": 0, "end_only": 0}
    for b, (pos, ys) in enumerate(zip(positions, targets)):
        if not pos:
            continue
        if len(pos) != len(ys) or len(pos) != len(set(pos)) or ys.count(1) != 1:
            raise ValueError("Each Think needs unique boundaries and exactly one true end")
        if any(y not in (0, 1) for y in ys) or min(pos) < 0 or max(pos) >= logits.shape[1]:
            raise ValueError("Invalid termination boundary")
        z = logits[b, pos].float()
        # Avoid modifying model logits, and exclude the end token from the max.
        competitor = torch.maximum(z[:, :end_id].max(-1).values, z[:, end_id + 1:].max(-1).values)
        y = torch.tensor(ys, device=z.device, dtype=z.dtype)
        d = (1 - 2 * y) * (z[:, end_id] - competitor)
        base = F.softplus(d)
        weighted = (1 + torch.sigmoid(d).detach()) * base
        cont, end = y.eq(0), y.eq(1)
        def aggregate(v):
            return .5 * (v[cont].mean() + v[end].mean()) if cont.any() else v[end].mean()
        ordinary.append(aggregate(base))
        dynamic.append(aggregate(weighted))
        stats["think"] += 1
        stats["continue"] += int(cont.sum())
        stats["end_only"] += int(not cont.any())
    zero = logits.reshape(-1)[:1].float().sum() * 0
    return (torch.stack(ordinary).sum() if ordinary else zero,
            torch.stack(dynamic).sum() if dynamic else zero, stats)


def grad_norm(grads):
    return math.sqrt(sum(float(torch.linalg.vector_norm(g.float()).double().square()) for g in grads if g is not None))


def match_scale(g1, g2):
    n1, n2 = grad_norm(g1), grad_norm(g2)
    if not math.isfinite(n1 + n2) or (n1 == 0 and n2 != 0):
        raise FloatingPointError(f"Unmatchable auxiliary gradients: {n1}, {n2}")
    return (n2 / n1 if n1 else 1.), n1, n2
