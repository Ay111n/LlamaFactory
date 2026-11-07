"""Gold-aligned auxiliary losses. Legacy termination objectives are untouched."""
import torch
import torch.nn.functional as F

GOLD_MODES = {
    "gold_margin": ("gold", "margin"),
    "dynamic_gold_margin": ("dynamic_gold", "dynamic"),
    "dynamic_boundary_ce": ("dynamic_ce", "margin"),
    "matched_gold_margin": ("gold", "matched_margin"),
    "matched_dynamic_boundary_ce": ("dynamic_ce", "matched_margin"),
}


def gold_boundary_losses(logits, labels, positions, kinds, end_id, input_ids=None):
    """Sum per-Think sample losses; caller divides by WINDOW Think count.

    p indexes logits; labels[p+1] is the gold token. 'kinds' holds binary
    continue/end flags, never vocabulary IDs. No token/label mutation occurs.
    """
    if len(positions) != logits.shape[0] or len(kinds) != len(positions):
        raise ValueError("Boundary batch size mismatch")
    zero = logits.reshape(-1)[:1].float().sum() * 0
    values = {k: [] for k in ("gold", "dynamic_gold", "dynamic_ce")}
    stats = dict(think=0, **{"continue": 0}, end_only=0)
    parts = {f"{k}_{c}": 0. for k in values for c in ("continue", "end")}
    for b, (pos, ys) in enumerate(zip(positions, kinds)):
        if not pos:
            if ys:
                raise ValueError("Labels without boundaries")
            continue
        if len(pos) != len(ys) or len(pos) != len(set(pos)) or ys.count(1) != 1:
            raise ValueError("Think requires unique positions and one true end")
        if min(pos) < 0 or max(pos) + 1 >= labels.shape[1] or any(y not in (0, 1) for y in ys):
            raise ValueError("Invalid Gold boundary")
        gold = labels[b, torch.as_tensor(pos, device=labels.device) + 1]
        is_end = torch.tensor(ys, device=logits.device, dtype=torch.bool)
        if (gold < 0).any() or (gold >= logits.shape[-1]).any():
            raise ValueError("Gold token is masked/padded/outside vocabulary")
        if not torch.equal(gold.eq(end_id), is_end):
            raise ValueError("Gold token/end label mismatch")
        if input_ids is not None and not torch.equal(gold, input_ids[b, torch.as_tensor(pos, device=input_ids.device)+1]):
            raise ValueError("Gold token differs from input_ids[p+1]")
        z = logits[b, pos].float()
        stop = z[:, end_id]
        d_continue = stop - z.gather(1, gold[:, None]).squeeze(1)
        # Compute only the true-end LSE; no exp subtraction or -inf mask clone.
        end_z = z[is_end]
        non_stop = torch.logaddexp(torch.logsumexp(end_z[:, :end_id], -1),
                                  torch.logsumexp(end_z[:, end_id+1:], -1))
        d = d_continue.clone()
        d[is_end] = non_stop - end_z[:, end_id]
        weight = 1 + d.sigmoid().detach()
        point = {"gold": F.softplus(d), "dynamic_gold": weight * F.softplus(d),
                 "dynamic_ce": weight * F.cross_entropy(z, gold, reduction="none")}
        cont = ~is_end
        has_cont = bool(cont.any())
        for key, vector in point.items():
            end_loss = vector[is_end].mean()
            cont_loss = vector[cont].mean() if has_cont else zero
            values[key].append(.5*(cont_loss+end_loss) if has_cont else end_loss)
            parts[key+"_continue"] += float(cont_loss.detach()) * (.5 if has_cont else 0)
            parts[key+"_end"] += float(end_loss.detach()) * (.5 if has_cont else 1)
        stats["think"] += 1
        stats["continue"] += int(cont.sum())
        stats["end_only"] += int(not has_cont)
    result = {k: torch.stack(v).sum() if v else zero for k, v in values.items()}
    return result, stats, parts
