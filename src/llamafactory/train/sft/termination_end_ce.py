"""End-only CE auxiliary objectives; retain original metadata and class weights."""
import torch
import torch.nn.functional as F

END_MODES = {'end_ce': 'margin', 'dynamic_end_ce': 'dynamic', 'matched_end_ce': 'matched_margin'}


def end_boundary_losses(logits, labels, positions, kinds, end_id, input_ids=None):
    if len(positions) != logits.shape[0] or len(kinds) != len(positions):
        raise ValueError('Boundary batch size mismatch')
    zero = logits.reshape(-1)[:1].float().sum() * 0
    ordinary, dynamic = [], []
    stats = {'think': 0, 'continue': 0, 'end_only': 0}
    for b, (pos, ys) in enumerate(zip(positions, kinds)):
        if not pos:
            if ys: raise ValueError('Targets without positions')
            continue
        if len(pos) != len(ys) or len(set(pos)) != len(pos) or ys.count(1) != 1:
            raise ValueError('Think requires unique positions and one true end')
        if any(y not in (0, 1) for y in ys) or min(pos) < 0 or max(pos)+1 >= labels.shape[1]:
            raise ValueError('Invalid boundary')
        next_pos = torch.tensor(pos, device=labels.device) + 1
        gold = labels[b, next_pos]
        ends = torch.tensor(ys, device=gold.device, dtype=torch.bool)
        if (gold < 0).any() or (gold >= logits.shape[-1]).any() or not torch.equal(gold.eq(end_id), ends):
            raise ValueError('Masked/out-of-range or mismatched boundary target')
        if input_ids is not None and not torch.equal(gold, input_ids[b, next_pos]):
            raise ValueError('Gold differs from input_ids[p+1]')
        p = pos[ys.index(1)]
        h = F.cross_entropy(logits[b, p:p+1].float(), gold[ends], reduction='sum')
        alpha = .5 if 0 in ys else 1.
        # exp(-CE) is stable even when the probability underflows to zero.
        weight = 2. - (-h).exp().detach()
        ordinary.append(alpha*h)
        dynamic.append(alpha*weight*h)
        stats['think'] += 1
        stats['continue'] += ys.count(0)
        stats['end_only'] += int(0 not in ys)
    return (torch.stack(ordinary).sum() if ordinary else zero,
            torch.stack(dynamic).sum() if dynamic else zero, stats)
