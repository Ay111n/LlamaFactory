"""Offline target-probability weighting audit. No torch, model, inference or training."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_rows(path, data):
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8")


def quantiles(x):
    return {str(p): float(np.quantile(x, p)) for p in [0, .1, .25, .5, .75, .9, .95, .99, 1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    inp, out = args.input_dir.resolve(), args.output_dir.resolve()
    assert inp != out and not out.exists(), "Preserve existing diagnostics; use a new output directory"
    manifest = read(inp / "sha256_manifest.json")
    for name, digest in manifest.items():
        assert sha(inp / name) == digest, name
    selected = read(inp / "selected.json")
    assert len(selected) == len({r["uid"] for r in selected}) == 32
    tok = Tokenizer.from_file(str(args.tokenizer))
    records = []
    for sample_index, r in enumerate(selected):
        n, uid = r["body_tokens"], r["uid"]
        answer = r["pair"]["tibetan"]["answer"]
        gold = r["student"]["meta"]["target"][:n]
        encoded = tok.encode(answer, add_special_tokens=False)
        assert encoded.ids == gold
        assert [list(x) for x in encoded.offsets] == r["offsets"]
        with np.load(inp / "tokens" / (uid + "_baseline.npz"), allow_pickle=False) as b, np.load(inp / "tokens" / (uid + "_teacher.npz"), allow_pickle=False) as q:
            lp, lq = b["baseline_logp"][:n].astype(np.float64), q["plus_logp"][:n].astype(np.float64)
            assert np.isfinite(lp).all() and np.isfinite(lq).all()
            assert np.max(lp) <= 0 and np.max(lq) <= 0
            p, qt = np.exp(lp), np.exp(lq)
            one_minus_p = -np.expm1(lp)
            # Avoid cancellation when both cached probabilities are close to one.
            delta = qt * (-np.expm1(np.minimum(lp - lq, 0)))
            a = np.divide(delta, one_minus_p, out=np.zeros(n), where=one_minus_p > 0)
            assert np.all((a >= 0) & (a <= 1 + 1e-12))
            a = np.clip(a, 0, 1)
            assert np.max(np.abs(a * one_minus_p - delta)) < 1e-12
            for t in range(n):
                lo, hi = r["offsets"][t]
                records.append({"id": uid + ":" + str(t), "sample_index": sample_index, "uid": uid, "t": t,
                    "gold_id": gold[t], "source_chars": answer[lo:hi], "decoded_token": tok.decode([gold[t]], skip_special_tokens=False),
                    "context": answer[max(0, lo-100): min(len(answer), hi+140)], "char_range": [lo, hi],
                    "p": float(p[t]), "q": float(qt[t]), "a": float(a[t]), "delta": float(delta[t]), "ce": float(-lp[t]),
                    "baseline_logit_l1": float(2*one_minus_p[t]), "raw_added_logit_l1_lambda1": float(2*delta[t]),
                    "baseline_top3": [{"id": int(j), "text": tok.decode([int(j)], skip_special_tokens=False), "p": float(v)} for j,v in zip(b["baseline_top_ids"][t],b["baseline_top_probs"][t])],
                    "teacher_top3": [{"id": int(j), "text": tok.decode([int(j)], skip_special_tokens=False), "p": float(v)} for j,v in zip(q["plus_top_ids"][t],q["plus_top_probs"][t])],
                    "reference_gain_logp": float(lq[t]-q["empty_logp"][t]),
                    "whitespace_only": bool(answer[lo:hi]) and answer[lo:hi].isspace()})
    a = np.array([r["a"] for r in records])
    ce = np.array([r["ce"] for r in records])
    p = np.array([r["p"] for r in records])
    q = np.array([r["q"] for r in records])
    g = np.array([r["baseline_logit_l1"] for r in records])
    lam_stats = {}
    for lam in [.2, 1., 2.]:
        w = (1 + lam*a)/(1 + lam*a.mean())
        net_positive = np.maximum(w-1,0)*g
        order = np.argsort(-a, kind="stable")
        m = int(np.ceil(.1*len(a)))
        lam_stats[str(lam)] = {"weight_quantiles": quantiles(w), "weight_cv": float(w.std()/w.mean()),
            "fraction_upweighted_after_normalization": float(np.mean(w>1)),
            "fraction_within_10pct_of_one": float(np.mean(np.abs(w-1)<=.1)),
            "top10pct_weight_share": float(w[order[:m]].sum()/w.sum()),
            "baseline_ce_token_mean": float(ce.mean()), "weighted_ce_token_mean": float(np.mean(w*ce)),
            "sum_logit_l1_ratio": float(np.sum(w*g)/np.sum(g)),
            "positive_logit_l1_change_over_baseline": float(net_positive.sum()/g.sum()),
            "negative_logit_l1_change_over_baseline": float(np.sum(np.maximum(1-w,0)*g)/g.sum())}
        if lam == 1:
            for i,r in enumerate(records):
                r.update(weight=float(w[i]), positive_net_logit_l1=float(net_positive[i]), net_ce_change=float((w[i]-1)*ce[i]))
    sample_summary, peaks = [], []
    for i,r in enumerate(selected):
        rr = [x for x in records if x["uid"] == r["uid"]]
        local_w = np.array([x["weight"] for x in rr])
        sample_summary.append({"uid":r["uid"], "sample_index":i,"tokens":len(rr),"mean_a":float(np.mean([x["a"] for x in rr])),
            "mean_weight":float(local_w.mean()), "baseline_ce":float(np.mean([x["ce"] for x in rr])),
            "weighted_ce":float(np.mean([x["ce"]*x["weight"] for x in rr]))})
        for score, group in [("a","max_weight_per_sample"),("positive_net_logit_l1","max_net_boost_per_sample")]:
            peaks.append({**max(rr,key=lambda x:x[score]), "review_group": group})
    positive_order = sorted(records, key=lambda r:(-r["positive_net_logit_l1"],r["id"]))
    high_net = positive_order[:int(np.ceil(.1*len(records)))]
    peak_ids={r["id"] for r in peaks if r["review_group"]=="max_net_boost_per_sample"}
    pool=sorted([r for r in high_net if r["id"] not in peak_ids],key=lambda r:hashlib.sha256(("weight-review-3407:"+r["id"]).encode()).hexdigest())
    review=[r for r in peaks if r["review_group"]=="max_net_boost_per_sample"] + [{**r,"review_group":"top10pct_net_boost_hash24"} for r in pool[:24]]
    hist=rows(inp/"semantic_review.jsonl")
    windows={r["id"]:r for r in rows(inp/"windows.jsonl")}
    historical=[]
    for h in hist:
        win=windows[h["window_id"]]
        rr=[r for r in records if r["uid"]==win["uid"] and win["start"]<=r["t"]<win["end"]]
        historical.append({**h,"tokens":len(rr),"mean_weight":float(np.mean([r["weight"] for r in rr])),
            "positive_net_logit_l1":float(sum(r["positive_net_logit_l1"] for r in rr)),
            "baseline_logit_l1":float(sum(r["baseline_logit_l1"] for r in rr)),
            "note_scope":"Historical window labels concerned earlier log-probability gates; not new token labels."})
    historical_groups={}
    for group in sorted({r["primary_gain_interpretation"] for r in historical}):
        rr=[r for r in historical if r["primary_gain_interpretation"]==group]
        historical_groups[group]={"windows":len(rr),"tokens":sum(r["tokens"] for r in rr),
            "token_mean_weight":sum(r["mean_weight"]*r["tokens"] for r in rr)/sum(r["tokens"] for r in rr),
            "net_boost_over_baseline_logit_l1":sum(r["positive_net_logit_l1"] for r in rr)/sum(r["baseline_logit_l1"] for r in rr)}
    sorted_weight=sorted(records,key=lambda r:(-r["a"],r["id"]))[:int(np.ceil(.1*len(records)))]
    summary={"samples":32,"body_tokens":len(records),"active_raw_fraction":float(np.mean(a>0)),"mean_a":float(a.mean()),"a_quantiles":quantiles(a),
        "p_exact_one_count":int(np.sum(p==1)),"q_exact_one_count":int(np.sum(q==1)),
        "top10pct_weight_p_gt_099_fraction":float(np.mean([r["p"]>.99 for r in sorted_weight])),
        "top10pct_weight_net_boost_share":sum(r["positive_net_logit_l1"] for r in sorted_weight)/sum(r["positive_net_logit_l1"] for r in records),
        "top10pct_net_boost_share":sum(r["positive_net_logit_l1"] for r in high_net)/sum(r["positive_net_logit_l1"] for r in records),
        "whitespace_only_token_count":sum(r["whitespace_only"] for r in records),
        "whitespace_only_net_boost_share":sum(r["positive_net_logit_l1"] for r in records if r["whitespace_only"])/sum(r["positive_net_logit_l1"] for r in records),
        "lambda_sensitivity":lam_stats,"sample_mean_weight_range":[min(r["mean_weight"] for r in sample_summary),max(r["mean_weight"] for r in sample_summary)],
        "historical_window_groups":historical_groups,
        "sample_equal_baseline_ce":float(np.mean([r["baseline_ce"] for r in sample_summary])),
        "sample_equal_weighted_ce":float(np.mean([r["weighted_ce"] for r in sample_summary]))}
    out.mkdir(parents=True)
    save(out/"protocol.json",{"script_sha256":sha(Path(__file__)),"source_dir":str(inp),"input_hashes":manifest,"tokenizer_sha256":sha(args.tokenizer),
        "python":sys.executable,"numpy":np.__version__,"primary_lambda":1,"sensitivity_lambdas":[.2,2],
        "normalization":"Token mean over this fixed 32-response body-token batch; not future minibatch simulation",
        "student":"Completed ordinary bilingual 10k SFT endpoint, not initial sft-new2",
        "teacher":"Existing frozen crossref teacher with English reference",
        "formula":"a=max(q-p,0)/(1-p), w=(1+lambda*a)/mean(1+lambda*a), stop-gradient assumed",
        "numerics":"float64; delta=q*(-expm1(min(logp-logq,0))); denominator=-expm1(logp); p=1 implies q<=p and a=0; saturation remains limited by cache precision",
        "logit_proxy":"For CE, per-position logit gradient L1=2(1-p); positive net change=2*max(w-1,0)*(1-p). No parameter gradient or optimizer analysis.",
        "review_selection":"32 per-response maxima of positive net logit boost; plus fixed-hash 24 from global top10% by boost, excluding those maxima. Purposive diagnostic review, not population precision.",
        "no_model_load":True,"no_inference":True,"no_training":True,"no_parameter_update":True})
    save_rows(out/"tokens.jsonl",records)
    save_rows(out/"sample_summary.jsonl",sample_summary)
    save_rows(out/"peaks.jsonl",peaks)
    save_rows(out/"review_candidates.jsonl",review)
    save_rows(out/"historical_windows.jsonl",historical)
    save(out/"summary.json",summary)
    for name,digest in manifest.items(): assert sha(inp/name)==digest,name
    save(out/"verification.json",{"source_manifest_files_unchanged":len(manifest),"tokenizer_exact_targets_and_offsets":32,"finite_bounded_a":True,"gradient_factor_identity_max_error_lt":1e-12,"tokens":len(records),"review_positions":len(review)})
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__ == "__main__":
    main()
