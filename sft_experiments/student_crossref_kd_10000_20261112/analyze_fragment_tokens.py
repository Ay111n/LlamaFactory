"""CPU-only secondary summaries of frozen fragment-probe arrays."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/"runtime/diagnostics/fragment_value_20261112"
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 rows=read(OUT/"selected.json")
 wins=[json.loads(x) for x in (OUT/"windows.jsonl").read_text(encoding="utf-8").splitlines()]
 records=[];ces={x:[] for x in ["baseline","plus","empty","natural"]};full_ce=[]
 for r in rows:
  uid=r["uid"];n=r["body_tokens"];y=np.array(r["student"]["meta"]["target"])[:n]
  b=np.load(OUT/"tokens"/(uid+"_baseline.npz"));q=np.load(OUT/"tokens"/(uid+"_teacher.npz"))
  full_ce.append(float(-b["baseline_logp"].mean()))
  mask=np.zeros(n,dtype=bool)
  for w in wins:
   if w["uid"]==uid and w["R"]>.1 and w["G"]>.1:mask[w["start"]:w["end"]]=True
  for mode in ces:ces[mode].append(float(-(b if mode=="baseline" else q)[mode+"_logp"][:n].mean()))
  for t in range(n):records.append({"group":"selected" if mask[t] else "not_selected","teacher_gold":bool(q["plus_top_ids"][t,0]==y[t]),"baseline_gold":bool(b["baseline_top_ids"][t,0]==y[t]),"same_top1":bool(q["plus_top_ids"][t,0]==b["baseline_top_ids"][t,0]),"joint_positive":bool(q["plus_logp"][t]-q["empty_logp"][t]>.1 and q["plus_logp"][t]-b["baseline_logp"][t]>.1)})
 result={"body_ce_sample_mean":{k:float(np.mean(v)) for k,v in ces.items()},"R_G_window_correlation":float(np.corrcoef([w["R"] for w in wins],[w["G"] for w in wins])[0,1]),"groups":{},"token_level_joint_positive_count":sum(r["joint_positive"] for r in records),"body_tokens":len(records),"baseline_all_target_ce":float(np.mean(full_ce)),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 for label in ["selected","not_selected"]:
  rs=[x for x in records if x["group"]==label]
  result["groups"][label]={"tokens":len(rs),**{k:sum(x[k] for x in rs)/len(rs) for k in ["teacher_gold","baseline_gold","same_top1","joint_positive"]},"teacher_top1_gold_baseline_not":sum(x["teacher_gold"] and not x["baseline_gold"] for x in rs),"baseline_top1_gold_teacher_not":sum(x["baseline_gold"] and not x["teacher_gold"] for x in rs)}
 # Prior completed run, used only to check numerical consistency across execution environments.
 result["baseline_all_target_ce_difference_from_previous_gradient_diagnostic"]=result["baseline_all_target_ce"]-1.2841292107477784
 (OUT/"token_diagnostics.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
