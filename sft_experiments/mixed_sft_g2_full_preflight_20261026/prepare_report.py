"""Finalize a blocked preflight package; contains no training entrypoint."""
import csv,hashlib,inspect,json,os,platform,shutil
from collections import Counter
from pathlib import Path
import yaml,torch,transformers,accelerate,peft,spacy
from transformers import AutoConfig,AutoModelForCausalLM,Trainer
from accelerate import Accelerator
from peft import LoraConfig,get_peft_model

R=Path(os.environ['G2_PREFLIGHT_ROOT']);P=json.loads((R/'paths.json').read_text());A=R/'diagnostics/preflight_analysis';D=Path(P['data'])
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def table(path,rows):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 A.mkdir(parents=True,exist_ok=True)
 ce=json.loads((R/'ce_equivalence.json').read_text());confirm=json.loads((R/'ce_scale_confirmation.json').read_text());data=json.loads((D/'preparation_manifest.json').read_text())
 assert not ce['gate_pass'],'This stopped-report route is specifically for the failed CE gate'
 # Shape-only meta construction: no base tensor loading, forward or update.
 cfg=AutoConfig.from_pretrained(P['base'],local_files_only=True)
 with torch.device('meta'):
  base=AutoModelForCausalLM.from_config(cfg)
  names=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and not n.endswith('lm_head')]
  targets=sorted({n.rsplit('.',1)[-1] for n in names})
  model=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=64,lora_alpha=128,lora_dropout=.05,target_modules=targets,modules_to_save=['lm_head']))
 layout=[(n,list(p.shape)) for n,p in model.named_parameters() if p.requires_grad]
 count=sum(p.numel() for p in model.parameters() if p.requires_grad)
 reference=Path(P['framework'])/'sft_experiments/mixed_sft_gold_margin_20261023/diagnostics/gradient_attribution_20261025/sources/final_G2.json'
 old=json.loads(reference.read_text())
 expected=[(n,list(shape)) for n,shape in old['parameter_layout']]
 layout_same=sorted(layout)==sorted(expected)
 model_report={'gate_pass_static':layout_same and count==835190784,'basis':'shape-only meta model from current original YAML base/LoRA and saved small G2 actual parameter layout',
  'trainable_count':count,'trainable_tensors':len(layout),'target_modules':targets,'modules_to_save':['lm_head'],
  'same_names_shapes_as_small_G2':layout_same,'reference':str(reference),'real_full_model_forward_backward':'not run after failed CE gate'}
 (A/'trainable_parameter_comparison.json').write_text(json.dumps(model_report,indent=2))
 table(A/'trainable_parameters.csv',[{'name':n,'shape':str(s),'numel':int(__import__('math').prod(s)),'match_small_G2':(n,s) in expected} for n,s in layout])
 # Keep the original YAML untouched. A blocked configuration cannot start training.
 original=yaml.safe_load(Path(P['yaml']).read_text());proposed=dict(original)
 proposed.update(do_train=False,output_dir=P['model_output'],data_seed=3407,optim='adamw_torch',packing=False,train_on_prompt=False,
                 termination_loss_mode='dynamic_gold_margin',termination_loss_coef=.0012,termination_dataset_path=str(D/'dataset'))
 name=R/'g2_full_49800.BLOCKED.yaml'
 name.write_text('# BLOCKED: CE lambda=0 gradient mismatch; do_train=false intentionally. Do not launch full training.\n'+yaml.safe_dump(proposed,allow_unicode=True,sort_keys=False))
 diff=[{'key':k,'original_yaml':json.dumps(original.get(k),ensure_ascii=False),'prepared':json.dumps(proposed.get(k),ensure_ascii=False),
        'status':'unchanged' if original.get(k)==proposed.get(k) else 'explicit addition' if k not in original else 'changed / blocked'} for k in sorted(set(original)|set(proposed))]
 table(A/'configuration_comparison.csv',diff)
 group_rows=[]
 for key,v in data['domain_length_groups'].items():group_rows.append({'group':key,**v})
 table(A/'domain_length_groups.csv',group_rows)
 stats=Counter();zero_letters=[];fine_groups={}
 for line in (D/'sample_audit.jsonl').open():
  s=json.loads(line);key='think' if s['think'] else 'no_think';stats[key+'_ce_tokens']+=s['ce_tokens']
  if s['think'] and s['reasoning_ascii_letters']==0:zero_letters.append(s['source_index'])
  classification=s.get('mmlu_classification') or {}
  category=classification.get('category','unclassified') if isinstance(classification,dict) else str(classification)
  subject=classification.get('subject','unclassified') if isinstance(classification,dict) else 'unclassified'
  length=s['length'];bucket='0-1024' if length<=1024 else '1025-2048' if length<=2048 else '2049-3072' if length<=3072 else '3073-4096'
  for group in [str(s['domain'])+'/'+category,str(s['domain'])+'/'+category+'/'+subject,str(s['domain'])+'/'+category+'/'+bucket]:
   v=fine_groups.setdefault(group,Counter());v['count']+=1;v['think']+=s['think'];v['continue_boundaries']+=s['continue_boundaries'];v['end_only']+=s['end_only']
 table(A/'fine_domain_length_groups.csv',[{'group':k,**dict(v)} for k,v in sorted(fine_groups.items())])
 details={'effective_ce_tokens_by_type':dict(stats),'nonempty_reasoning_without_ascii_letters_indices':zero_letters,
          'note':'ASCII letters are descriptive, not a semantic/language classifier; no sample is removed'}
 (A/'additional_data_statistics.json').write_text(json.dumps(details,indent=2))
 source={'native_training_step':inspect.getsource(Trainer.training_step),'native_item_count':inspect.getsource(Trainer._get_num_items_in_batch),
         'accelerator_backward':inspect.getsource(Accelerator.backward)}
 (R/'ce_source_evidence.json').write_text(json.dumps(source,indent=2))
 assets=[Path(P['yaml']),Path(P['train']),Path(P['provenance']),D/'source_metadata.jsonl',D/'raw_source_manifest.json',D/'sample_audit.jsonl',D/'boundary_audit.jsonl']
 assets+=list((D/'dataset').glob('*'))
 assets+=[Path(P['base'])/n for n in ['config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja']]
 assets+=[Path(P['framework'])/'src/llamafactory/train/sft'/n for n in ['termination_gold.py','gold_termination_trainer.py','termination_trainer.py','termination_boundaries.py','trainer.py']]
 assets+=list(R.glob('*.py'))+[R/'paths.json',name,Path(P['data_scripts'])/'audit_data.py']
 fingerprints={'files':{str(p):sha(p) for p in assets if p.is_file()},'runtime':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
  'accelerate':accelerate.__version__,'peft':peft.__version__,'spacy':spacy.__version__},'base_weights_rescanned':False,
  'source_original_yaml_untouched':True,'optimizer_updates':0,'full_training_started':False,'historical_checkpoint_section_cancelled_by_user':True}
 (R/'input_fingerprints.json').write_text(json.dumps(fingerprints,indent=2));(A/'input_fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
 protocol={'candidate_selection':'manual','evaluate_retained_checkpoints_on_666_DEV':True,'evaluate_now':False,
           'historical_checkpoint_audit':False,'historical_checkpoint_re_evaluation':False,'automatic_checkpoint_selection':False,
           'TEST_model_inference':False,'full_training_authorized':False,'future_save_steps':1500,'future_save_total_limit':10,
           'future_max_steps_from_3_epochs':9339,'future_selection_rule':'list results and let user choose; no automatic best checkpoint'}
 (R/'future_checkpoint_policy.json').write_text(json.dumps(protocol,indent=2))
 gates={'status':'blocked_ce_gradient_mismatch','can_start':False,'data_and_boundaries':data['gate_pass'],
        'ce_lambda0_loss_and_gradient_equivalence':False,'static_parameter_layout':model_report['gate_pass_static'],
        'real_full_model_gradient_probes':'not_run_due_to_failed_CE_gate','original_checkpoint_section':'cancelled_by_user',
        'formal_training_started':False,'optimizer_updates':0,'explanation':'Native Accelerator.backward divides by16, old G2 uses raw backward; loss values match but parameter gradients do not'}
 (R/'gates.json').write_text(json.dumps(gates,indent=2));(A/'gates.json').write_text(json.dumps(gates,indent=2))
 c=data['counts'];ov=data['overlap_summary'];examples=confirm['reports']
 lines=['# G2 全量 Mixed SFT 训练前检查：未通过，禁止启动','',
 '按最新指示，本轮取消原始checkpoint及历史评测协议检查。未来对训练保留的checkpoint在666 DEV逐一列结果，由用户人工挑选；不设置自动选择规则。',
 '', '## 1. 关键门禁：原生CE与旧G2的梯度不一致','',
 '检查的是当前training环境、当前LLaMA-Factory源码及指定原始YAML；不把当前还原结果反推为历史运行时的确定事实。',
 '在小型Qwen3+PEFT模型上，保持同一初始参数、输入、监督mask和RNG，实际调用原生CustomSeq2SeqTrainer.training_step与G2(λ=0)的training_step；这不是8B实模型测试，没有optimizer创建或更新。9个场景全部出现同一差异：loss相同，G2梯度为原生梯度的16倍。当前源码中的额外÷16与模型尺寸无关；发现该问题后按要求停止，不再用8B前反向重复此已确认差异。',
 '', '|窗口|原生loss|G2 λ=0 loss|G2/原生梯度范数比|梯度余弦|G2梯度÷16后最大误差|', '|---|---:|---:|---:|---:|---:|']
 for x in examples:lines.append(f"|{x['n']}条，Think={x['think']}|{x['native_loss']:.9f}|{x['g2_lambda0_loss']:.9f}|{x['g2_to_native_gradient_norm_ratio']:.9f}|{x['gradient_cosine']:.9f}|{x['maximum_error_after_dividing_g2_by_16']:.3g}|")
 lines+=['', '原生模型接受num_items_in_batch，因此CE标量已按整个窗口有效token数N归一化；Trainer将其交给Accelerator.backward，当前Accelerate仍除以gradient_accumulation_steps=16。旧G2的TerminationTrainer直接loss.backward，跳过这次缩放。两条路径的标量相同，不能据loss相同判定训练梯度公平。',
 '实际数据所有prompt首token均为忽略标签，因此原生统计全部有效labels与G2统计有效shifted labels的分母在本数据上相同；这里不是监督mask或token分母差异造成的16倍。尾部8条窗口也复现16倍，因为Accelerator的插件配置仍为16。',
 '此前小规模S0/G2共用自定义窗口反向路径，其内部对照不因这一发现自动失效；但不能据此宣称它们与原生CE全量训练路径已严格一致。',
 '', '## 2. 公平比较的处理方案（本轮不实施）','',
 '优先方案：仅在新的全量G2入口中复用原生CE计算与反向缩放，包括尾部窗口；让CE+λ辅助总目标共同经过原生Accelerator.backward。保留Gold Margin公式、detached权重、0.5/1结束系数、Think分母及λ=0.0012。先重新验证λ=0损失和梯度，再做真实模型固定窗口诊断。',
 '不能仅把λ除以16，因为这无法修复CE梯度差异；不能用学习率缩放声称等价，AdamW、裁剪和数值状态会受影响。另一方案是CE/G2两臂统一数学token平均并重新建立全量CE对照，但会更换比较口径，需要独立决定与授权。',
 '按“发现影响公平比较的问题时停止”要求，本轮未修改任何旧trainer、原始YAML、历史结果或λ；未运行真实全模型梯度探针，也未启动训练。',
 '', '## 3. 数据与边界','',
 f"实际ShareGPT共{data['rows']}条，Think {c['think']}、NoThink {c['no_think']}。分类来自非空reasoning正文，并逐条核对原始reasoning哈希；不是根据是否存在结束标记判断。原始本地数据与实际训练ShareGPT的UID映射、问题、reasoning及答案哈希通过。",
 f"原处理器全量input_ids/labels/mask校验通过：完整序列{data['length_min']}–{data['length_max']} tokens，截断{c['truncated']}条；全部prompt掩码正确，全部有效目标继续保留标准CE。Think均具有唯一、有效且受监督的真实结束位置，stop ID=151668；NoThink的空Think包装仍参与原CE，不增加辅助项。",
 f"沿用spaCy 3.8.7与原规则函数，共{c['candidate_continue']:,}个继续候选，保留{c['continue_boundaries']:,}，排除{c['excluded_continue']:,}；仅结束边界Think为{c['end_only']}条，结束系数为1。排除原因逐位置保存，不删除任何样本。与原6000条缓存的input_ids/labels/mask/positions/targets/source_index全部相同，差异0条。",
 f"有效CE token共{c['ce_tokens']:,}；其中Think {stats['think_ce_tokens']:,}、NoThink {stats['no_think_ce_tokens']:,}。Math/Knowledge及长度组见domain_length_groups.csv，更细MMLU category/subject与长度交叉分组见fine_domain_length_groups.csv；逐条记录及全部边界保存在finetuning缓存目录。",
 '', '## 4. 只读重复排查与覆盖边界','',
 f"49,800个训练UID唯一；DEV666、TEST1418仅用于只读重复排查，未做模型推理、方法筛选或参数调整。当前规则成功解析{ov['parsed_choice_train']:,}条训练选择题，其余{ov['unparsed_or_non_mcqa_train']:,}条为非该选项格式或非选择题，未把它们当作已完成题干/选项排查。",
 '在成功解析范围内，没有UID、标准化题干或题干+选项联合命中；仅选项集合相同的DEV有11题、TEST有26题，其中各1题选项顺序也相同。仅选项碰撞可能来自常见数字或通用选项，不足以认定题目污染；具体命中已列出，未修改或删除样本。',
 '标准化采用NFKC、casefold、去空白和Cf字符，仅移除确实观察到的两行固定指令前缀。末行题干命中仅作为候选，不把字符串相同或无命中扩大为语义同题/总体无泄漏结论。对未解析样本的完整题干排查仍有覆盖缺口。',
 '', '## 5. 配置、参数与待完成项','',
 f"当前原始YAML解析得到：seed3407、data_seed=None、AdamW Torch、cosine、warmup0.05、BF16、FA2、梯度检查点开启，use_reentrant_gc=True、gradient_checkpointing_kwargs=None。当前SeedableRandomSampler在N=49800时，data_seed省略与显式3407的完整索引顺序一致：{ce['sampler']['omitted_vs_explicit_3407_equal']}。这不证明历史运行时版本完全相同。",
 f"原始lora_target=all在当前基座上展开为七类投影。形状级meta构造得到{count:,}个可训练参数、{len(layout)}个可训练张量，含lm_head；与已验证小规模G2记录的实际名称/形状完全一致：{layout_same}。这是静态核验，未将其冒充完整实模型前反向验收。",
 '不同长度下逐样本平均(Σ sample_mean /B)与整窗口token平均(Σ token_CE /N)一般不相同；本轮构造的16条窗口分别为2.776366与2.777066，8条窗口为2.775079与2.777493。原生路径当前的CE标量属于后者，但反向仍另有上述÷16缩放。',
 '原始检查点配置另有兼容性风险：原生默认reentrant=True，而旧G2的周期探针使用autograd.grad；此组合需先验证/适配。可使用支持reentrant的backward探针并保存恢复已有梯度和RNG；不能未经说明把原梯度检查点配置改成False。该后续适配本轮因主门禁失败而未实施。',
 '已准备g2_full_49800.BLOCKED.yaml，保持49,800条、原顺序规则、batch1/累积16、3epochs（当前实现推导9339步）、LR1e-4、rank64/alpha128/dropout0.05、lm_head、每1500步保存及limit10。do_train=false是主动禁用，不是可启动正式配置。未改变原始YAML。',
 '真实模型固定领域/长度/Think比例窗口的梯度R、余弦及继续/结束分项：NOT RUN（被CE门禁阻止）；不会用小模型或旧小规模探针冒充本轮全量实模型诊断。',
 '', '## 结论','',
 '1. 公平CE训练路径：未实现当前原生路径的梯度等价，不能启动。',
 '2. 全量数据与终止边界：全部通过；重复排查仅完成声明的规则覆盖范围。',
 '3. 原始checkpoint与历史评测：按用户最新指示取消；未来仅列出保留checkpoint的666 DEV结果供人工选择。',
 '4. 正式训练门禁：CLOSED。需要先处理CE反向缩放口径，重新通过λ=0测试，完成reentrant兼容及真实全模型固定窗口诊断；不能标记“可以启动”。']
 (A/'report.md').write_text('\n'.join(lines),encoding='utf-8')
 print(json.dumps(gates),flush=True)
if __name__=='__main__':main()
