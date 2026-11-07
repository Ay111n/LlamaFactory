"""Summarize existing training-probe gradients; no DEV/TEST or optimization."""
import argparse,csv,json,math,datetime,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['G1','G2','G3','G1-M','G3-M']
FIELDS=['continue_loss','stop_loss','aux_loss','effective_continue_loss','effective_stop_loss','continue_grad_norm','stop_grad_norm','aux_actual_grad_norm','aux_components_sum_norm','ce_grad_norm',
 'continue_stop_norm_ratio','cos_continue_stop','cos_continue_ce','cos_stop_ce','cos_aux_ce','cos_aux_plain_stop_ce',
 'continue_projection_share_sum','stop_projection_share_sum','continue_projection_share_actual','stop_projection_share_actual',
 'rounding_residual_projection_share_actual','sum_vs_original_relative_l2_error','cos_sum_aux_actual','cancellation_factor',
 'probe_match_coefficient','effective_scale','effective_continue_norm','effective_stop_norm','effective_aux_norm','effective_continue_R','effective_stop_R','effective_aux_R',
 'relative_residual_after_best_fit_stop_ce','best_fit_stop_ce_multiplier','lambda_only_continue_R','lambda_only_stop_R','lambda_only_aux_R','interaction_energy_fraction']

def read(path):return [json.loads(line) for line in path.open(encoding='utf-8')]
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dist(values):
 x=np.asarray([v for v in values if v is not None],dtype=float)
 if len(x)==0:return {'n':0,'mean':None,'median':None,'min':None,'p10':None,'p90':None,'max':None}
 assert np.isfinite(x).all()
 return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)),max=float(x.max()))
def f(v,n=3):return 'NA' if v is None else f'{v:.{n}g}'

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostic-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--historical-records-root',type=Path);args=p.parse_args()
 root=args.diagnostic_root;out=args.output;out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for path in sorted((root/'raw').glob('*.jsonl')):
  if '_cross' not in path.name:rows+=read(path)
 assert len(rows)==80 and len({(r['stage'],r['arm'],r['window_index']) for r in rows})==80
 order={(stage,arm):i for i,(stage,arm) in enumerate((s,a) for s in ['initial','final'] for a in ARMS)}
 rows.sort(key=lambda r:(order[r['stage'],r['arm']],r['window_index']))
 for r in rows:
  for part,key in [('continue','continue_grad_norm'),('stop','stop_grad_norm'),('aux','aux_actual_grad_norm')]:r['lambda_only_'+part+'_R']=r['lambda']*r[key]/r['ce_grad_norm']
  nc,ns=r['continue_grad_norm'],r['stop_grad_norm']
  r['interaction_energy_fraction']=2*nc*ns*r['cos_continue_stop']/(nc*nc+ns*ns) if nc*ns else 0.
 for stage in ['initial','final']:
  for arm in ARMS:assert [r['window_index'] for r in rows if r['stage']==stage and r['arm']==arm]==list(range(8))
 sources=[json.loads(p.read_text()) for p in (root/'sources').glob('*.json')]
 assert len(sources)==6 and all(s['parameters_unchanged'] and s['no_optimizer_created'] and s['no_training_call'] for s in sources)
 assert len({json.dumps(s['parameter_layout']) for s in sources})==1,'Parameter-coordinate layouts differ'
 stats={};summary=[]
 for stage in ['initial','final']:
  for arm in ARMS:
   block=[r for r in rows if r['stage']==stage and r['arm']==arm]
   detail={key:dist([r[key] for r in block]) for key in FIELDS};stats[stage+'/'+arm]=detail
   nt=sum(r['window_think'] for r in block)
   summary.append(dict(stage=stage,arm=arm,windows=8,Think_occurrences=nt,
    continue_loss_window_mean=detail['continue_loss']['mean'],stop_loss_window_mean=detail['stop_loss']['mean'],
    continue_loss_Think_pooled=sum(r['continue_loss']*r['window_think'] for r in block)/nt,
    stop_loss_Think_pooled=sum(r['stop_loss']*r['window_think'] for r in block)/nt,
    continue_norm_mean=detail['continue_grad_norm']['mean'],stop_norm_mean=detail['stop_grad_norm']['mean'],aux_norm_mean=detail['aux_actual_grad_norm']['mean'],
    norm_ratio_median=detail['continue_stop_norm_ratio']['median'],stop_projection_actual_mean=detail['stop_projection_share_actual']['mean'],
    stop_projection_actual_min=detail['stop_projection_share_actual']['min'],stop_projection_actual_max=detail['stop_projection_share_actual']['max'],
    cos_continue_stop_mean=detail['cos_continue_stop']['mean'],cos_continue_ce_mean=detail['cos_continue_ce']['mean'],cos_stop_ce_mean=detail['cos_stop_ce']['mean'],
    cos_aux_stop_ce_mean=detail['cos_aux_plain_stop_ce']['mean'],effective_continue_R_mean=detail['effective_continue_R']['mean'],effective_stop_R_mean=detail['effective_stop_R']['mean'],effective_aux_R_mean=detail['effective_aux_R']['mean'],
    match_coefficient_median=detail['probe_match_coefficient']['median'],sum_vs_original_error_max=detail['sum_vs_original_relative_l2_error']['max'],
    min_cos_sum_actual=detail['cos_sum_aux_actual']['min'],negative_continue_stop_windows=sum(r['cos_continue_stop'] is not None and r['cos_continue_stop']<0 for r in block),negative_continue_CE_windows=sum(r['cos_continue_ce'] is not None and r['cos_continue_ce']<0 for r in block)))
 for s in summary:
  block=[r for r in rows if r['stage']==s['stage'] and r['arm']==s['arm']]
  for name,effective in [('raw',False),('effective',True)]:
   weights=np.array([r['aux_actual_grad_norm']**2*(r['effective_scale']**2 if effective else 1.) for r in block])
   for part in ['continue','stop','rounding_residual']:
    s[name+'_energy_'+part+'_projection']=float(np.dot(weights,[r[part+'_projection_share_actual'] for r in block])/weights.sum())
   s[name+'_largest_window_energy_fraction']=float(weights.max()/weights.sum())
   s[name+'_top2_window_energy_fraction']=float(np.sort(weights)[-2:].sum()/weights.sum())
  a=np.array([r['aux_actual_grad_norm'] for r in block]);b=np.array([r['plain_balanced_stop_ce_grad_norm'] for r in block]);cos=np.array([r['cos_aux_plain_stop_ce'] for r in block])
  s['raw_stacked_aux_plain_stop_ce_cosine']=float(np.dot(a*b,cos)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
 scalars=['arm','stage','window_index','analysis_seed','window_ce_tokens','window_think','microbatches']+FIELDS
 write_csv(out/'per_window_components.csv',[{**{key:r[key] for key in scalars},'continue_boundary_count':r['counts']['continue'],'end_only_sample_count':r['counts']['end_only']} for r in rows]);write_csv(out/'stage_summary.csv',summary)
 (out/'component_distributions.json').write_text(json.dumps(stats,indent=2,allow_nan=False))
 paired=[]
 lookup_rows={(r['stage'],r['arm'],r['window_index']):r for r in rows}
 for stage in ['initial','final']:
  for i in range(8):
   a=lookup_rows[stage,'G1',i];b=lookup_rows[stage,'G2',i]
   paired.append({'stage':stage,'window_index':i,'G2_minus_G1_stop_projection_pp':100*(b['stop_projection_share_actual']-a['stop_projection_share_actual']),
    'G2_over_G1_continue_stop_norm_ratio':b['continue_stop_norm_ratio']/a['continue_stop_norm_ratio'] if a['continue_stop_norm_ratio'] else None,
    'G2_minus_G1_continue_CE_cosine':b['cos_continue_ce']-a['cos_continue_ce'] if b['cos_continue_ce'] is not None and a['cos_continue_ce'] is not None else None})
 write_csv(out/'dynamic_vs_plain_paired.csv',paired)
 cross=read(root/'raw/initial_pair_cross.jsonl')+read(root/'raw/final_pair_cross.jsonl')
 lookup={(r['stage'],r['arm'],r['window_index']):r for r in rows}
 for r in cross:
  for arm,name in [('G2','G2_effective_aux_norm'),('G3-M','G3M_effective_aux_norm')]:r[name]=lookup[r['stage'],arm,r['window_index']]['effective_aux_norm']
 cross_summary={}
 for stage in ['initial','final']:
  block=[r for r in cross if r['stage']==stage]
  a=np.array([r['G2_effective_aux_norm'] for r in block]);b=np.array([r['G3M_effective_aux_norm'] for r in block]);c=np.array([r['cos_cross_model_aux'] for r in block])
  cross_summary[stage]={'median_window_cosine':float(np.median(c)),'effective_stacked_gradient_cosine':float(np.dot(a*b,c)/np.sqrt(np.dot(a,a)*np.dot(b,b)))}
 (out/'cross_model_summary.json').write_text(json.dumps(cross_summary,indent=2))
 write_csv(out/'G2_G3M_cross_model.csv',[{k:r.get(k) for k in ['stage','window_index','cos_cross_model_aux','cos_cross_model_continue','cos_cross_model_stop','G2_effective_aux_norm','G3M_effective_aux_norm']} for r in cross])

 # Historical full-objective records are a consistency reference, not missing components.
 exp=root.parents[1];cal=json.loads((exp/'gold_calibration.json').read_text())['rows']
 paths=json.loads((exp/'paths.json').read_text());hist={}
 for arm in ARMS:
  record_root=args.historical_records_root if args.historical_records_root else Path(paths['model_root'])/'runs'
  hist[arm]={r['window_index']:r for r in read(record_root/arm/'gold_gradient_probes.jsonl') if r['step']==1125}
 comparison=[]
 objective={'G1':'gold','G2':'dynamic_gold','G3':'dynamic_ce','G1-M':'gold','G3-M':'dynamic_ce'}
 for r in rows:
  h=cal[r['window_index']] if r['stage']=='initial' else hist[r['arm']][r['window_index']]
  old=h[objective[r['arm']]+'_norm']
  comparison.append(dict(stage=r['stage'],arm=r['arm'],window_index=r['window_index'],recomputed_aux_norm_FP64=r['aux_actual_grad_norm'],historical_aux_norm=old,relative_norm_difference=(r['aux_actual_grad_norm']-old)/old if old else None,recomputed_CE_norm_FP64=r['ce_grad_norm'],historical_CE_norm=h['ce_norm']))
 write_csv(out/'historical_total_comparison.csv',comparison)

 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
 for ax,stage in zip(axes,['initial','final']):
  for j,arm in enumerate(ARMS):
   values=np.asarray([r['continue_stop_norm_ratio'] for r in rows if r['stage']==stage and r['arm']==arm])
   ax.scatter(j+np.linspace(-.12,.12,8),np.maximum(values,1e-12),s=25,alpha=.65,color='#47789d')
   ax.scatter(j,max(float(np.median(values)),1e-12),s=70,marker='_',color='#a54141')
  ax.set_yscale('log');ax.axhline(1,color='#666',lw=.8);ax.axhline(.1,color='#aaa',lw=.8,ls=':')
  ax.set_xticks(range(5),ARMS);ax.set_title(stage);ax.set_xlabel('Training objective')
 axes[0].set_ylabel('Continue / stop gradient L2 norm')
 fig.tight_layout();fig.savefig(out/'gradient_norm_ratios.png',dpi=180);fig.savefig(out/'gradient_norm_ratios.svg');plt.close(fig)

 fig,ax=plt.subplots(figsize=(10,6))
 labels=[]
 for i,s in enumerate(summary):
  key=s['stage']+'/'+s['arm'];labels.append(key);pos=neg=0.
  for field,color,title in [('continue_projection_share_actual','#4c83a6','Continue'),('stop_projection_share_actual','#d79948','Stop'),('rounding_residual_projection_share_actual','#888888','BF16 residual')]:
   value=100*stats[key][field]['mean'];left=pos if value>=0 else neg
   ax.barh(i,value,left=left,color=color,label=title if i==0 else None)
   if value>=0:pos+=value
   else:neg+=value
 ax.set_yticks(range(10),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.7);ax.axvline(100,color='#999',lw=.7,ls=':')
 ax.set_xlabel('Mean signed projection onto ORIGINAL auxiliary gradient (%)');ax.legend(loc='best');fig.tight_layout()
 fig.savefig(out/'signed_projection_contributions.png',dpi=180);fig.savefig(out/'signed_projection_contributions.svg');plt.close(fig)

 fig,ax=plt.subplots(figsize=(9,4.5))
 final_summary=[s for s in summary if s['stage']=='final'];x=np.arange(5)
 ax.bar(x-.24,[100*s['stop_projection_actual_mean'] for s in final_summary],width=.24,label='Equal-window mean',color='#4c83a6')
 ax.bar(x,[100*s['raw_energy_stop_projection'] for s in final_summary],width=.24,label='Gradient-energy weighted, before matching',color='#d79948')
 ax.bar(x+.24,[100*s['effective_energy_stop_projection'] for s in final_summary],width=.24,label='Energy weighted after lambda*c',color='#678b71')
 ax.set_xticks(x,ARMS);ax.set_ylabel('Stop signed projection share (%)');ax.axhline(100,color='#999',lw=.7,ls=':');ax.legend(fontsize=8);fig.tight_layout()
 fig.savefig(out/'window_vs_energy_contributions.png',dpi=180);fig.savefig(out/'window_vs_energy_contributions.svg');plt.close(fig)

 names=['cos_continue_stop','cos_continue_ce','cos_stop_ce','cos_aux_plain_stop_ce']
 values=np.array([[stats[s['stage']+'/'+s['arm']][k]['median'] for k in names] for s in summary],dtype=float)
 fig,ax=plt.subplots(figsize=(8,6));im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu')
 ax.set_yticks(range(10),labels);ax.set_xticks(range(4),['C vs S','C vs CE','S vs CE','Aux vs stop CE'])
 for i in range(10):
  for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',color='white' if abs(values[i,j])>.65 else 'black')
 fig.colorbar(im,ax=ax,label='Median cosine over eight fixed windows');fig.tight_layout()
 fig.savefig(out/'gradient_direction_cosines.png',dpi=180);fig.savefig(out/'gradient_direction_cosines.svg');plt.close(fig)

 fig,ax=plt.subplots(figsize=(7,4))
 for i,key in enumerate(['cos_cross_model_continue','cos_cross_model_stop','cos_cross_model_aux']):
  for stage,offset,color in [('initial',-.14,'#47789d'),('final',.14,'#d79948')]:
   vals=[r[key] for r in cross if r['stage']==stage and r[key] is not None]
   ax.scatter(np.full(len(vals),i+offset),vals,color=color,s=24,alpha=.65,label=stage if i==0 else None)
   if vals:ax.plot([i+offset-.07,i+offset+.07],[np.median(vals)]*2,color='black',lw=1.3)
 ax.set_xticks(range(3),['Continue','Stop','Combined aux']);ax.set_ylim(-1.05,1.05);ax.axhline(0,color='#999',lw=.7)
 ax.set_ylabel('G2 / G3-M cosine (aligned parameter coordinates)');ax.legend();fig.tight_layout();fig.savefig(out/'G2_G3M_cross_model_cosines.png',dpi=180);plt.close(fig)

 report=['# 继续边界与真实结束位置：离线梯度归因','',
  '本轮只重算现有模型的训练探针前向/VJP；未创建 optimizer、未更新参数、未生成答案、未读取 DEV/TEST。原损失实现未修改。',
  '共同初始状态复用一次前向得到五种目标；最终使用各自 checkpoint-1125。仅保留最终权重，没有可用中间权重，历史总梯度记录不被当作分项梯度。','',
  '## 主要统计','',
  '|阶段/方法|继续损失均值|结束损失均值|继续/结束梯度范数中位数|继续投影均值|结束投影均值|继续-结束余弦均值|继续-CE余弦均值|辅助-仅结束CE余弦均值|',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for s in summary:report.append('|'+s['stage']+'/'+s['arm']+'|'+'|'.join([f(s['continue_loss_window_mean']),f(s['stop_loss_window_mean']),f(s['norm_ratio_median']),f"{100*stats[s['stage']+'/'+s['arm']]['continue_projection_share_actual']['mean']:.2f}%",f"{100*s['stop_projection_actual_mean']:.2f}%",f(s['cos_continue_stop_mean']),f(s['cos_continue_ce_mean']),f(s['cos_aux_stop_ce_mean'])])+'|')
 report+=['','投影针对独立原始合并 VJP：p_c=<g_c,g_aux>/||g_aux||²，p_s 同理。另单列数值残差方向贡献 1−p_c−p_s。分项之和方向上的投影也保存在逐窗口表中。负贡献或大于 100% 的贡献原样保留，不作截断。','',
  '## 损失、范数与合并贡献的区别','']
 report.append('cancellation_factor=||g_c+g_s||/(||g_c||+||g_s||)；正交向量也会使它小于 1，不能把 1−该值称为抵消百分比。逐窗口另存 2<g_c,g_s>/(||g_c||²+||g_s||²)，其负值才表示负交互能量。')
 for s in summary:
  key=s['stage']+'/'+s['arm'];v=stats[key]
  report.append(f"- {key}：结束投影均值 {100*s['stop_projection_actual_mean']:.2f}%（范围 {100*s['stop_projection_actual_min']:.2f}%～{100*s['stop_projection_actual_max']:.2f}%）；继续/结束范数中位数 {f(s['norm_ratio_median'])}。继续与结束方向冲突 {s['negative_continue_stop_windows']}/8 窗口，继续与全 token CE 方向冲突 {s['negative_continue_CE_windows']}/8。")
 report+=['','## 动态权重与匹配控制','',
  '初始状态共享相同参数与前向，所以初始 G1/G2 的差异直接反映 detached 动态权重；最终不同模型之间同时存在训练轨迹差异，不将其全部归因为权重本身。',
  '匹配组使用当前固定探针窗口重新计算 c=||g_G2||/||g_candidate||，同时报告匹配前与 λc 缩放后范数和相对 CE 的强度。这里的 c 是只读反事实探针值，不能冒充历史某个 optimizer update 的系数。正标量缩放不改变方向、范数比或投影份额。',
  '匹配只在候选自身参数上成立，不意味着与实际 G2 训练路径的梯度逐步相同。','',
  '|阶段/方法|c 中位数|λc·‖g_c‖/‖g_CE‖均值|λc·‖g_s‖/‖g_CE‖均值|λc·‖g_aux‖/‖g_CE‖均值|',
  '|---|---:|---:|---:|---:|']
 for s in summary:report.append('|'+s['stage']+'/'+s['arm']+'|'+'|'.join(f(s[k]) for k in ['match_coefficient_median','effective_continue_R_mean','effective_stop_R_mean','effective_aux_R_mean'])+'|')
 for stage in ['initial','final']:
  block=[r for r in paired if r['stage']==stage]
  report.append(f"\n{stage} 同批配对：G2−G1 的结束投影变化均值 {f(dist([r['G2_minus_G1_stop_projection_pp'] for r in block])['mean'])} 个百分点；继续/结束范数比的倍数中位数 {f(dist([r['G2_over_G1_continue_stop_norm_ratio'] for r in block])['median'])}。初始比较控制了参数与 dropout；最终比较含训练轨迹差异。")
 report+=['','## 每批平均与梯度能量汇总','',
  '每批投影均值对 8 个窗口等权。为避免少数大梯度窗口被掩盖，另计算 Σ<g_s,i,g_a,i>/Σ||g_a,i||²；匹配后用各窗口 (λc_i)² 加权分子和分母。这是在独立窗口梯度的直和空间中汇总能量，不是把 8 批当成一个 optimizer batch，也不是 Adam 更新贡献。',
  '|最终方法|结束投影：窗口等权|结束投影：匹配前能量汇总|结束投影：λc后能量汇总|继续投影：λc后能量汇总|最大单窗口能量占比（λc后）|',
  '|---|---:|---:|---:|---:|---:|']
 for s in final_summary:
  report.append('|'+s['arm']+'|'+'|'.join(f'{100*s[k]:.2f}%' for k in ['stop_projection_actual_mean','raw_energy_stop_projection','effective_energy_stop_projection','effective_energy_continue_projection','effective_largest_window_energy_fraction'])+'|')
 report+=['', '因此，不能把“G3/G3-M 多数普通批次有较大的继续贡献”概括成整体能量由继续项主导；少数结束梯度较大的窗口可能主导能量。两种口径必须并列阅读。匹配正标量不改变单窗口投影，但会改变窗口之间的能量权重。']
 report+=['','## G2 与 G3-M 的结构','']
 for stage in ['initial','final']:
  block=[r for r in cross if r['stage']==stage]
  report.append(f"- {stage}：继续、结束、合并辅助梯度的跨模型余弦中位数分别为 "+' / '.join(f(dist([r[k] for r in block])['median']) for k in ['cos_cross_model_continue','cos_cross_model_stop','cos_cross_model_aux'])+'。')
  report.append(f"  按 λc 后梯度幅度加权的直和空间余弦为 {f(cross_summary[stage]['effective_stacked_gradient_cosine'])}；该值强调高能窗口，与每窗口余弦中位数回答不同问题。")
 report+=['','最终跨模型余弦使用同名同形状的 LoRA/lm_head 坐标；参数值不同。相似的贡献结构或梯度方向均不等于生成行为相同。','',
  '## 是否主要是强化真实结束 CE','',
  'Gold Margin 的真实结束项在数学上等于 stop token CE；动态目标对它再乘 detached 权重。本报告额外计算同类别平衡和 Think 分母下、不加动态权重的真实结束 CE 梯度，作为独立方向参照。辅助与该参照的余弦、最佳标量拟合残差均已保存，而非只比较损失数值。']
 finals=[s for s in summary if s['stage']=='final'];gold=[s for s in finals if s['arm'] in ['G1','G2','G1-M']]
 dominant=[s['arm'] for s in gold if s['stop_projection_actual_mean']>=.8 and s['cos_aux_stop_ce_mean']>=.9 and s['sum_vs_original_error_max']<.02]
 report+=['',f'按透明的描述性参考（结束投影均值≥80%、辅助与仅结束 CE 余弦均值≥0.9、分解数值误差<2%），最终 Gold 目标中满足的组别：{dominant}。这些不是方法优劣或统计显著性阈值。','', '## 下一步建议','']
 if len(dominant)>=2:
  recommendation='值得优先加入“仅真实 </think> 位置 CE 加权”的简化消融，检验当前复杂目标的大部分效果是否能由结束 CE 解释；在行为对照完成前，不据局部梯度直接删除继续边界。继续保留完整边界版本作对照。'
 else:
  recommendation='目前梯度证据不足以把完整监督视为仅结束 CE 的替代，应保留继续边界；仅结束 CE 仍可作为后续必要消融，但不能据本轮直接取代完整目标。'
 report.append(recommendation)
 best=next(s for s in finals if s['arm']=='G3-M')
 report.append(f"G3-M 的继续项实际方向投影均值为 {100*(1-best['stop_projection_actual_mean']-stats['final/G3-M']['rounding_residual_projection_share_actual']['mean']):.2f}%，与 CE 的余弦均值为 {f(best['cos_continue_ce_mean'])}。若该贡献不可忽略，仍应保留完整继续边界版本，而非把 Gold 目标的结论推广到边界 CE。")
 report.append('后续若做仅结束 CE 消融，应保持结束项原有类别系数、Think 分母及可比梯度强度，避免删除继续边界后自动把结束项从 0.5 改为 1 而引入强度混淆。本轮不执行该训练。')
 report.append('这里的“仅结束 CE”仅指简化辅助项；原始全有效 token 的 SFT CE 仍应保留，No-think 样本规则不变。')
 report+=['','## 数值核验与适用范围','']
 max_error=max(s['sum_vs_original_error_max'] for s in summary)
 min_cos=min(s['min_cos_sum_actual'] for s in summary)
 report.append(f'分项和相对独立原始合并 VJP 的最大 L2 误差为 {f(max_error,5)}，最小方向余弦为 {f(min_cos,6)}。FP32 单元测试验证了严格分解；实际 BF16 反向中的独立 VJP 可能有舍入残差。不能把残差误称为任务梯度抵消。')
 if max_error>=.02:report.append('**部分 BF16 数值残差超过 2%；对应窗口的精细贡献判断需谨慎，不宣称分项与原始更新精确等价。**')
 report+=['',
  '全部分项保留原 0.5/0.5 类别平衡；仅结束边界权重为 1。CE 以整窗口有效 shifted token 数归一化，辅助以整窗口 Think 数归一化，微批累积后不再除以 16。范数和点积采用 FP64 归约，梯度向量保持 FP32。',
  '8 个窗口的输入、labels 和 mask 哈希与五组历史固定探针一致。最终 checkpoint 的 CPU/CUDA RNG 状态在五组间相同，按窗口恢复并在计算后还原；保持原训练模式 dropout。初始状态使用原 seed-3407 LF 初始化后的 CUDA 状态。',
  '历史 step-1125 探针在最后一次 optimizer 更新前，现有最终 checkpoint 在更新后；历史前三组的范数还使用了旧 FP32 归约。historical_total_comparison.csv 因此只作一致性参考，不预设逐值相等。',
  '每组的 8 个窗口是固定诊断集合，不是随机总体样本，也不是多种子重复。这里报告经验分布与方向结构，不将组间差异解释为总体统计显著或生成因果效应。',
  '探针使用训练数据的金标前缀（teacher forcing），不覆盖自由生成中的错误前缀；继续项在这些探针上弱，不等于它在所有生成状态都无用。',
  '损失中的规则边界包含格式 token；不把全部继续边界称为语义正文。全部梯度覆盖原全部 835,190,784 个可训练参数。',
  '本轮仅完成离线归因。没有新训练、参数更新、DEV/TEST 筛选或生成调用。']
 (out/'report.md').write_text('\n'.join(report),encoding='utf-8')
 result={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':80,'stages':['initial','final'],'arms':ARMS,'parameters_unchanged_all':True,'optimizer_updates':0,'DEV_or_TEST_used':False,'max_component_sum_relative_error':max_error,'min_sum_original_cosine':min_cos,'gold_stop_dominant_groups_by_descriptive_reference':dominant,'recommendation':recommendation,'diagnostic_root':str(root),'source_manifest_sha256':hashlib.sha256((root/'protocol_manifest.json').read_bytes()).hexdigest()}
 (out/'completed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
