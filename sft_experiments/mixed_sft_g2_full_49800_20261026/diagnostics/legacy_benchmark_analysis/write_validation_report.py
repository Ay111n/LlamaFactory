"""Summarize saved validation evidence; never loads a model or changes training."""
import csv,json,sys
from pathlib import Path

def main(root):
 r=Path(root);paths=json.loads((r/'paths.json').read_text());out=Path(paths['analysis_root'])
 unit=json.loads((r/'unit_gate.json').read_text());cases=json.loads((r/'unit_gradient_equivalence.json').read_text())
 ce=json.loads((r/'real_ce_equivalence.json').read_text());grad=json.loads((r/'initial_gradients.json').read_text());update=json.loads((r/'real_update.json').read_text())
 env=json.loads((r/'environment_evidence.json').read_text());gates=json.loads((r/'gates.json').read_text());selection=json.loads((Path(paths['data_root'])/'probe_selection.json').read_text())
 records=[]
 for row,spec in zip(grad['rows'],selection):records.append(dict(window=spec['name'],length_min=spec['length_min'],length_max=spec['length_max'],**row))
 with (out/'initial_probe_components.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 for name in ['unit_gate.json','unit_gradient_equivalence.json','unit_probe_invariance.json','real_ce_equivalence.json','real_checkpointing.json','initial_gradients.json','real_update.json','gates.json','environment_evidence.json','protocol.json']:
  (out/name).write_bytes((r/name).read_bytes())
 text=['# Full-G2 原生训练路径修复与启动验收','',
 '门禁已通过，仅启动一组Full-G2。没有重新训练标准CE，没有启动其他全量方法，没有使用TEST。后续只对1500/3000/4500/6000/7500/9000步的保留checkpoint执行原666 DEV评测，按步数列出，交由用户人工选择。','',
 '## 修复范围','',
 '新增实验专用FullG2Trainer，继承原生CustomSeq2SeqTrainer。原生compute_loss提供唯一的标准CE及整窗口有效token分母；同一次前向的logits调用未改动的gold_boundary_losses，取dynamic_gold并按整个窗口Think数归一化。CE与λ辅助项作为共同目标，交给继承的原生training_step和Accelerator.backward；没有直接调用raw loss.backward，也没有额外手动÷16，更没有将λ单独除以16。',
 '原始G2公式、显式detach、继续/结束0.5平衡及仅结束系数1、λ=0.0012保持不变；NoThink仅参与原CE。所有有效监督token继续参加CE。旧Trainer、原始YAML与历史实验结果保持原样，其指纹在冻结门禁中核验。',
 '配置沿用原始YAML。必要接口差异为新输出目录、只读引用已逐条校验的全量Arrow缓存、G2模式/系数、保留边界元数据的remove_unused_columns=false。显式data_seed3407与当前原生省略data_seed的索引顺序已验证一致；显式AdamW Torch与原始YAML当前解析结果一致。dataloader workers4、use_reentrant_gc=True及gradient_checkpointing_kwargs=None均保留，未沿用小规模实验的非reentrant覆盖。',
 '', '## 梯度一致性与reentrant门禁','',
 f"FP32小型Qwen3+PEFT共{len(cases)}个对照场景：完整16条窗口、Think数0/1/8/15/16、尾部8条与3条窗口，分别覆盖reentrant开启/关闭、λ=0与λ=0.0012。新路径与原生CE或原生缩放下CE+λG2的逐参数最大误差为{max(x['max_parameter_gradient_error'] for x in cases):.3g}；共同梯度裁剪结果亦一致。旧Gold公式、detach、极端logits和4096边缘测试复用通过。",
 '探针使用支持reentrant的Accelerator.backward来计算VJP，保留并恢复每个已有.grad对象、值、模块training标记及RNG，不调用optimizer.step。非零已有梯度、参数值、CPU RNG、Python/NumPy RNG和模式恢复测试通过。不是通过禁用原生reentrant来规避兼容问题。',
 f"真实8B的同一16条混合有效窗口：原生CE loss={ce['native_loss']:.12f}，新G2 λ=0 loss={ce['full_g2_lambda0_loss']:.12f}；监督token及前反向后的模型CUDA RNG一致。逐参数最大绝对梯度误差{ce['max_parameter_gradient_error']:.8g}，通过预设rtol={ce['rtol']}、atol={ce['atol']}。BF16/FA2验收是在容差内一致，不主张逐位完全相同。两侧Accelerator累积缩放均为16。",
 '', '## 全量数据代表窗口的初始梯度','',
 '|窗口|微批数|Think数|长度范围|R=λ‖g_aux‖/‖g_CE‖|继续项loss|结束项loss|', '|---|---:|---:|---|---:|---:|---:|']
 for x in records:text.append(f"|{x['window']}|{x['microbatches']}|{x['window_think']}|{x['length_min']}–{x['length_max']}|{x['R']:.6g}|{x['continue_loss']:.6g}|{x['stop_loss']:.6g}|")
 text+=['',f"R的P90={grad['R_distribution']['p90']:.6g}，最大值={grad['R_distribution']['max']:.6g}，门禁通过。固定窗口含全NoThink、Think数1/8/15/16、Math、长序列（含实际最长4062）及实际尾部8条，属于预设代表性诊断，不是总体梯度分布。全部835,190,784个可训练参数参与范数和方向统计；参数名/形状与已验证小规模G2一致。",
 '继续/结束分项均来自原始G2目标的掩码VJP，保留类别系数和动态权重。CSV另列各自梯度范数、与CE的余弦、两项余弦、有符号投影和独立合并VJP残差。BF16独立反向存在舍入，残差单列，不冒充精确可加。范数已包含原生共同反向缩放；R因两项共同缩放而不受该常数影响。',
 '', '## 真实更新与正式初始化','',
 f"一次真实有效批次更新验证通过，LoRA B采样位置实际变化。原配置warmup的初始LR为0，因此仅验证模型将调度器定位到同一9339步计划的首个正LR（位置1，{update['scheduled_positive_lr']:.10g}），执行一次optimizer更新后丢弃全部模型/优化器状态；没有重新选择学习率或warmup。正式训练另启进程，重新加载原始藏文SFT基座、重设seed3407，按原生计划从LR0开始，未延续验证更新。",
 '', '## 历史环境证据边界','',
 '当前验收环境：'+json.dumps(env['current']['packages'],ensure_ascii=False)+'。',
 '历史Mixed SFT的完整运行时依赖、代码版本及原生训练状态未能完整恢复；既有配置核验记录也明确标注这些项未核实。当前训练环境快照不能反推为历史运行时。本次严格对齐的是当前可还原的原始YAML及原生LLaMA-Factory路径，不宣称历史每一步数值行为已全部证明一致。该不确定性按用户要求保留，没有重训标准CE来替代证据。',
 '', '## 训练、保存与后续DEV','',
 '49,800条原ShareGPT语义输入及原顺序，Qwen3/4096，rank64/alpha128/dropout0.05，七类投影和lm_head，batch1/累积16，3epochs共9339步，LR1e-4/AdamW Torch/cosine/warmup0.05/BF16/FA2，使用原生checkpointing与裁剪。每1500步保存、limit10，保证六个指定checkpoint保留。原生末尾的必要最终训练状态保留用于审计/恢复，不自动扩充预定选择范围。',
 '训练逐更新记录CE、继续/结束辅助loss、共同总loss、Think/边界数、监督token数、来源索引和裁剪前梯度范数。完整梯度向量诊断在训练前及最终保存后运行；最终诊断只释放已保存的内存优化器状态，不修改已落盘checkpoint或训练轨迹。',
 '训练完成后，以保持RoPE buffer精度的两阶段logits导出验证和一次导出文件SHA清单核验六个候选；随后复用原666 DEV英文指令、贪心、seed42、max_tokens4096及答案提取，不强制结束Think。所有候选按步数列自然闭合、思考截断、抽取率、正确数/666与准确率，不自动选best。',
 '', '状态入口：训练工程pipeline_status.json；本报告只证明启动门禁通过，不代表9339步或DEV已完成。所有新产物使用mixed_sft_g2_full_49800_20261026独立目录。']
 (out/'validation_report.md').write_text('\n'.join(text),encoding='utf-8')
 print('VALIDATION_REPORT_WRITTEN')
if __name__=='__main__':main(sys.argv[1])
