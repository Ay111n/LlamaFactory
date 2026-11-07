> 本目录管理 SFT 实验。训练数据和模型归相邻 finetuning，评测归相邻 benchmark。当前路径配置见 [LAYOUT.md](LAYOUT.md)。

# Mixed SFT 动态终止损失小规模实验

历史路径和哈希按生成时的记录解释。

本实验只运行 6,000 条、3 epochs 的 S0/S1/S2/S1-M 及 666 DEV。没有全量训练入口，不使用 1,418 TEST 筛选方法。

## 当前实现

- LLaMA-Factory 实验分支：`codex/mixed-sft-dynamic-termination`，基于 `8cc0d8eb`。
- 新增 `termination_loss_mode`、`termination_loss_coef`、`termination_dataset_path`；未设置实验数据路径时沿用原训练流程。
- `termination_loss.py` 实现逐样本类别平衡与 detached 动态权重；`termination_trainer.py` 实现整个有效批次的 Think 归一化和 S1-M。
- S1-M 的梯度向量用 FP32 在 CPU 累加；保留每个参数的完整向量，在 optimizer step 前做全参数范数匹配，然后裁剪总梯度。没有改成微批范数平均或近似匹配。
- 规则边界来自 spaCy 英文 sentencizer 与换行；不代表语义完整性。原始模板文本获取 offsets，逐样本核对 SFT token IDs/labels，避免藏文 token 解码再编码带来的差异。
- `lm_head` 独立训练，导出时明确关闭重新绑定，并验证保存后重新加载的 logits 一致。

## 环境

按用户要求使用已有 `/opt/anaconda3/envs/training`，Python 已为 3.11.15，未升级 Python。训练环境补充 spaCy 3.8.7；依赖修改前保存 `training_requirements_before.txt`、`training_conda_before.txt`。现有推理环境 `/home/nlp/miniconda3` 用于复用 B0 的 vLLM 推理。

曾创建的临时预处理环境已删除，新的独立训练环境创建已中止。

## 已完成核验

- 49,800 条原始记录与 ShareGPT 转换记录的用户文本、英文 reasoning、最终答案逐条对应通过；来源记录见 `source_provenance.jsonl`。
- 子集 Think/No-think 为 3,183/2,817，最大实际训练长度 4,016，无 4096 截断。
- 规则继续边界候选 305,762 个，映射并去重后保留 219,572 个；排除 86,190 个，原因保存在逐边界审计中。仅结束边界样本为 1 条，其结束项权重为 1。
- 数值、标签、detach、极端值、重复边界与梯度累积单元验证通过。
- 真实 Transformers/Qwen3 小模型训练器集成验证四模式通过，最大梯度误差小于 9e-8；额外验证诊断探针不改变参数、原梯度或 RNG。
- 32 条格式分层样本由 Codex 检查英文 Think、继续边界上下文及真实结束下一 token，详见 `manual_review_results.json`；这是抽样检查，不是所有边界的人类标注。
- 历史配置的实证、一致项、差异和未知项见 `historical_config_comparison.csv`。

## 运行与门禁

2026-10-22 按用户要求改为利用空闲 GPU 并行运行。当前控制器为 `run_parallel_pipeline.py`：接管已在 GPU 0 启动的 S0，在 GPU 6 启动 S1-M、GPU 7 启动 S1，S2 排队；如果其他同型号 L20 完全空闲，也可分配给排队组。各组仍为独立单卡训练，batch 1、累积 16、1,125 步和共同 λ=0.0012 不变，不使用 DDP。分配前同时核对显存、利用率及计算进程，排除其他任务占用的 GPU。大模型 CPU 导出串行执行，以控制主机内存峰值。

原串行控制器已停止，但保留原文件及其冻结哈希；已完成的真实更新验证和正在运行的 S0 均保留。并行授权与新增调度器哈希记录在 `parallel_execution_manifest.json`，逐组 GPU/PID/阶段记录在 `pipeline_status.json`。调度资格判断已通过四种模拟场景验证，见 `parallel_scheduler_validation.json`。

`run_experiment.py calibrate` 在基座上校准共同 λ，不更新参数。校准只控制初始尺度。

`run_pipeline.py` 是保留的原串行流程。现有并行控制器复用相同验证门禁；真实 S1-M 更新验证已通过，耗时约 186 秒，峰值分配显存约 31.0 GiB。某组失败时不再启动新的排队组，其他已运行的独立组可完成并保留结果；不会绕过门禁或自动改实验定义。

正式训练各 1,125 步；每步保存辅助统计，100 步及末步保存固定训练探针梯度统计。状态见 `pipeline_status.json`，具体阶段日志独立保存。

评测逐题 prompt 与历史 B0 哈希对齐，保留原英文指令、贪心、4096 上限及自然 EOS。只添加原始输出 token 证据，不增加强制终止机制。

## 收尾交付

完成四组后生成 `metrics.csv`、`paired_transitions.csv`、`paired_statistics.json`、`historical_cohorts.csv` 和 `report.md`。历史 86/580 为固定子群；恢复与退化主要比较同预算 S0。只有计数门槛通过且配对证据支持时才提出全量建议，流程随后停止。

## 2026-10-22 晚间训练完成检查与导出校验修复

S0、S1、S2 均已完成 1,125 步、3 epochs，逐步记录完整且有限。训练检查表和说明见 `training_review_S0_S1_S2.csv`、`training_review_S0_S1_S2.md`。三组平均 CE 均约 0.780；S1/S2 的训练分布终止 margin 损失降低，但自然闭合能力与准确率仍需 DEV 自由生成结果。

原导出校验的 `model.to(bfloat16)` 把非持久 RoPE 频率缓存一起舍入，而加载模型时会按配置以 FP32 重建这些缓存。小模型已复现该精度差异；只转换参数、保留缓存的回归测试 logits 误差为 0。原冻结脚本、导出权重和失败记录全部保留。

新增 `verify_export_v2.py` 对已有文件逐张量做 bitwise 比较，并在相同 FP32 RoPE 缓存条件下，按原 `rtol=atol=1e-3` 校验重载 logits。S0 的 399 个保存张量已全部逐位匹配；各组必须通过完整校验才启动 DEV。`export_arm_v2.py` 为尚未导出的 S1-M 保留同样的缓存精度处理。

当前控制器为 `recover_eval_pipeline.py`，接管原 S1-M 训练进程，不重训已完成组。它串行执行 CPU 较重的核验，在 GPU 0/1/7 上分别推进三组 DEV；S1-M 仍在 GPU 6。三组 DEV 完成后，`partial_dev_review.py` 会先生成明确标为初步结果的报告；最终四组报告等待 S1-M 完成。新增修复代码的哈希与理由见 `export_recovery_manifest.json`，当前状态见 `recovery_status.json`。没有启动全量训练。

## 评测归属补充（2026-10-22）

历史 B0 逐题预测、manifest、summary 现归 benchmark/eval/results/mixed_sft_dynamic_termination_20261022/historical_b0/。后续各组 DEV 预测和运行清单写入同一 results 工程；分析脚本、指标、配对统计和报告归 benchmark/eval/analysis/mixed_sft_dynamic_termination_20261022/。

训练目录保存训练、导出、校准和集成验证记录。training_review 对应训练步数与损失审计。L20 运行记录显示 S1-M 已按用户要求停止，尚未完成。历史冻结 manifest 与运行状态保留原时点信息。
