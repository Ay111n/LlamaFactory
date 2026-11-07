跨语言推理损失的吞吐优化诊断，2026-11-03

当前状态：原全量训练已按用户要求停止；本目录只包含有步数上限的对照和诊断，不恢复全量训练。
实际结论见 throughput_report.txt 和 comparison.json；原始记录保存在 equivalence/、baseline/、cached/、pipeline/。

实现
optimized.py 直接导入相邻 gold_reasoning_preflight_20261103 中冻结的 ReferenceTargets、projected_log_probs、answer_kl、ReasoningTrainer。
CachedTargets 复用同一条样本的 FP32、完整词表 Q 与无推理教师分布。
PipelineTargets 使用一个后台线程和独立 CUDA 流，最多预取下一条 Think 样本；教师保持冻结和 eval。
PipelineTrainer 只调整准备与执行的调度，沿用原 CE、KL、梯度缩放和窗口归一化。
没有量化教师、降低参考计算精度、截断词表、改变样本顺序或更换数据。

验证
equivalence：真实8B模型、全部835190784个可训练参数的梯度对照；确定性FA2；最长序列、最长答案与纯no_think窗口；零优化器更新。
baseline/cached/pipeline：每版从sft-new2全新初始化，同样12个累积批次，每个16条，原9339步日程；前3步排除，后9步比较吞吐。
每版末尾用已有真实优化器状态检查最长输入与最长答案，只反向、不更新。
component_profile：独立的4样本CUDA分析，含profiling开销，不用于上面吞吐加速比。
gpu_samples.jsonl：独立低频利用率采样。

GPU和解释器
所有实际测试只使用物理GPU1（学生）和GPU2（教师），通过UUID绑定。
解释器固定 /opt/anaconda3/envs/training/bin/python。
远程目录 /home/nlp/data/lyl/LLaMA-Factory/sft_experiments/gold_reasoning_throughput_20261103。
本地目录 E:\毕业论文\LlamaFactory\sft_experiments\gold_reasoning_throughput_20261103。

边界
4卡双学生/双教师方案仅为设计建议，尚未实现或实测。
如继续开发，需要处理两条反向分支的跨rank同步、全局token/sample归一化、固定有效批量16和有界显存。
当前入口仅供有界诊断；恢复全量训练需单独接入验证后的实现和原检查点保存策略。
原全量训练停止记录在相邻 gold_reasoning_full_49800_20261103/user_stop.json。
