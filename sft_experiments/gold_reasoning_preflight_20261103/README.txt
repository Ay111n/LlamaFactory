本实验只实现并运行短程预检，不授权或自动启动全量训练。

位置
本地：E:\毕业论文\LlamaFactory\sft_experiments\gold_reasoning_preflight_20261103
L20：/home/nlp/data/lyl/LLaMA-Factory/sft_experiments/gold_reasoning_preflight_20261103
衍生输入：/home/nlp/data/lyl/finetuning/sft/processed_data/gold_reasoning_preflight_20261103
预检模型：/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_preflight_20261103

训练合同
学生从 sft-new2 全新初始化，沿用原 LLaMA-Factory 配置：
LoRA rank64、alpha128、dropout0.05、all targets，额外完整训练 lm_head；
BF16、FlashAttention2、梯度检查点、microbatch1、累积16、学习率1e-4、
cosine、warmup_ratio0.05、seed/data_seed3407、cutoff4096、3 epoch。
学习率日程保持完整 9339 步的跨度，由回调在第50步硬停止。
仅在 train 参数 --steps 显式设置为100时允许100步，入口没有全量运行模式。

教师 CE-9000 在独立 L20 上冻结，以 FP32/SDPA 前向，TF32 关闭。
学生放逻辑 cuda:0，教师放逻辑 cuda:1。不是两进程 DDP。
所有调用使用 /opt/anaconda3/envs/training/bin/python，无环境升级或混用。

损失
保留 native LLaMA-Factory 标准 CE，包括完整 assistant 中原有的所有监督。
Think 的回答正文额外计算 KL(Q* || student_plus) 和 KL(teacher_zero || student_zero)。
lambda=mu=0.1，beta=0.5，截断c=2；完整161280词表，不使用top-k近似。
Q* 的投影与已通过的固定1000条诊断相同：gold与任一候选的相对优势不低于普通教师。
Q* 依赖冻结教师的有/无英文推理分布和当前位置gold，gold不作为当前位置预测的输入。
额外损失先在每条样本回答token上平均，再在当前累积窗口全部样本上平均；
no_think贡献零。CE仍按原native监督token数归约，所有分支经过相同Accelerator.backward缩放。
先完成有推理分支的反向，再运行无推理锚定，避免同时保留两个学生计算图。
分块KL在backward重算冻结目标，不保存全样本的两份教师词表分布。

数据
读取原49,800条ShareGPT数据和已核验CE预处理结果，逐条重新分词核对全部input_ids及labels。
只添加运行时有/无推理视图的索引。固定1000条Think诊断集按UID和问题文本哈希排除。
原始文件、诊断集和历史模型保持不变。

入口
prepare_data.py：CPU数据核对和衍生视图准备。
test_losses.py：独立排序解法验证投影，并与dense autograd比较分块KL数值及梯度。
run_preflight.py validate：真实8B、全835190784个可训练参数的CE/梯度等价检查，0次更新。
run_preflight.py train --steps 50：真实50次优化器更新，要求前置检查通过。
launch_preflight.py：仅在指定两张卡空闲时依次运行validate和50步train；不会终止其他进程。
summarize_preflight.py：汇总已完成步骤的标量损失、显存、每步耗时及全量时间估计。

证据
本次活动运行是 attempt02；完成后的入口为 LATEST.json。
data_manifest.json、unit_checks.json、provenance.json，以及 attempt02/ 下面的
ce_equivalence.json、steps.jsonl、long_sequence_memory.json、completion.json、summary.json、report.txt。
attempt01保留首次容差检查失败记录，未执行优化器更新，不是已完成的训练预检。
一致性检查开启确定性FlashAttention，并包括native CE自身重复对照；训练计时使用原默认设置。
长序列探针在50步之后运行，保留真实优化器状态，仅反向传播、不更新参数。
预检adapter仅作技术验证；未来主实验需要从sft-new2重新初始化。
CE一致性范围为当前可恢复的native训练栈，不能据此声称历史运行软件逐字节一致。
耗时估计不包含DEV/TEST评测、后续服务器争用、全部检查点写盘及模型导出。
手动调用分析脚本时显式使用 --run-id attempt02；重做预检应使用新的run-id，已有步骤文件拒绝覆盖。
