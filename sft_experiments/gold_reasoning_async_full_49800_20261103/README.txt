缓存＋异步预取全量训练：49,800 条，3 epoch，GPU 1 / GPU 2

用户已明确授权启动。学生从sft-new2全新初始化，教师为冻结的CE-9000。
使用相邻gold_reasoning_throughput_20261103/optimized.py中的已验证PipelineTrainer和PipelineTargets，不复制或修改损失实现。
有效批量16，LoRA rank64/alpha128/dropout0.05，并训练lm_head；BF16学生、FP32教师；lambda=mu=0.1、beta=0.5、cap=2。
缓存当前样本的完整词表目标，最多异步预取下一条Think样本；样本顺序和损失归一化沿用预检配置。
原49,800条数据；固定1,000条诊断集不参与梯度更新。未加载短测adapter，未续接已停止的19步运行。

3 epoch共9,339步。保存1500、3000、4500、6000、7500、9000、9339，保留全部7份。
每份checkpoint包含adapter（含训练的lm_head）、optimizer、scheduler、RNG和trainer_state。
每个更新步及保存时检查教师冻结和预取队列清空；训练正常结束或异常退出时关闭后台预取线程。
没有预检步数停止条件；不自动启动DEV/TEST评测。

代码/日志：/home/nlp/data/lyl/LLaMA-Factory/sft_experiments/gold_reasoning_async_full_49800_20261103
模型：/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_async_full_49800_20261103/main
本地：E:\毕业论文\LlamaFactory\sft_experiments\gold_reasoning_async_full_49800_20261103
解释器：/opt/anaconda3/envs/training/bin/python
状态：ssh L20 '/opt/anaconda3/envs/training/bin/python /home/nlp/data/lyl/LLaMA-Factory/sft_experiments/gold_reasoning_async_full_49800_20261103/status.py'

按吞吐短测25.45秒/步外推，纯训练约66小时（2.75天），保存耗时另计；实时速度以服务器steps.jsonl为准。
teacher_seconds是等待预取结果的时间，不是教师总计算时间。
prepared.json、input_manifest.json和source_manifest.json记录实际输入及代码核验；process.json记录脱离SSH会话的后台进程。
startup_verified为启动检查快照，之后状态以服务器实时日志为准。
