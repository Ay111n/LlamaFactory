修正 Q 的全量训练：49,800 条，3 epoch，GPU 1 / GPU 2

用户已明确授权启动全量训练。
学生：GPU 1，sft-new2 全新初始化，LoRA rank64 / alpha128 / dropout0.05，并训练 lm_head。
教师：GPU 2，CE-9000，冻结 FP32；完整词表目标使用此前预检通过的投影。
lambda=mu=0.1，beta=0.5，cap=2；学生使用原 BF16 / FlashAttention2 / 梯度检查点。
原49,800条输入、标签和混合比例保持；固定1000条诊断集不参与梯度更新。
每卡microbatch1、累积16，有效批量16；3 epoch预计9,339步；学习率1e-4，cosine，warmup5%。

保存策略
在1500、3000、4500、6000、7500、9000和最终9339步保存完整checkpoint。
保存adapter、optimizer、scheduler、RNG和trainer_state；保留全部7个checkpoint。
模型输出：/home/nlp/data/lyl/finetuning/model/sft_experiments/gold_reasoning_full_49800_20261103/main
不载入CE学生或50步预检adapter；没有自动DEV/TEST评测。

代码和记录
本目录：/home/nlp/data/lyl/LLaMA-Factory/sft_experiments/gold_reasoning_full_49800_20261103
本地：E:\毕业论文\LlamaFactory\sft_experiments\gold_reasoning_full_49800_20261103
训练解释器：/opt/anaconda3/envs/training/bin/python
损失和Trainer直接导入相邻gold_reasoning_preflight_20261103中的已验证实现，不修改该实现。
full_training.yaml、protocol.json、input_manifest.json、source_manifest.json为本次配置与来源。
process.json记录后台进程；training.log、steps.jsonl、trainer_logs.jsonl、progress.json记录运行。
checkpoints.jsonl在每次完整写盘后记录；completion.json只在9339步和全部checkpoint完成后写入。
failure.json / launch_failure.json / process_exit.json用于失败或退出诊断；不会自动重启或覆盖结果。

启动与查看
prepare_full.py进行来源核验，launch_full.py启动脱离SSH会话的后台监督进程。
status.py只读查看进度。使用ssh L20和上述training解释器运行。
launch_full.py拒绝重复启动；指定GPU被占用时停止启动，不终止其他进程。
全量时间按预检均速约103小时纯训练，另计保存、评测和导出。
