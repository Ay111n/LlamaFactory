# SFT 实验工程

本目录管理实验代码、调度和训练诊断。数据及模型由相邻 finetuning 管理，评测输入、预测与分析归相邻 benchmark。具体路径由所选配置中的 framework_root、data_root、model_root 和 benchmark_root 决定。

Windows 默认读取本目录的 paths.local.json，Linux 默认读取 paths.json。可用 SFT_EXPERIMENT_PATHS 显式选择配置。外部处理脚本使用 SFT_EXPERIMENT_ROOT 定位本工程。缺少模型、数据或环境时报告实际缺项。

历史 manifest、输入哈希和运行状态按原时点解释。训练和评测是否通过，以相应运行的记录为准。
