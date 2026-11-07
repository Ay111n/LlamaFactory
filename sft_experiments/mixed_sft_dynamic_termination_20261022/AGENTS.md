# SFT 实验子工程

- 实验代码、配置、调度、测试和诊断归此目录；核心 trainer/loss 复用本训练框架的 src/，不复制实现。
- 数据和模型处理脚本归 finetuning；评测实现及结果归 benchmark。统一通过 experiment_paths.py 路由。
- 新数据写 data_root，模型及原生 checkpoint 状态写 model_root；运行日志写实验目录，评测日志与结果写 benchmark。
- 冻结历史证据不改写。迁移后未实测的训练链路不得标为通过；不得自动选择服务器路径或启动实验。
