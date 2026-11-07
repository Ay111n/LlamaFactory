# mixed_sft_g2_full_49800_20261026 训练诊断

自定义训练记录归本目录；模型、adapter、checkpoint 和原生恢复状态仍归 finetuning。

- `run_records/`：后续运行的自定义日志、配置和完成记录。
- `record_locations.json`：本次迁入的历史模型目录附属诊断位置；通过 `diagnostic_path()` 读取。
- `legacy_benchmark_analysis/`、`model_records/`：原始历史材料，未据此重判实验完成或有效性。

完整来源、旧路径和 SHA-256 见[整理记录](../../../../docs/inventory/separation_20261107/README.md)。
Windows 默认读取 `../paths.local.json`；原 `../paths.json` 保留原机器配置含义。缺少模型、数据或环境时应明确报缺，不恢复划分、不启动训练。
