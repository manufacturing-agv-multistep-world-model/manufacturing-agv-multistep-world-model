# V11 独立订单流重训练交接清单

## 运行前

把本机当前核心包中的下列文件覆盖到 GPU 电脑上的同名位置：

- `agv_dt_env.py`
- `train_world_model_multistep.py`
- `analyze_multistep_decision_attribution.py`
- `tests/test_decision_attribution_protocol.py`
- `scripts/102_train_v11_physics_factorial_ablation.ps1`
- `scripts/106_train_v11_independent_arrival_factorial.ps1`
- `EXPERIMENT_REGISTRY.csv`

不要复制或改名旧的 `pi_gwm_multistep_v11_factorial_*` 模型来冒充新模型。新模型目录必须包含 `arrival_v4`。

## GPU 电脑运行命令

在 `JMS_MULTISTEP_WORLD_MODEL_CORE_2026-08-11` 根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\106_train_v11_independent_arrival_factorial.ps1 -Condition all -RequireGpu -CpuThreads 4
```

该命令会训练四个条件，每个条件三个模型种子，共十二个模型：

1. 完整物理图世界模型；
2. 去掉物理一致性损失；
3. 去掉显式物理特征；
4. 同容量纯数据图模型。

## 运行结束后的基本检查

- 控制台最后应显示 `Independent-arrival V11 factorial training finished.`
- 每个模型目录都应包含 `physics_graph_world_model_multistep.pt`、`training_args.json`、`training_history.csv` 和 `run_summary.txt`。
- 每个 `training_args.json` 的 `transition_schema_version` 必须是 `assignment_visible_congestion_independent_arrival_streams_v4`。
- 模型目录名必须包含 `arrival_v4`，不能覆盖旧 V11 目录。

## 需要拷回本机的内容

把 GPU 电脑上 `world_model_runs` 中所有以下目录完整拷回：

- `pi_gwm_multistep_v11_arrival_v4_shared`
- `pi_gwm_multistep_v11_arrival_v4_full_seed42`、`seed43`、`seed44`
- `pi_gwm_multistep_v11_arrival_v4_no_physics_loss_seed42`、`seed43`、`seed44`
- `pi_gwm_multistep_v11_arrival_v4_no_physical_features_seed42`、`seed43`、`seed44`
- `pi_gwm_multistep_v11_arrival_v4_data_only_seed42`、`seed43`、`seed44`

拷回后先做模型完整性和多步预测复核，暂时不要直接重跑 N1 控制对比。
