# V9 多步物理图世界模型实验说明

## 这次升级解决什么问题

旧模型只预测下一次决策后的结果，控制器也只比较当前动作。它可以作为监督预测器，但不足以支撑论文中“世界模型进行未来想象和滚动规划”的核心创新表述。

V9 同时预测车辆状态、当前 20 节点路网状态、系统全局状态和时间、能耗、阻塞等结果，并把自己的预测重新作为下一步输入。控制器在模型内部比较多条未来动作序列，每次只执行第一步，然后读取真实数字孪生状态重新规划。这是可检验的多步世界模型和滚动时域控制，不再是一步打分器。模型维度由场景文件自动读取，不在网络代码中写死节点数量。

## 运行顺序

### 1. 三个随机种子训练，耗时较长，由训练电脑运行

先在训练电脑确认当前 Python 使用的是 CUDA 版 PyTorch：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

第二行必须输出 `True`，第三行应显示显卡名称。仅仅安装了 NVIDIA 显卡和驱动还不够，当前 Python 环境中的 PyTorch 也必须支持 CUDA。

推荐使用以下命令。`-RequireGpu` 会在 CUDA 不可用时立即停止，防止误用 CPU 长时间训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\68_train_multistep_world_model_v9_seeds.ps1 -RequireGpu
```

脚本检测到 CUDA 后会自动启用显卡、锁页内存、异步数据传输和自动混合精度。默认把 CPU 数学线程限制为 4，并在 Windows 中使用较低进程优先级，避免训练期间影响其他工作。如果散热较弱，可进一步限制为 2 线程：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\68_train_multistep_world_model_v9_seeds.ps1 -RequireGpu -CpuThreads 2
```

若不带 `-RequireGpu`，同一脚本也能在 CPU 环境运行，但速度会明显更慢。CPU 与 GPU 的浮点结果不要求逐位相同，论文使用三个训练种子和统计结果判断稳定性。

预期生成：

```text
world_model_runs/pi_gwm_multistep_v9_seed42
world_model_runs/pi_gwm_multistep_v9_seed43
world_model_runs/pi_gwm_multistep_v9_seed44
```

每个目录必须包含模型检查点、训练历史、训练参数和运行摘要。选择最佳轮次时使用完整轨迹分组后的开放环验证损失，验证阶段不使用教师答案。

### 2. 多步开放环预测诊断，训练完成后运行

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\69_diagnose_multistep_world_model_v9.ps1
```

该实验报告第 1、3、5、10 步的车辆状态误差、节点状态误差、全局状态误差以及时间、能耗、阻塞等指标误差。论文必须如实展示误差随预测步数增长的情况，而不是只展示最好的第一步结果。

### 3. 规划深度消融，耗时较长

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\70_run_multistep_planning_ablation_v9.ps1
```

该实验在统一 4 小时物理时间下比较：

| 方法 | 含义 |
|---|---|
| H1 | 只看下一步，作为一步世界模型控制基线 |
| H3 | 看未来三步，V9 默认方案 |
| H5 | 看未来五步，检查更深规划是否带来收益或误差累积 |

最终选择 H3 不能只凭经验。需要同时比较产能、单位任务能耗、阻塞车辆时间、死锁、任务等待时间和每次决策的计算耗时。

## 可以写进论文的结论边界

完成代码不等于创新已经成立。只有满足以下证据后，才能把多步世界模型放在摘要和贡献第一条：

1. 新随机种子上的三步和五步预测误差可控，物理趋势方向正确。
2. H3 相比 H1 在至少一项关键系统指标上稳定改善，且不以明显降低产能为代价。
3. H5 若没有继续改善，应解释为模型误差与规划深度之间的权衡，而不能删除该结果。
4. 三个训练种子和多个评估种子的结论方向一致，并报告均值、标准差和配对统计检验。
5. AnyLogic 继续承担独立系统趋势验证，不宣称它验证了神经网络内部机理。

如果 H3 不优于 H1，论文仍可报告负结果，但不能把“多步规划提高调度性能”作为核心结论。此时需要先诊断状态误差、动作候选覆盖和效用权重，而不是调整图表隐藏差异。
