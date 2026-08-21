# AnyLogic 独立验证结果及 JMS 写作边界

## 正式协议

- AnyLogic：steady/rush 两类任务流，1/4/8 h 三个物理时域，每个组合采用 3 个真实固定种子，共 18 次正式运行。
- Python：匹配相同路网、3 台 AGV、运动学、基线容量和随机任务流，采用 10 个种子，共 60 次参考运行。
- AnyLogic 原始 CSV 保留不变；主统计仅保留 1/4/8 h，早期 600 s、5 h 和 10 h 记录不进入主分析。

## 核心结果

| 场景 | 时域 | AnyLogic UPH | Python UPH | 相对差异 |
|---|---:|---:|---:|---:|
| steady | 1 h | 29.667 | 29.017 | +2.2% |
| steady | 4 h | 30.000 | 31.741 | -5.5% |
| steady | 8 h | 32.708 | 32.592 | +0.4% |
| rush | 1 h | 46.000 | 46.577 | -1.2% |
| rush | 4 h | 49.333 | 50.018 | -1.4% |
| rush | 8 h | 49.208 | 50.767 | -3.1% |

六个场景-时域组合的 UPH 最大绝对相对差异为 5.5%，低于预设的 10%-20% 趋势验证容许范围。8 h steady 的停止时 backlog 分别为 2.67 和 3.10，两个引擎均显示系统可以稳定消化任务；8 h rush 的 backlog 分别为 286.0 和 298.6，且 AnyLogic AGV 利用率接近 100%，两个引擎均显示需求超过车队服务能力。

## 可写入论文的结论

在匹配的 20 节点路网、车队规模、运动学参数、路径容量和随机任务流条件下，Python 数字孪生与 AnyLogic 独立离散事件模型在 throughput-capacity 关系和拥堵状态转换方面表现一致。steady 工况在长时域保持有限 backlog，而 rush 工况在两个引擎中均进入 AGV 容量饱和并形成随时域增长的任务积压。该结果支持数字孪生在制造系统层面的运输能力和拥堵趋势有效性。

## 必须保留的限制

- 等待时间方向一致，但数值量级存在差异；最大相对差异超过 UPH。这来自事件排序、路径预约和服务时间语义差异，因此不能宣称逐点数值等价。
- AnyLogic 验证不独立验证 Python 中的非线性电池模型。
- AnyLogic 验证不复现或证明 PI-GWM-GMAPPO 的策略最优性。
- 当前模型不支持真实接触动力学意义上的碰撞验证，也不用于证明精确 deadlock 数量一致。
- 合理表述是 independent cross-platform DES trend validation，而不是 real-world deployment validation。

## 剩余工作

1. 保存一张正式 AnyLogic GUI 全景图和一张 8 h 最终统计图至 `paper_outputs/anylogic_validation/screenshots/`。
2. 在论文方法部分加入 AnyLogic 模型、三种子协议和 1/4/8 h 时域设置。
3. 在结果部分加入 `figure_anylogic_validation` 和 UPH 对比表。
4. 在讨论部分加入上述验证边界，避免把跨平台趋势一致写成所有 KPI 的精确一致。
