# AnyLogic AGV 验证模型小白搭建步骤

本文档对应项目：

```text
JMS_AGV_DT_interpretable_params_2026-07-01
```

目标：

搭建 **1 个 AnyLogic 独立验证模型**，用于论文中验证 Python 高保真数字孪生的路网、任务流和 KPI 趋势是否能在另一个仿真平台中复现。

重要定位：

AnyLogic 里 **不训练 AI**，也不重新实现复杂 MARL。AnyLogic 只负责做独立验证：同一个路网、同一批 AGV 参数、同一类任务流，看看吞吐量、等待/阻塞、路径执行趋势是否和 Python 数字孪生一致。

## 0. 你最终只需要搭几个模型？

只搭 **1 个 AnyLogic 模型**：

```text
AGV_DT_AnyLogic_Validation.alpx
```

后面不同算法不用重新建模型，只换输入数据：

- `Nearest`：最近车启发式调度。
- `DT-aware`：高保真 DT 规则调度。
- `PI-GWM-GMAPPO`：最终学习策略的轨迹回放。

也就是说：

模型只有一个，验证数据可以有多组。

## 1. 第一阶段目标：先跑通最小模型

不要一上来就搭完整 20 节点，也不要一上来就导入 CSV。

第一关只做一个最小可运行模型：

```text
Home1 / Home2 / Home3 -> A -> B
```

只要能看到 3 台 AGV 从各自起点出发，把货物从 `A` 运到 `B`，第一阶段就成功。

第一阶段需要：

- 3 台 AGV。
- 3 个起始点：`Home1`、`Home2`、`Home3`。
- 1 个取货点：`A`。
- 1 个送货点：`B`。
- 1 条简单流程：`source -> moveByTransporter -> sink`。
- 先跑 `600 s` 或 `1800 s`。

## 2. 打开或新建 AnyLogic 项目

打开这个文件：

```text
AGV_DT_AnyLogic_Validation/AGV_DT_AnyLogic_Validation.alpx
```

如果你是新建模型：

1. 打开 AnyLogic。
2. 新建一个模型。
3. 打开左侧工程树里的 `Main`。
4. 打开 `Simulation` 实验。
5. 找到停止时间 `Stop time`。
6. 第一轮先设置为 `600` 秒。

最终正式验证已使用可运行 8 小时的授权版本完成 `1h / 4h / 8h`；早期 PLE 的 `5h` 结果仅作为补充历史记录，不进入主统计。

## 3. 先创建 5 个最小节点

在 `Main` 画布里，先放 5 个节点。

如果你要照完整论文场景来画，可以先打开这份场景图和坐标说明：

```text
docs/anylogic_scene_layout/ANYLOGIC_SCENE_LAYOUT_GUIDE_CN.md
docs/anylogic_scene_layout/anylogic_scene_layout_map.png
docs/anylogic_scene_layout/anylogic_node_coordinates_for_drawing.csv
docs/anylogic_scene_layout/anylogic_edge_distance_table_for_drawing.csv
```

建议从 AnyLogic 的 `Space Markup` 或 `Material Handling` 相关面板里找 `Node`、`Point Node`、`Network Node` 之类的对象。

第一批节点如下：

| 节点名 | 作用 |
|---|---|
| `Home1` | 第 1 台 AGV 起点 |
| `Home2` | 第 2 台 AGV 起点 |
| `Home3` | 第 3 台 AGV 起点 |
| `A` | 取货点 / 出库点 |
| `B` | 送货点 / 仓库点 |

摆放建议：

- `Home1`、`Home2`、`Home3` 放在左侧，竖着排列。
- `A` 放在它们右侧。
- `B` 放在 `A` 的右上方或右侧。

第一阶段不追求美观，先追求能跑通。

## 4. 用路径把节点连起来

用 AnyLogic 的路径对象把节点连起来。

先连这些：

| 从 | 到 | 说明 |
|---|---|---|
| `Home1` | `Home2` | AGV 停车通道 |
| `Home2` | `Home3` | AGV 停车通道 |
| `Home2` | `A` | 停车区到取货点 |
| `A` | `B` | 取货点到送货点 |

注意：

- 路径最好先设成双向。
- `Home1`、`Home2`、`Home3` 必须和 `A` 在同一个网络里。
- `A` 必须能走到 `B`。
- 不要出现孤立节点。

如果后面报错：

```text
归属地节点必须包含在一个相关的网络组中
```

通常就是 `Home` 节点没有正确连进同一个网络。

## 5. 添加 AGV 车队

从 `Material Handling Library` 里拖一个运输车队对象到 `Main`。

你现在的模型里可以继续叫：

```text
AGVFleet
```

不要强行改成 `transporterFleet`，之前你已经发现这个名字不一定能改。

车队参数建议：

| 参数 | 设置值 |
|---|---|
| 名称 | `AGVFleet` |
| 小车数量 | `3` |
| 归属地 / Home location | `Home1`、`Home2`、`Home3` |
| 单车容量 | `1` |
| 最大速度 | `1.2 m/s` |
| 加速度 | `0.5 m/s^2` |
| 网络 | 选择你刚才画的节点/路径网络 |

关键检查：

- 3 台车要分别停在 3 个 Home 点。
- Home 点必须属于 AGVFleet 所在的同一个路网。
- 如果提示找不到 home location，先检查路径连接，不要先改代码。

## 6. 添加最简单流程

在 `Main` 里放 3 个流程块：

```text
source -> moveByTransporter -> sink
```

连接顺序就是：

```text
source 的出口 -> moveByTransporter 的入口 -> sink 的入口
```

## 7. 设置 source

点击 `source`，右侧属性里这样设置：

| 属性 | 设置 |
|---|---|
| 到达方式 | 按速率到达 |
| 到达速率 | 先设 `1 per minute`，或者 `1/60 per second` |
| 智能体类型 | `Agent` |
| 到达位置 | 节点 `A` |

如果你看到“空间未设置”类似报错，通常是 `source` 生成的智能体没有放到节点 `A` 上。

所以一定要检查：

- `source` 的到达位置是不是选了 `A`。
- 智能体是否有空间位置。

## 8. 设置 moveByTransporter

点击 `moveByTransporter`，右侧属性这样设置：

| 属性 | 设置 |
|---|---|
| 目的地类型 | 节点 |
| 目的地节点 | `B` |
| 获取运输车 | 勾选 |
| 车队 | `AGVFleet` |
| 拾起位置类型 | 节点 |
| 拾起节点 | `A` |
| 装载时间 | `triangular(0.5, 1, 1.5)` 秒 |
| 卸载时间 | `triangular(0.5, 1, 1.5)` 秒 |
| 任务抢占策略 | 先用默认或无抢占 |
| 运输车释放后 | 没有任务就返回当前归属地或保持默认 |

第一阶段目的地固定为 `B`，不要先设置复杂目的地。

## 9. 设置 sink

第一阶段 `sink` 不需要复杂设置。

只要能接收完成的实体即可。

后面我们会在 `sink` 的进入动作里加计数器，用于统计完成任务数量。

## 10. 第一次运行检查

把 `Simulation` 停止时间设为：

```text
600 s
```

然后点击运行。

你应该看到：

- 3 台 AGV 出现在 `Home1`、`Home2`、`Home3`。
- `source` 在 `A` 产生任务。
- 某台 AGV 从 Home 点开到 `A`。
- AGV 把任务从 `A` 运到 `B`。
- 任务进入 `sink`。
- 模型不报错。

如果这一步跑通，说明最小 AnyLogic 模型成功。

这一步完成后，请给我截图。

## 11. 常见错误怎么处理

### 错误 1：找不到空闲 home location

报错类似：

```text
Can't find a free home location for transporter
```

处理方法：

1. 检查是不是只有 1 个 home 点，却设置了 3 台 AGV。
2. 检查 `Home1`、`Home2`、`Home3` 是否都是有效节点。
3. 检查 AGVFleet 的 home locations 是否填写了 3 个不同节点。

### 错误 2：归属地节点不在相关网络组中

报错类似：

```text
归属地节点必须包含在一个相关的网络组中
```

处理方法：

1. 检查 `Home1-Home2-Home3` 是否用路径连起来。
2. 检查 `Home2` 是否连接到 `A`。
3. 检查 `A` 是否连接到 `B`。
4. 检查 AGVFleet 选择的网络是不是这套节点所在网络。

### 错误 3：智能体空间未设置

报错类似：

```text
智能体来自模块 moveByTransporter 的空间未设置
```

处理方法：

1. 点击 `source`。
2. 找到到达位置。
3. 设置为节点 `A`。
4. 点击 `moveByTransporter`。
5. 把拾起位置也设置为节点 `A`。

## 12. 最小模型跑通后，再补完整 20 节点

最小模型跑通后，再补下面这些节点。

完整 20 节点如下：

| 节点名 | 类型 | 说明 |
|---|---|---|
| `A` | 取货点 | 主取货 / 出库点 |
| `G1` | 引导点 | 主干路节点 |
| `G2` | 引导点 | 主干路节点 |
| `G2_G3_Mid` | 控制点 | 长直走廊中间控制点 |
| `G3` | 引导点 | 主干路节点 |
| `G4` | 引导点 | 主干路节点 |
| `G5` | 引导点 | 主干路节点 |
| `G6` | 引导点 | 主干路节点 |
| `B` | 仓库点 | 主仓库端点 |
| `Home1` | 起点 | 第 1 台 AGV 停车点 |
| `Home2` | 起点 | 第 2 台 AGV 停车点 |
| `Home3` | 起点 | 第 3 台 AGV 停车点 |
| `Charge` | 充电点 | 充电区 |
| `P1_Packaging` | 工位 | 包装工位 |
| `P2_Labeling` | 工位 | 贴标 / 打码工位 |
| `PrepBuffer` | 缓冲区 | 预处理缓存区 |
| `PassingBuffer` | 缓冲区 | A 点上方避让 / 等待区 |
| `MaterialBuffer` | 缓冲区 | G5 左侧物料缓存 |
| `W1_Storage` | 仓储位 | 仓库储位 1 |
| `W2_Storage` | 仓储位 | 仓库储位 2 |

## 13. 完整 20 节点连接关系

按下面关系连线。

主干路：

| 从 | 到 |
|---|---|
| `Home1` | `Home2` |
| `Home2` | `Home3` |
| `Home2` | `A` |
| `A` | `G1` |
| `G1` | `G2` |
| `G2` | `G2_G3_Mid` |
| `G2_G3_Mid` | `G3` |
| `G3` | `G4` |
| `G4` | `G5` |
| `G5` | `G6` |
| `G6` | `B` |

分支路：

| 从 | 到 | 说明 |
|---|---|---|
| `A` | `Charge` | 充电分支 |
| `G2` | `P1_Packaging` | 包装工位分支 |
| `G2_G3_Mid` | `P2_Labeling` | 贴标工位分支 |
| `G3` | `PrepBuffer` | 缓冲区分支 |
| `A` | `PassingBuffer` | 避让等待分支 |
| `G5` | `MaterialBuffer` | 物料缓存分支 |
| `B` | `W1_Storage` | 仓储位分支 |
| `B` | `W2_Storage` | 仓储位分支 |

建议所有路径先设为双向。等模型稳定后，如果要更像论文里的通道约束，再逐步设置窄通道容量。

## 14. 完整模型跑通前，不要急着导入 CSV

顺序一定是：

1. 先跑通 `A -> B`。
2. 再补完整 20 节点。
3. 再让 `moveByTransporter` 送到不同目的地。
4. 再统计 KPI。
5. 最后才导入 Python 生成的 dispatch CSV。

不要跳步，不然后面报错会很难定位。

## 15. 加最简单 KPI 统计

最小模型跑通后，可以加计数器。

在 `Main` 里新增变量：

```java
int alThroughput = 0;
int alTaskCount = 0;
double alStartTime = 0;
```

在 `Main` 的启动代码里写：

```java
alStartTime = time();
```

在 `sink` 的进入动作里写：

```java
alThroughput++;
alTaskCount++;
```

在仿真结束时输出：

```java
double elapsedHour = max(1e-9, time() / 3600.0);
traceln("AnyLogic throughput = " + alThroughput);
traceln("AnyLogic UPH = " + alThroughput / elapsedHour);
traceln("AnyLogic task count = " + alTaskCount);
```

第一阶段只统计吞吐量和 UPH 就够了。

## 16. 论文验证阶段怎么跑

后面论文里需要的验证不是看动画好不好看，而是看 KPI 趋势。

推荐运行时间：

| 情况 | 运行时间 |
|---|---|
| AnyLogic 可以跑 8 小时 | `1h / 4h / 8h` |
| 免费版最多只能稳定跑 5 小时 | `1h / 4h / 5h` |

推荐方法：

| 强度 | 方法 |
|---|---|
| 最低可接受 | 只跑 `DT-aware` |
| 更适合 JMS | 跑 `Nearest`、`DT-aware`、`PI-GWM-GMAPPO` |

论文里不要写 AnyLogic 证明了真实工厂部署，只写：

```text
AnyLogic was used as an independent DES/material-handling validation platform.
Fixed decision traces generated by the Python DT were replayed under the same
20-node network and AGV physical parameters. The validation focuses on trend
consistency in throughput, route execution, and blocking/waiting indicators.
```

中文意思：

```text
AnyLogic 被用作独立离散事件 / 物料搬运验证平台。Python 数字孪生生成的固定调度轨迹
在相同 20 节点路网和 AGV 物理参数下回放。验证重点是吞吐量、路径执行、
等待/阻塞等指标的趋势一致性，而不是逐秒完全一致。
```

## 17. 每完成一步你给我什么

完成最小节点和路径后：

- 给我一张画布截图。

完成 AGVFleet 设置后：

- 给我一张 AGVFleet 属性截图。

完成第一次运行后：

- 给我运行截图。
- 如果报错，把完整错误复制给我。

完成 KPI 统计后：

- 给我控制台输出里的 throughput 和 UPH。

完成 CSV 回放后：

- 给我 AnyLogic 导出的结果 CSV。
- 给我最终运行截图。
- 告诉我 stop time 是 `3600 s`、`14400 s`、`18000 s` 还是 `28800 s`。
