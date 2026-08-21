# JMS 当前投稿图件说明

本目录是多步世界模型主线唯一允许用于当前稿件的正式图件包，图号与 `manuscript/current` 中英文稿一致。

## 图件清单

| 图号 | 文件前缀 | 论文作用 | 数据来源 |
|---|---|---|---|
| Figure 1 | `figure_1_multistep_world_model_framework` | 数字孪生、配对多步世界模型与证据/权限阶梯 | 方法示意图 |
| Figure 2 | `figure_2_cad_derived_scene` | 20节点CAD派生场景、功能分区和路线类型 | `agv-test2/simplified_cad_scenario` |
| Figure 3 | `figure_3_detailed_world_model_architecture` | 图状态编码、冻结骨干、配对预测头、多时域输出与权限门 | 方法示意图 |
| Figure 4 | `figure_4_physics_factorial_evidence` | 物理损失和完整物理组合对多步预测的作用 | M1正式析因实验 |
| Figure 5 | `figure_5_counterfactual_decision_evidence` | 未见轨迹动作排序与非执行影子建议 | N6与N7正式确认实验 |
| Figure 6 | `figure_6_paired_formulation_boundary` | 直接配对监督与绝对结果相减形式的证据边界 | V15.1冻结独立确认与固定权重敏感性 |
| Figure 7 | `figure_7_graph_representation_boundary` | 图模型与同参数扁平网络的贡献边界 | N13冻结独立确认实验 |
| Figure 8 | `figure_8_anylogic_validation` | Python与AnyLogic容量、拥堵趋势及产能一致性 | A1独立验证 |

## 结论边界

- Figure 3 仅解释正式实现结构和冻结/训练边界，不是定量性能证据。
- Figure 4 支持完整物理图模型降低综合多步误差，不支持每个单项效应都显著改善。
- Figure 5 支持候选动作排序和稀疏、较可靠的影子建议，不支持闭环产能或能效显著提高。
- Figure 6 支持直接配对监督的点估计方向，但主配对区间跨零，不支持稳定的形式优越性。
- Figure 7 的点估计支持图模型，但配对区间跨零，不支持图结构稳定优于同参数扁平网络。
- Figure 8 仅验证系统容量和拥堵趋势，不验证电池机理、世界模型最优性或真实工厂部署。
- Figure 2 是去标识的CAD派生拓扑，不能声称公开了原始工厂CAD。

## 统计与可追溯性

- Figure 4 的柱为20个配对评价轨迹的均值，误差棒为标准差；森林图为轨迹配对自助法95%置信区间。
- Figure 5 的排序集和影子集使用不同的未见轨迹种子，影子建议从未执行。
- Figure 6 使用12条冻结独立确认轨迹、410个完整决策状态、1,257个配对候选和5,000次轨迹配对自助法；四组固定权重敏感性不更新模型。
- Figure 7 使用12条冻结独立确认轨迹和5,000次轨迹配对自助法；两个学习头均有56,457个可训练参数。
- Figure 8 的AnyLogic每个组合有3个种子，Python每个组合有10个种子；误差棒为95%置信区间。
- 每张定量图的绘图源数据均保存在 `source_data`。
- 八张图均可由 `tools/build_submission_figures.py` 重建。

## 输出格式

每张图提供可编辑SVG、PDF、300 dpi PNG和LZW压缩600 dpi TIFF。论文正文优先嵌入PNG或PDF，投稿单图优先提供TIFF或PDF。
