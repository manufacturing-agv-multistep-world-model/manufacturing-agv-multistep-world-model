# 正式世界模型谱系

本文最终反事实模型采用冻结式逐阶段训练。后一级读取前一级冻结权重，但只训练新增模块；确认数据不参与任何一级参数更新。

## 阶段关系

1. V12 从头训练充电拥堵感知的十步物理图状态骨干。
2. V13 从对应种子的 V12 权重初始化，冻结原骨干，仅训练未来 80 决策步充电排队风险头。
3. V14.1 从对应种子的 V13 权重初始化，冻结全部 V13 参数，仅训练 120/360/720 s 配对反事实效应头。

## 冻结权重哈希

| 阶段 | 种子 | 相对路径 | SHA-256 |
|---|---:|---|---|
| V12 | 42 | `pi_gwm_multistep_v12_charge_seed42/physics_graph_world_model_multistep.pt` | `452E78A3770C598B6878F617651070461036D42BEF5D1515B73908E7155D4738` |
| V12 | 43 | `pi_gwm_multistep_v12_charge_seed43/physics_graph_world_model_multistep.pt` | `88C484309CA5AB9780213258708C76E0C7194D0F9D565E0B6620D6C3AB881887` |
| V12 | 44 | `pi_gwm_multistep_v12_charge_seed44/physics_graph_world_model_multistep.pt` | `1156EAD3CECBF8D01956FADAF0DABC34FBC0EEC701A3DB770385C0E5D61527BC` |
| V13 | 42 | `pi_gwm_multistep_v13_multiscale_v2_seed42/physics_graph_world_model_multistep.pt` | `5691E96D7F69ADF811AEE689AE87AFE799BA14D8721E3FAA02F9C865AA1E9B28` |
| V13 | 43 | `pi_gwm_multistep_v13_multiscale_v2_seed43/physics_graph_world_model_multistep.pt` | `EC1860DDA920637C81FAB2328CE2459F75C460EEA9CFECDC7E08DBF2817DA2DE` |
| V13 | 44 | `pi_gwm_multistep_v13_multiscale_v2_seed44/physics_graph_world_model_multistep.pt` | `3E7F46C762592E0A75219900138285C20159DC21DD91FAED75854FC4159C8318` |
| V14.1 | 42 | `pi_gwm_counterfactual_v141_seed42/physics_graph_world_model_counterfactual.pt` | `A8D894FCCD94FD63090B4A81E196CBE246F9032DEDEDA932A7FE4968BFE14B3A` |
| V14.1 | 43 | `pi_gwm_counterfactual_v141_seed43/physics_graph_world_model_counterfactual.pt` | `FFCEE1B230C34EF7680D4871792DDF05D5456A60E3B3C1A721573FF3A3950C56` |
| V14.1 | 44 | `pi_gwm_counterfactual_v141_seed44/physics_graph_world_model_counterfactual.pt` | `FA50AC594622F84ABC19211F3D70930B148549A37410D9F2D10C2C5C6F69925F` |

## 可复现文件

每个种子目录均保留权重、训练参数、训练历史和运行摘要。V12 共享缓存位于 `pi_gwm_multistep_v12_charge_shared`，V14.1 配对训练缓存位于 `pi_gwm_counterfactual_v141_shared`。发布副本已将历史本机绝对路径转换为仓库相对路径；重新运行时应使用当前仓库脚本，不应依赖特定电脑盘符。
