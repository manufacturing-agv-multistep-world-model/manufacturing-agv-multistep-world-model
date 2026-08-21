# V15.0 参数预算匹配扁平基线谱系

## 冻结用途

这三套模型只用于比较 V14.1 的图与物理表征是否在最终候选动作排序任务中提供增量价值。它们不是新的主模型，也不替代多步物理图世界模型。

## 训练来源

- 共享数据：`pi_gwm_counterfactual_v141_shared/paired_material_samples_v2_seed14100.pkl.gz`
- 完整配对样本：1690
- 分组训练样本：1408
- 分组验证样本：282
- 训练/验证划分种子：14100
- 模型随机种子：42、43、44
- 输入：动态智能体状态、动态节点状态、全局状态、联合动作的扁平拼接
- 明确排除：邻接矩阵、静态节点/边物理属性、图消息传递
- 网络：`192 -> 192 -> 96 -> 9`
- 可训练参数：56,457，与 V14.1 配对反事实价值头精确相同
- 损失、尺度、事件权重、优化器及早停规则：与 V14.1 配对阶段相同

## 冻结权重

| 种子 | 最佳验证损失 | SHA-256 |
|---:|---:|---|
| 42 | 0.1406157911 | `75DEEE7C96401B30F63F5AC437EFFEF215A47A7CB00E8296EE1B8BFBED72CFA0` |
| 43 | 0.1407243423 | `4C4FE2FFF630ADE4AA63FBA599603D72539A8982C9E41FCC6155B57A6419147F` |
| 44 | 0.1416074075 | `88E510A444FDD492EF098AB6A51AEC8A09756A501B45264E8B5134FA322C6621` |

对应目录：

- `flat_mlp_counterfactual_v150_seed42`
- `flat_mlp_counterfactual_v150_seed43`
- `flat_mlp_counterfactual_v150_seed44`

每个目录均包含 `flat_counterfactual_baseline.pt`、`training_history.csv` 和 `training_audit.json`。

## 结论边界

训练损失不能作为架构优越性证据。只有冻结的种子 17400 正式轨迹比较完成后，才能根据 `docs/V150_FLAT_MLP_RANKING_BASELINE_PROTOCOL_CN.md` 的预先规则判断图与物理表征是否获得支持。

