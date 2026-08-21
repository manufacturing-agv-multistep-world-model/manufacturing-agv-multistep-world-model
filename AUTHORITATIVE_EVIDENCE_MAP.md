# 正式证据映射

| 级别 | 论文中可以写的结论 | 唯一证据目录 | 主文件 | 边界 |
|---|---|---|---|---|
| 正式主结果 M1 | 物理特征和物理损失提高多步预测的综合精度 | `experiment_results/v11_physics_factorial_arrival_v4_independent_v2` | `condition_summary.csv`, `paired_episode_bootstrap.csv` | 预测有效性，不等于控制性能 |
| 正式主结果 N6 | 冻结模型在 720 s 物理时域上改善候选动作排序 | `experiment_results/world_model_counterfactual_v144_ranking_confirmation_v1` | `V144_RANKING_CONFIRMATION_AUDIT.md` | 仅为离线排序有效性 |
| 正式边界结果 N13 | 完整模型的平均遗憾和 Top-1 优于同参数扁平网络，但配对区间跨零，未证明稳定图结构增量优势 | `experiment_results/v150_graph_vs_flat_confirmation_seed17400` | `V150_ARCHITECTURE_COMPARISON_AUDIT.md`, `paired_state_ranking_rows.csv` | 不得表述为图架构显著优越 |
| 正式主结果 N7 | 三模型一致和冻结收益门槛可产生稀疏但总体有益的影子建议 | `experiment_results/world_model_counterfactual_v145_shadow_confirmation_parallel_v2` | `V145_SHADOW_CONFIRMATION_AUDIT.md` | 建议未执行，不是闭环绩效 |
| 独立验证 A1 | Python 与 AnyLogic 重现 steady/rush 的产能和积压趋势 | `paper_outputs/anylogic_validation/final` | `ANYLOGIC_MULTI_SEED_AUDIT.md`, `anylogic_summary_95ci.csv` | 不验证非线性电池、学习策略最优性或逐点排队时间 |
| 安全压力开发 N10 | 有限权限和安全回退在过载压力测试中未引入额外死锁 | `experiment_results/v147b_r1_recovery_synchronized_4h_development_v2` | `V147B_R1_4H_DEVELOPMENT_AUDIT.md` | 存在产能地板效应，只能作为安全压力证据 |
| 弃权边界 N11 | 正常工况下冻结门槛导致零接管，显示选择性弃权 | `experiment_results/v148_nominal_steady_4h_development_v1` | `V148_NOMINAL_4H_DEVELOPMENT_AUDIT.md` | 未通过权限活性门 |
| 工况边界 N12a | 稳态到达与通道压力的组合会导致产能地板 | `experiment_results/v149s_steady_stress_4h_development_v1` | `V149S_BOUNDARY_4H_DEVELOPMENT_AUDIT.md` | 不可报告 EER 性能改善 |
| 工况边界 N12b | rush 到达和基线容量下系统可运行，但仅发生两次接管且未形成性能改善 | `experiment_results/v149d_rush_baseline_4h_development_v1` | `V149D_BOUNDARY_4H_DEVELOPMENT_AUDIT.md` | 不再继续事后调门槛寻找好结果 |

## 数字使用原则

1. 主文量化结论只能来自 M1、N6、N7、N13 和 A1。
2. N10-N12b 只用于讨论安全、弃权和运行边界，不得用于宣称闭环优越性。
3. 所有区间估计均保留轨迹或 episode 内相关性，不把单个决策点当成独立样本。
4. 未列在本表的旧实验不进入当前论文。
5. V12-V13-V14.1 冻结式主模型谱系及九个权重哈希以 `world_model_runs/MODEL_LINEAGE_CN.md` 为准；V15.0 三个等参数平坦基线权重以 `world_model_runs/V150_FLAT_BASELINE_LINEAGE_CN.md` 为准。
6. V14.1 稀疏分量逐点误差不优于零效应预测，正式结论仅支持相对动作排序和选择性影子建议。
7. V15.0 参数预算匹配确认已完成：完整模型平均遗憾较低且 Top-1 较高，但配对 95% 区间跨零、仅 7/12 条轨迹占优；不得写成图结构显著优于非图学习器。
