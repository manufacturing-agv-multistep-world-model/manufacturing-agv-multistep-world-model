# AnyLogic 多随机种子正式验证运行卡

## 本轮目标

完成 `steady/rush × seed 1/2/3 × 1/4/8 h`，共 18 个正式组合。
`5 h` 是 PLE 阶段留下的补充稳定性记录，不进入主验证的完整性要求。

当前 `seed=1` 的 1/4/8 h 和全部三个种子的 8 h 已完成；还需补充
seed 2/3 在 steady/rush 下的 1 h 和 4 h，共 8 次运行。

当前验证对象是：相同 20 节点路网、3 台 AGV、运动学参数、基线通行能力和随机任务流下的独立离散事件仿真趋势。
它不验证电池非线性模型，也不验证 Graph-MAPPO 的最优性。

## 每次运行必须同时修改的三处

仅修改 `alRunSeed` 不会改变真正随机数，这是本轮最重要的注意事项。

1. 在 `Main` 中设置场景：
   - `alRushMode = false`：steady。
   - `alRushMode = true`：rush。
2. 在 `Main` 中设置记录标签：`alRunSeed = 2` 或 `3`。
3. 打开 `Simulation` 实验属性，在“随机性/随机数生成”中选择固定种子，并把真实种子也设置为同一个数值 2 或 3。
4. 在 `Simulation` 实验停止时间中设置：
   - 1 h：`3600` 秒。
   - 4 h：`14400` 秒。
   - 8 h：`28800` 秒。

运行前核对：真实固定种子 = `alRunSeed`。两者不一致时，不要运行。

## 建议运行顺序

| 完成 | 场景 | 真实固定种子 | alRunSeed | 停止时间 |
|---|---|---:|---:|---:|
| [x] | steady | 2 | 2 | 3600 s |
| [x] | steady | 2 | 2 | 14400 s |
| [x] | rush | 2 | 2 | 3600 s |
| [x] | rush | 2 | 2 | 14400 s |
| [x] | steady | 3 | 3 | 3600 s |
| [x] | steady | 3 | 3 | 14400 s |
| [x] | rush | 3 | 3 | 3600 s |
| [x] | rush | 3 | 3 | 14400 s |

## 每次运行后检查

控制台最后必须同时出现：

- `Scenario = ...`
- `Run seed = ...`
- `Elapsed hours = 1.0/4.0/5.0`
- `CSV saved to: ...anylogic_validation_results.csv`

CSV 路径：

`AGV_DT_AnyLogic_Validation/Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv`

8 次运行完成后，不再按原始 CSV 总行数验收，因为文件中可能保留 600 s、5 h、10 h
试运行或完全重复的历史行。正式验收只检查：清洗后是否包含
`steady/rush × seed 1/2/3 × 1/4/8 h` 共 18 个唯一组合。
600 s 和 10 h 数据只作为 smoke/扩展测试，5 h 作为 PLE 补充记录，均不进入主统计；
完全重复行由审计脚本去重。

## 自动检查命令

先生成匹配的 Python 运动学基准：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\47_run_anylogic_python_reference.ps1
```

再审计 AnyLogic CSV 并生成图表：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\48_analyze_anylogic_validation.ps1
```

正式统计输出到：

`paper_outputs/anylogic_validation/final/`

如果审计报告显示 `Missing required combinations: 0` 且没有伪多种子警告，本轮完成。

## Professional 版 8 h 扩展验证

在合法授权的 AnyLogic Professional 环境中，追加与 Python 主实验一致的
`steady/rush × seed 1/2/3 × 8 h` 六个组合。原有 1/4/5 h 数据保留，不覆盖、不删除。

8 h 的停止时间为 `28800 s`。每次运行仍必须保证 Simulation 实验中的真实固定种子
与 Main 中用于记录的 `alRunSeed` 完全一致。

| 完成 | 场景 | 真实固定种子 | alRunSeed | 停止时间 |
|---|---|---:|---:|---:|
| [x] | steady | 1 | 1 | 28800 s |
| [x] | rush | 1 | 1 | 28800 s |
| [x] | steady | 2 | 2 | 28800 s |
| [x] | rush | 2 | 2 | 28800 s |
| [x] | steady | 3 | 3 | 28800 s |
| [x] | rush | 3 | 3 | 28800 s |

每次结束后，控制台必须显示 `Elapsed hours = 8.0`，并确认 CSV 新增行的
`scenario`、`seed` 和 `horizon_h` 分别与本次设置一致。
