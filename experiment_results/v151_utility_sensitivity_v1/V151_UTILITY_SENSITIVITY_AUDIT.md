# V15.1 fixed utility-weight sensitivity

This is a post-confirmation robustness analysis and does not replace the frozen primary test.

| Weight set | Direct regret | Direct reduction | Direct top-1 | Absolute regret | Absolute - direct | 95% CI | Positive episodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| equal | 0.3466 | 0.315 | 0.402 | 0.4605 | +0.1139 | [-0.0072, +0.2280] | 8/12 |
| energy_priority | 0.6929 | 0.328 | 0.424 | 0.9111 | +0.2182 | [-0.0345, +0.4710] | 8/12 |
| throughput_priority | 0.3910 | 0.217 | 0.383 | 0.4466 | +0.0556 | [-0.0058, +0.1190] | 8/12 |
| queue_priority | 0.3879 | 0.340 | 0.405 | 0.5485 | +0.1606 | [+0.0409, +0.2731] | 10/12 |

Point-estimate directional robustness across all frozen weights: **YES**.

Statistical significance is not claimed unless the corresponding interval excludes zero.
