# V14.9-S bounded guarded closed-loop development audit

Phase: horizon_isolated_development; physical horizon: 4.000 h; environment seeds: [18001, 18002, 18003, 18004, 18005].

Three frozen V14.1 models may execute one complete joint AGV action only when their choice is unanimous, the normalized 720-second utility gain is at least 0.15, no hard safety action is active, the 720-second cooldown has elapsed, and the per-run authority budget is not exhausted. All other decisions execute the guarded DT-aware fallback.

| Method | UPH | EER | Wait (s) | Conflicts | Blocking | Deadlocks | P95 decision (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Guarded DT-aware baseline | 7.649 | 708.593 | 1275.56 | 6984.600 | 211.800 | 157.400 | 0.0003 |
| V14.9-S horizon-isolated world model | 7.649 | 708.557 | 1279.60 | 6981.200 | 213.800 | 157.000 | 0.0043 |

## Development continuation criteria

- [x] Every paired run reaches the 4-hour physical horizon
- [x] Every policy pair receives an identical exogenous task stream
- [ ] Every paired method completes at least 20 tasks so operational KPIs are identifiable
- [x] No V14.9-S run has an out-of-battery event or timeout
- [x] V14.9-S introduces no additional deadlocks in any paired run
- [x] Mean physical route-blocked time is no more than 101% of the guarded baseline
- [x] Mean conflict events are no more than 105% of the guarded baseline
- [x] V14.9-S UPH is at least 95% of the guarded baseline in every run
- [x] V14.9-S EER is no more than 105% of the guarded baseline in every run
- [x] At least three isolated overrides execute and total authority remains below 5%
- [x] Every run respects the 8-override authority budget
- [x] Mean V14.9-S P95 decision time is at most 2 seconds

Proceed to a separately frozen confirmation protocol: **NO**.

This is development evidence only and must not be reported as a confirmatory closed-loop performance result.
