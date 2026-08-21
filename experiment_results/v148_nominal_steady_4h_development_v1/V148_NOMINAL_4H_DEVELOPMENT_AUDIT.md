# V14.8 bounded guarded closed-loop development audit

Phase: horizon_isolated_development; physical horizon: 4.000 h; environment seeds: [17001, 17002, 17003, 17004, 17005].

Three frozen V14.1 models may execute one complete joint AGV action only when their choice is unanimous, the normalized 720-second utility gain is at least 0.15, no hard safety action is active, the 720-second cooldown has elapsed, and the per-run authority budget is not exhausted. All other decisions execute the guarded DT-aware fallback.

| Method | UPH | EER | Wait (s) | Conflicts | Blocking | Deadlocks | P95 decision (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Guarded DT-aware baseline | 27.790 | 55.395 | 247.19 | 117.200 | 6.200 | 0.400 | 0.0003 |
| V14.8 horizon-isolated world model | 27.790 | 55.395 | 247.19 | 117.200 | 6.200 | 0.400 | 0.0206 |

## Development continuation criteria

- [x] Every paired run reaches the 4-hour physical horizon
- [x] Every policy pair receives an identical exogenous task stream
- [x] Every paired method completes at least 20 tasks so operational KPIs are identifiable
- [x] No V14.8 run has an out-of-battery event or timeout
- [x] V14.8 introduces no additional deadlocks in any paired run
- [x] Mean physical route-blocked time is no more than 101% of the guarded baseline
- [x] Mean conflict events are no more than 105% of the guarded baseline
- [x] V14.8 UPH is at least 95% of the guarded baseline in every run
- [x] V14.8 EER is no more than 105% of the guarded baseline in every run
- [ ] At least three isolated overrides execute and total authority remains below 5%
- [x] Every run respects the 8-override authority budget
- [x] Mean V14.8 P95 decision time is at most 2 seconds

Proceed to a separately frozen confirmation protocol: **NO**.

This is development evidence only and must not be reported as a confirmatory closed-loop performance result.
