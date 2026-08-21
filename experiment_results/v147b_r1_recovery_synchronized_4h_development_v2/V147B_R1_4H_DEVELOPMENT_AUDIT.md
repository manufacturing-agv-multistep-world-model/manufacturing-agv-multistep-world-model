# V14.7b-R1 bounded guarded closed-loop development audit

Phase: horizon_isolated_development; physical horizon: 4.000 h; environment seeds: [16001, 16002, 16003, 16004, 16005].

Three frozen V14.1 models may execute one complete joint AGV action only when their choice is unanimous, the normalized 720-second utility gain is at least 0.15, no hard safety action is active, the 720-second cooldown has elapsed, and the per-run authority budget is not exhausted. All other decisions execute the guarded DT-aware fallback.

| Method | UPH | EER | Wait (s) | Conflicts | Blocking | Deadlocks | P95 decision (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Guarded DT-aware baseline | 13.196 | 1031.282 | 2996.38 | 6268.000 | 169.400 | 105.600 | 0.0003 |
| V14.7b-R1 horizon-isolated world model | 13.192 | 1031.271 | 3000.89 | 6271.000 | 172.400 | 105.400 | 0.0011 |

## Development continuation criteria

- [x] Every paired run reaches the 4-hour physical horizon
- [x] Every policy pair receives an identical exogenous task stream
- [x] Every paired method completes at least one task so EER is identifiable
- [x] No V14.7b-R1 run has an out-of-battery event or timeout
- [x] V14.7b-R1 introduces no additional deadlocks in any paired run
- [x] Mean physical route-blocked time is no more than 101% of the guarded baseline
- [x] Mean conflict events are no more than 105% of the guarded baseline
- [x] V14.7b-R1 UPH is at least 95% of the guarded baseline in every run
- [x] V14.7b-R1 EER is no more than 105% of the guarded baseline in every run
- [x] At least three isolated overrides execute and total authority remains below 5%
- [x] Every run respects the 8-override authority budget
- [x] Mean V14.7b-R1 P95 decision time is at most 2 seconds

Proceed to a separately frozen confirmation protocol: **YES**.

This is development evidence only and must not be reported as a confirmatory closed-loop performance result.
