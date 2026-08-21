# V14.5 preregistered non-acting shadow-recommender confirmation

Untouched data seed: 15600; trajectories: 12; paired candidate samples: 1252.

The baseline DT-aware action was always executed. Recommendations were logged only when all three frozen models selected the same non-baseline candidate and the ensemble normalized utility margin was at least 0.15.

- Eligible decision states: 413
- Shadow recommendations: 116
- Recommendation coverage: 0.281
- Beneficial-recommendation precision: 0.750
- Mean true normalized gain: 0.3509
- Median true normalized gain: 0.1896
- Trajectory-bootstrap mean-gain 95% CI: [0.2422, 0.4652]

## Trajectory-level stability

| Episode | Recommendations | Coverage | Precision | Mean true gain |
|---:|---:|---:|---:|---:|
| 0 | 5 | 0.143 | 0.800 | +0.6800 |
| 1 | 8 | 0.229 | 0.625 | +0.2227 |
| 2 | 12 | 0.353 | 0.750 | +0.5468 |
| 3 | 8 | 0.235 | 0.750 | +0.1203 |
| 4 | 10 | 0.294 | 0.700 | +0.4014 |
| 5 | 11 | 0.314 | 0.909 | +0.4544 |
| 6 | 6 | 0.176 | 0.667 | +0.1880 |
| 7 | 10 | 0.286 | 0.500 | +0.0364 |
| 8 | 12 | 0.353 | 0.917 | +0.2643 |
| 9 | 10 | 0.286 | 0.900 | +0.7458 |
| 10 | 11 | 0.324 | 0.818 | +0.2436 |
| 11 | 13 | 0.382 | 0.615 | +0.3220 |

## Preregistered continuation criteria

- [x] Exactly 12 untouched complete trajectories are evaluated
- [x] At least 1000 paired candidate samples are available
- [x] At least 350 complete terminal decision states are eligible
- [x] Every target component has at least 85% valid physical-time coverage
- [x] At least 80 shadow recommendations are issued
- [x] Recommendation coverage is between 15% and 45%
- [x] At least 70% of recommendations have positive true utility
- [x] Mean true normalized recommendation gain is positive
- [x] Trajectory-bootstrap 95% mean-gain interval is above zero
- [x] At least 9 of 12 trajectories have positive mean recommendation gain

Proceed to a separately preregistered guarded closed-loop pilot: **YES**.

Passing establishes shadow-recommendation reliability only. No recommendation was executed, so this is not a closed-loop system-performance claim.
