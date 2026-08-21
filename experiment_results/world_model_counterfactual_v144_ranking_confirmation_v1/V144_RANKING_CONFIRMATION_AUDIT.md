# V14.4 preregistered independent action-ranking confirmation

Untouched data seed: 15400; trajectories: 12; paired candidate samples: 1249.

The primary endpoint is candidate-action ranking at the 720-second physical horizon. Energy, completed tasks, and charge-queue time receive equal weights after frozen training-scale normalization. The unchanged DT-aware action is the zero-effect baseline.

- Eligible decision states: 413
- Model mean regret: 0.28844
- Baseline mean regret: 0.44182
- Regret reduction: 0.347
- Top-1 agreement: 0.482
- Random-choice top-1 expectation: 0.254
- Episode-bootstrap 95% CI: [0.198, 0.506]

## Episode-level stability

| Episode | States | Regret reduction | Top-1 |
|---:|---:|---:|---:|
| 0 | 35 | +0.773 | 0.571 |
| 1 | 35 | +0.852 | 0.514 |
| 2 | 34 | +0.289 | 0.382 |
| 3 | 33 | +0.117 | 0.576 |
| 4 | 34 | +0.408 | 0.559 |
| 5 | 34 | +0.663 | 0.382 |
| 6 | 34 | +0.036 | 0.588 |
| 7 | 35 | +0.235 | 0.486 |
| 8 | 35 | +0.604 | 0.514 |
| 9 | 35 | +0.286 | 0.371 |
| 10 | 35 | -0.098 | 0.343 |
| 11 | 34 | +0.387 | 0.500 |

## Preregistered continuation criteria

- [x] Exactly 12 untouched complete trajectories are evaluated
- [x] At least 1000 paired candidate samples are available
- [x] At least 350 complete terminal decision states are ranked
- [x] Every target component has at least 85% valid physical-time coverage
- [x] Terminal energy and task effects each have at least 10% nonzero coverage
- [x] Ensemble terminal ranking regret is reduced by at least 15%
- [x] Trajectory-bootstrap 95% interval for regret reduction is above zero
- [x] At least 9 of 12 trajectories have positive regret reduction
- [x] Top-1 agreement exceeds random choice by at least 8 percentage points
- [x] At least two of three frozen model seeds independently reduce regret

Proceed to shadow decision evaluation: **YES**.

Passing supports action-ranking validity only. It is not a closed-loop throughput, energy, or safety claim.
