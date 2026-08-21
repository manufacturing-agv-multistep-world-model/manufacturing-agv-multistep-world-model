# V15.0 parameter-budget-matched architecture comparison

Phase: **confirmation**; data seed: 17400; complete trajectories: 12; paired candidates: 1256.

Both learned methods use three-seed ensembles and the same paired candidate states. The flat MLP has exactly the same 56,457 trainable parameters as the V14.1 counterfactual head but receives no adjacency matrix or static physical features.

| Method | Mean regret | Regret reduction vs unchanged action | Top-1 |
|---|---:|---:|---:|
| Full V14.1 physics-graph model | 0.30430 | 0.300 | 0.385 |
| Parameter-matched flat MLP | 0.32917 | 0.243 | 0.320 |
| Unchanged DT-aware action | 0.43463 | 0.000 | - |

Mean paired flat-minus-graph regret: **+0.02487**.
Trajectory-bootstrap 95% CI: [-0.05232, +0.10295].

## Trajectory-level comparison

| Episode | States | Graph regret | Flat regret | Flat - graph | Graph top-1 | Flat top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 35 | 0.21869 | 0.48061 | +0.26192 | 0.371 | 0.229 |
| 1 | 34 | 0.26643 | 0.34358 | +0.07715 | 0.529 | 0.412 |
| 2 | 34 | 0.29718 | 0.18721 | -0.10998 | 0.353 | 0.382 |
| 3 | 35 | 0.29957 | 0.26211 | -0.03746 | 0.429 | 0.371 |
| 4 | 35 | 0.34707 | 0.28388 | -0.06319 | 0.257 | 0.229 |
| 5 | 32 | 0.26481 | 0.27717 | +0.01236 | 0.438 | 0.250 |
| 6 | 35 | 0.09745 | 0.12009 | +0.02264 | 0.571 | 0.400 |
| 7 | 35 | 0.19546 | 0.29228 | +0.09682 | 0.314 | 0.314 |
| 8 | 35 | 0.47355 | 0.41428 | -0.05927 | 0.200 | 0.229 |
| 9 | 33 | 0.31611 | 0.36202 | +0.04591 | 0.303 | 0.333 |
| 10 | 35 | 0.40294 | 0.18671 | -0.21623 | 0.486 | 0.371 |
| 11 | 35 | 0.46833 | 0.73386 | +0.26553 | 0.371 | 0.314 |

## Protocol integrity

- [x] The frozen number of complete trajectories is present
- [x] The frozen minimum paired candidate sample count is met
- [x] The frozen minimum complete decision-state count is met
- [x] Graph and flat methods rank identical state-action candidates
- [x] All six frozen models use identical train-only target scales
- [x] All graph heads and flat models have exactly 56,457 trainable parameters
- [x] Every target component has at least 85% physical-time coverage

## Frozen scientific-support criteria

- [x] Full V14.1 has lower mean regret than the flat MLP
- [ ] The paired trajectory-bootstrap 95% interval is above zero
- [ ] At least 9 of 12 confirmation trajectories favor Full V14.1
- [x] Full V14.1 top-1 agreement is not below the flat MLP

Protocol integrity: **PASS**.
Evidence supports an incremental graph/physics representation contribution: **NO**.

A negative comparison remains a valid frozen result and must not trigger seed or threshold replacement.
