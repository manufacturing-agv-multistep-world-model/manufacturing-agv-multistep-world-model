# V15.0 parameter-budget-matched architecture comparison

Phase: **development**; data seed: 16400; complete trajectories: 3; paired candidates: 89.

Both learned methods use three-seed ensembles and the same paired candidate states. The flat MLP has exactly the same 56,457 trainable parameters as the V14.1 counterfactual head but receives no adjacency matrix or static physical features.

| Method | Mean regret | Regret reduction vs unchanged action | Top-1 |
|---|---:|---:|---:|
| Full V14.1 physics-graph model | 0.08165 | 0.756 | 0.633 |
| Parameter-matched flat MLP | 0.11816 | 0.648 | 0.500 |
| Unchanged DT-aware action | 0.33521 | 0.000 | - |

Mean paired flat-minus-graph regret: **+0.03651**.
Trajectory-bootstrap 95% CI: [+0.00000, +0.07453].

## Trajectory-level comparison

| Episode | States | Graph regret | Flat regret | Flat - graph | Graph top-1 | Flat top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 0.24496 | 0.27995 | +0.03499 | 0.500 | 0.500 |
| 1 | 10 | 0.00000 | 0.00000 | +0.00000 | 0.600 | 0.500 |
| 2 | 10 | 0.00000 | 0.07453 | +0.07453 | 0.800 | 0.500 |

## Protocol integrity

- [x] The frozen number of complete trajectories is present
- [x] The frozen minimum paired candidate sample count is met
- [x] The frozen minimum complete decision-state count is met
- [x] Graph and flat methods rank identical state-action candidates
- [x] All six frozen models use identical train-only target scales
- [x] All graph heads and flat models have exactly 56,457 trainable parameters
- [x] Every target component has at least 85% physical-time coverage

## Directional development diagnostics (not formal evidence)

- [x] Full V14.1 has lower mean regret than the flat MLP
- [ ] The paired trajectory-bootstrap 95% interval is above zero
- [x] At least one of three development trajectories favors Full V14.1
- [x] Full V14.1 top-1 agreement is not below the flat MLP

Protocol integrity: **PASS**.
Evidence supports an incremental graph/physics representation contribution: **NOT EVALUATED (development only)**.

A negative comparison remains a valid frozen result and must not trigger seed or threshold replacement.
