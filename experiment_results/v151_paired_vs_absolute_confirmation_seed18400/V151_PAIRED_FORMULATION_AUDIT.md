# V15.1 paired-formulation confirmation

Fresh data seed: 18400; complete trajectories: 12; paired candidates: 1257.

Both methods use the same frozen V13 physics-graph backbone, identical 56,457-parameter action-value heads, the same training pairs and the same three initialization seeds. The only scientific change is the target: direct paired effects versus two absolute branch outcomes followed by subtraction.

| Formulation | Mean regret | Regret reduction vs unchanged action | Top-1 |
|---|---:|---:|---:|
| Direct paired-effect supervision | 0.34657 | 0.315 | 0.402 |
| Absolute outcomes then difference | 0.46049 | 0.090 | 0.176 |
| Unchanged DT-aware action | 0.50612 | 0.000 | - |

Mean paired absolute-minus-direct regret: **+0.11392**.
Trajectory-bootstrap 95% CI: [-0.00699, +0.22456].
Exact two-sided episode sign test: 8/12 nonzero trajectories favor direct paired supervision, p=0.387695.

## Trajectory-level comparison

| Episode | States | Direct regret | Absolute regret | Absolute - direct | Direct top-1 | Absolute top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 33 | 0.41983 | 0.16605 | -0.25378 | 0.303 | 0.212 |
| 1 | 35 | 0.20102 | 0.15452 | -0.04650 | 0.286 | 0.229 |
| 2 | 35 | 0.35104 | 0.59731 | +0.24627 | 0.429 | 0.143 |
| 3 | 33 | 0.52832 | 0.90945 | +0.38112 | 0.424 | 0.152 |
| 4 | 35 | 0.19776 | 0.54548 | +0.34772 | 0.543 | 0.229 |
| 5 | 35 | 0.57941 | 0.95324 | +0.37383 | 0.371 | 0.143 |
| 6 | 34 | 0.17367 | 0.38841 | +0.21473 | 0.382 | 0.118 |
| 7 | 35 | 0.36401 | 0.50012 | +0.13611 | 0.486 | 0.200 |
| 8 | 35 | 0.37378 | 0.16566 | -0.20812 | 0.457 | 0.229 |
| 9 | 34 | 0.22013 | 0.27102 | +0.05089 | 0.441 | 0.088 |
| 10 | 33 | 0.60176 | 0.55924 | -0.04252 | 0.424 | 0.121 |
| 11 | 33 | 0.15794 | 0.31405 | +0.15611 | 0.273 | 0.242 |

## Protocol integrity

- [x] The frozen 12 complete confirmation trajectories are present
- [x] Both methods rank identical state-action candidates
- [x] All six heads have exactly 56,457 trainable-stage parameters
- [x] The evaluation utility uses one common paired-training scale
- [x] All paired regret values are finite

## Frozen scientific-support criteria

- [x] Direct paired supervision has lower mean ranking regret
- [ ] The trajectory-bootstrap 95% interval is above zero
- [ ] At least 9 of 12 trajectories favor direct paired supervision
- [x] Direct paired top-1 agreement is not lower

Protocol integrity: **PASS**.
Evidence supports the paired-effect formulation contribution: **NO**.

A negative frozen result remains reportable and must not trigger seed, weight or threshold replacement.
