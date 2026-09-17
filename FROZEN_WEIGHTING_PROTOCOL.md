# Frozen Weighting Protocol

This file distinguishes the weights used by the final multistep evidence chain from legacy defaults and exploratory controllers. All values below were fixed before the corresponding independent evaluation or confirmation stage.

## Multistep transition backbone

The total loss is

```text
agent + 0.05 position + 0.05 target + node + global
      + 4.0 KPI + 0.50 physics
```

The six KPI outputs are ordered as `[reward, time, energy, blocking, deadlock, throughput]` and use component weights `(0.5, 8, 32, 2, 2, 2)`. The physics-referenced auxiliary term retains only `[time, energy, blocking]`, giving `(8, 32, 2)`. Each weighted component loss is divided by the sum of its active component weights.

Physics targets are normalized by `100 s`, `20 Wh`, and fleet size `3` for time, energy, and blocking, respectively. Five-step factorial training uses normalized temporal weights proportional to `0.90^(t-1)`. The later ten-step charge-congestion backbone uses weights proportional to `0.95^(t-1)`. The separate planning discount `0.95` is not a training-loss weight.

The `0.35` coefficient in `WORLD_MODEL_DEFAULTS` is retained only for the historical one-step baseline. `MULTISTEP_WORLD_MODEL_DEFAULTS`, the formal training scripts, and frozen multistep checkpoints use `0.50`.

## Counterfactual action-effect stage

The primary paired utility uses equal directional weights `1:1:1` for energy, completed tasks, and charging queue after train-only scale normalization. Lower energy, more completed tasks, and lower queue are preferred. Post-confirmation sensitivity uses four fixed settings: `1:1:1`, `2:1:1`, `1:2:1`, and `1:1:2`.

## Selective shadow authority

A shadow recommendation is issued only when all three frozen model initializations agree on the same nonbaseline action and the normalized utility margin is at least `0.15`. This gate controls recommendation coverage; it does not grant closed-loop authority.

## Scope

Earlier Graph-MAPPO reward coefficients and exploratory MPC utility weights are not part of the final confirmatory claim. They remain in legacy code only to preserve provenance.
