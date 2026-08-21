# AnyLogic multi-seed validation audit

- Formal rows retained: 18
- Unique real-seed labels: 3
- Required combinations missing: 0
- Data-integrity warnings: 0

## Integrity findings

- No aggregate-level integrity violation detected.

## Scope lock

- This model validates road-network/kinematics and congestion trends under matched stochastic task flows.
- It does not independently validate nonlinear battery physics, learned-policy optimality, or exact deadlock counts.
- Queueing-time magnitudes need not be identical because event ordering, path reservation, and service-time semantics differ between engines.
- Seed labels are credible only when the AnyLogic Simulation experiment fixed seed equals `alRunSeed` for every run.

## Primary validation result

- Maximum absolute cross-platform UPH difference: 5.5%.
- Both engines reproduce a stable steady regime and a capacity-saturated rush regime with growing backlog.
- The evidence supports system-level capacity and congestion-trend validity, not pointwise queueing-time equivalence.

## Cross-platform relative differences

- rush 1 h: UPH -1.2%; waiting time +8.5%; backlog -21.8%.
- rush 4 h: UPH -1.4%; waiting time +23.4%; backlog -10.0%.
- rush 8 h: UPH -3.1%; waiting time +35.4%; backlog -4.2%.
- steady 1 h: UPH +2.2%; waiting time -46.1%; backlog -4.8%.
- steady 4 h: UPH -5.5%; waiting time -52.6%; backlog -0.7%.
- steady 8 h: UPH +0.4%; waiting time -44.3%; backlog -14.0%.
