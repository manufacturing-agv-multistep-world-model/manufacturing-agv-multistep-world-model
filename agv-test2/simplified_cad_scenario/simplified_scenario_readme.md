# Simplified CAD-Derived AGV Scenario

This is the recommended middle-complexity scenario for the current paper stage.

## Design Principles

1. Preserve the real CAD trunk route:
   `A -> G1 -> G2 -> G2_G3_Mid -> G3 -> G4 -> G5 -> G6 -> B`
2. Avoid dense node splitting on long straight edges.
3. Keep added nodes and edges horizontal or vertical.
4. Use only necessary industrial function areas:
   - AGV homes
   - one charging bay
   - three workstations
   - one passing/waiting buffer
   - one material buffer
   - two warehouse slots

## Why This Version Is Safer Than the 36-Node Scenario

- Easier to reproduce in AnyLogic.
- Easier to justify from the CAD screenshot.
- Still complex enough for MARL/GNN because it has branches, bidirectional flows, bottlenecks, charging, and warehouse slotting.
- Less likely to be criticized as over-engineered or artificially complex.

## Suggested AnyLogic Build Order

1. Build the trunk first: `A-G1-G2-G2_G3_Mid-G3-G4-G5-G6-B`.
2. Run `A -> B` with 18 jobs.
3. Add `Home1-Home3` and `Charge`.
4. Add `P1_Packaging`, `P2_Labeling`, `PrepBuffer`, `PassingBuffer`, `MaterialBuffer`, and `W1/W2`.
5. Use `capacity_baseline` first.
6. Use `capacity_stress` for rush/deadlock experiments.

## Suggested Paper Wording

`The validation layout was abstracted from the plant CAD route. The CAD-derived trunk was preserved, while only necessary functional nodes were added through orthogonal branches to represent production stations, warehouse slots, charging, and waiting buffers. Long straight corridors were not over-discretized; only one intermediate control node was inserted on the longest corridor for occupancy-state observation.`
