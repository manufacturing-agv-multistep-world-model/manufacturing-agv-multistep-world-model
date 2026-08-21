# AnyLogic Independent Validation Checklist for JMS

Project: `JMS_AGV_DT_interpretable_params_2026-07-01`

Purpose:
AnyLogic is used as an independent discrete-event/material-handling validation platform, not as the main AI training platform. The JMS-safe claim is:

> The proposed Python high-fidelity DT and decision framework are externally checked in an AnyLogic model using the same road-network topology, AGV physical parameters, task flow, and fixed dispatch traces. The validation focuses on KPI trend consistency rather than exact numerical identity.

Status labels:

- `[ ]` Not started
- `[~]` In progress / generated but not audited
- `[x]` Completed and audited
- `[!]` Needs correction or rerun
- `[n/a]` Outside the locked validation scope

## G0. Validation Scope Lock

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G0.1 | Confirm AnyLogic version and runtime limit | Decide whether validation can use 1h/4h/8h or 1h/4h/5h | screenshot or note in `paper_outputs/anylogic_validation/` | Professional runtime used for the final 1h/4h/8h protocol; PLE 5 h runs retained only as supplementary history |
| [x] | G0.2 | Freeze validation scenario | Prevent topology mismatch with Python | `data/anylogic_nodes.csv`, `data/anylogic_edges.csv` | Uses the same 20-node simplified CAD-derived network as Python |
| [x] | G0.3 | Freeze validation claim wording | Avoid overclaiming | manuscript note | Independent DES trend validation under matched topology, kinematics, capacity, and stochastic task-flow assumptions; no battery or learned-policy validation claim |
| [x] | G0.4 | Lock real random-seed protocol | Prevent pseudoreplication | `docs/ANYLOGIC_MULTI_SEED_RUN_CARD_CN.md` | Simulation experiment fixed seed must equal `alRunSeed`; changing the label alone is invalid |

## G1. Python-to-AnyLogic Data Export

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G1.1 | Export road network CSV | Build the same AnyLogic nodes and paths | `AGV_DT_AnyLogic_Validation/data/anylogic_nodes.csv`, `anylogic_edges.csv` | The final AnyLogic model uses the matched 20-node kinematics/baseline network |
| [n/a] | G1.2 | Export fixed dispatch traces | Let AnyLogic replay decisions without reimplementing AI | `anylogic_dispatch_commands_*.csv` | Not required: the locked scope validates the independent DES capacity/congestion regime, not learned-policy replay |
| [x] | G1.3 | Export Python expected KPI | Provide comparison target | `paper_outputs/anylogic_validation/python_reference_runs.csv` | Contains 60 matched Python runs for steady/rush, 1/4/8 h, and 10 seeds |
| [x] | G1.4 | Export validation manifest | Make the run reproducible | `paper_outputs/anylogic_validation/final/anylogic_validation_manifest.json` | Records horizons, scenarios, seeds, row completeness, scope, and source-file hashes |

Recommended methods for AnyLogic validation:

- `DT-aware`: safest primary trace because it is rule-based and interpretable.
- `PI-GWM-GMAPPO`: optional trace to show the learned policy can be replayed independently.
- `Nearest`: optional baseline trace if time allows.

Recommended horizons:

- Preferred: `1h`, `4h`, `8h`.
- If AnyLogic PLE effectively limits runtime: `1h`, `4h`, `5h`, and keep Python main paper results at `1h/4h/8h`.

## G2. AnyLogic Model Geometry and Parameter Setup

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G2.1 | Build or verify 20 nodes in AnyLogic | Match Python road-network topology | AnyLogic GUI screenshot | 20-node simplified CAD-derived layout built and run successfully |
| [x] | G2.2 | Build or verify directed/undirected paths | Ensure AGV route feasibility | AnyLogic GUI screenshot | Network is connected and route execution succeeds |
| [x] | G2.3 | Set AGV fleet homes | Avoid startup errors | AnyLogic setting screenshot | 3 AGVs start at three valid home nodes in the same network group |
| [x] | G2.4 | Set physical parameters | Match high-fidelity DT assumptions | AnyLogic setting screenshot/table | Speed 1.2 m/s, acceleration/deceleration 0.5 m/s2, 8 s load/unload, and path capacities checked |
| [~] | G2.5 | Replace "collision" wording | Avoid overclaiming physical contact modeling | README/manuscript update | Working terminology uses conflict/blocking; final figures and manuscript still require audit |

## G3. Command-Replay Flow Setup

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G3.0 | Verify stochastic task-flow DES smoke tests | Validate steady/rush arrival modes before long runs | `Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv` | 600 s steady: 8 generated/7 completed; rush: 12 generated/7 completed; rush raises WIP, waiting, and AGV utilization |
| [n/a] | G3.1 | Import dispatch commands into AnyLogic database/table | Drive replay from Python traces | AnyLogic database screenshot | Outside the locked independent task-flow DES scope |
| [x] | G3.2 | Connect source / moveByTransporter / sink | Execute stochastic transport tasks | AnyLogic model screenshot | Agents move through the matched network using AGVFleet and complete source-to-sink jobs |
| [x] | G3.3 | Verify route execution | Catch node-space and fleet errors early | short-run screenshot/log | All formal runs finish without space, home-node, or free-home startup errors |
| [x] | G3.4 | Verify task-flow accounting | Preserve throughput logic | formal audit | Every formal row satisfies generated = completed + unfinished |

## G4. AnyLogic Validation Runs

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G4.1 | Run 1h validation | Check short-horizon consistency | `Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv` | Steady/rush completed with real seeds 1/2/3 |
| [x] | G4.2 | Run 4h validation | Check medium-horizon consistency | `Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv` | Steady/rush completed with real seeds 1/2/3 |
| [x] | G4.3 | Run 8h validation | Check longest independent horizon | `Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv` | Professional runs completed for steady/rush with real seeds 1/2/3 |
| [x] | G4.4 | Repeat every formal run with 3 real seeds | Reduce single-trace accident risk | `anylogic_formal_runs_clean.csv` | 18/18 combinations present; Simulation fixed seed matched `alRunSeed` during manual runs |

Minimum acceptable version for JMS submission:

- 1 method: `DT-aware`
- 3 horizons: `1h/4h/5h` or `1h/4h/8h`
- KPI comparison: throughput, UPH, travel distance or route count, waiting/blocking proxy

Stronger version:

- 2-3 methods: `Nearest`, `DT-aware`, `PI-GWM-GMAPPO`
- 3 horizons
- At least 3 traces/seeds per method-horizon pair

## G5. AnyLogic KPI Export

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G5.1 | Export AnyLogic KPI summary | Enable Python-side comparison | `Manufacturing_AGV_DT_Validation/anylogic_validation_results.csv` | Raw snapshot and 18-row formal clean dataset saved in `paper_outputs/anylogic_validation/final` |
| [ ] | G5.2 | Export optional trajectory/event log | Provide supplementary evidence | `anylogic_event_log.csv` or screenshot | Contains event timestamps and AGV ids |
| [x] | G5.3 | Save GUI evidence | Support "independent validation model" claim | screenshots in `paper_outputs/anylogic_validation/screenshots/raw/` | Named model-editor and representative rush-runtime screenshots preserve the network, AGV fleet, process flow, and Professional runtime title |

## G6. Python-vs-AnyLogic Comparison

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G6.1 | Merge Python and AnyLogic KPIs | Create comparison dataset | `python_anylogic_comparison.csv` | Final 1/4/8 h comparison generated from 3 AnyLogic and 10 Python seeds |
| [x] | G6.2 | Generate comparison table | Paper-ready validation evidence | comparison/summary CSV | Means, 95% CIs, and relative differences generated |
| [x] | G6.3 | Generate comparison figure | Visual proof of trend consistency | `figure_anylogic_validation.png` | PNG/PDF/SVG/TIFF generated with uncertainty for both platforms |
| [x] | G6.4 | Audit trend consistency | Avoid false precision | `ANYLOGIC_MULTI_SEED_AUDIT.md` | Maximum absolute UPH difference is 5.5%; congestion regimes agree; waiting-time magnitude differences are scope-limited |

Suggested acceptance thresholds:

- Throughput/UPH relative difference: ideally within 10-20% for replayed rule-based traces.
- Trend direction: must be consistent across horizons.
- Blocking/waiting: exact value may differ, but high/low congestion conclusions should not reverse.
- Energy: if AnyLogic does not model battery directly, report energy as Python-derived replay energy or omit direct AnyLogic energy claims.

Final audit supersedes the earlier seed-1 pilot note. The complete three-seed
1/4/8 h study supports throughput-capacity and congestion-regime consistency;
it does not support exact waiting-time or deadlock-count equivalence.

## G7. Manuscript Integration

| status | ID | task | purpose | output | acceptance check |
|---|---|---|---|---|---|
| [x] | G7.1 | Add AnyLogic validation subsection | Strengthen external-validity story | manuscript Sections 5.5 and 6.6 | Clearly states AnyLogic is independent DES capacity/congestion trend validation |
| [x] | G7.2 | Add validation table/figure | Make evidence visible | Figure 7 and Table 5 | Uses matched UPH, backlog, waiting-time, and uncertainty terminology |
| [x] | G7.3 | Add limitation sentence | Preempt reviewer attack | manuscript discussion | States that AnyLogic does not validate battery physics, learned-policy optimality, or exact deadlock counts |
| [ ] | G7.4 | Final wording audit | Avoid dangerous claims | revised manuscript | No unsupported "real-world deployment", "collision proof", "Nash/Pareto proof", or "optimal" claims |

## Step-by-Step Execution Order

1. Finish `G0`: lock whether the longest AnyLogic horizon is `8h` or `5h`.
2. Finish `G1`: generate the validation CSV package from Python.
3. Finish `G2`: verify AnyLogic network and AGV parameters.
4. Finish `G3`: run a 5-10 command replay smoke test.
5. Finish `G4`: run full validation horizons.
6. Finish `G5`: export AnyLogic KPI CSVs and screenshots.
7. Finish `G6`: generate Python-vs-AnyLogic comparison tables/figures.
8. Finish `G7`: put the validation into the manuscript with conservative wording.

## What To Send Back After Each AnyLogic Run

After each completed run, copy or show:

- AnyLogic final screenshot.
- Exported KPI CSV.
- AnyLogic console/log text if there are warnings.
- The exact stop time used, e.g., `3600 s`, `14400 s`, `18000 s`, or `28800 s`.
- Whether the run ended normally or was manually stopped.
