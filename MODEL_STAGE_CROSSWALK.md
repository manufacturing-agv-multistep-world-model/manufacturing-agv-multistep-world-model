# Model and Experiment Name Crosswalk

The manuscript uses descriptive scientific names. The repository retains immutable engineering identifiers in filenames, checkpoint metadata, scripts, and result directories so that published claims remain traceable to frozen artifacts.

| Internal identifier | Descriptive manuscript name | Role |
|---|---|---|
| `V10` | Earlier engineering implementation | Source of fixed engineering-priority loss weights |
| `V11` | Five-step physics-factorial backbone | Equal-parameter factorial of physical features and physics-referenced auxiliary supervision |
| `V12` | Ten-step charge-congestion backbone | Extends the transition backbone with a jointly trained congestion head |
| `V13` | Frozen future-risk backbone | Adds the auxiliary 80-decision-step charging-risk representation target |
| `V14.1` | Paired counterfactual action-effect model | Predicts candidate-minus-baseline effects at 120, 360, and 720 seconds |
| `V15.0` | Head-budget-matched nongraph comparison | Tests the representation boundary against a flat multilayer perceptron |
| `M1` | Physics-factorial experiment | Evaluates multistep prediction effects under equal trainable parameters |
| `N6` | Action-ranking confirmation | Confirms rule-relative candidate ranking on unseen trajectories |
| `F1` | Paired-formulation confirmation | Compares direct paired-effect and absolute-outcome formulations |
| `N13` | Nongraph-representation confirmation | Compares the graph representation with a head-budget-matched flat model |
| `N7` | Shadow-advice confirmation | Evaluates frozen selective recommendations without direct control authority |
| `A1` | Independent AnyLogic validation | Checks capacity and congestion trends in a separate simulation platform |

These identifiers must not be renamed inside reproducibility commands or archived paths. They are provenance keys, not scientific claims or model names intended for readers.
