# Reproducibility Guide

## 1. Reference Environment

- Python 3.9 (the final CPU verification used Python 3.9.9)
- Windows PowerShell 5.1 or PowerShell 7
- PyTorch 2.5-2.7
- Optional NVIDIA CUDA GPU for long training and counterfactual runs
- AnyLogic Professional 8.9.8 for the independent 8 h validation runs

Install the pinned Python dependencies with `pip install -r requirements.txt`.

## 2. Integrity First

Run these short checks before starting an experiment:

```powershell
python .\tools\audit_public_release.py
powershell -ExecutionPolicy Bypass -File .\scripts\00_run_multistep_core_tests.ps1
```

The audit verifies required evidence, prohibited private paths, and GitHub file-size limits. The test suite checks environment, model, and experiment invariants.

## 3. Authoritative Evidence

The claim-to-file mapping is frozen in `AUTHORITATIVE_EVIDENCE_MAP.md`. Primary evidence directories are:

| Evidence | Directory |
|---|---|
| Equal-parameter physics factorial | `experiment_results/v11_physics_factorial_arrival_v4_independent_v2/` |
| Unseen-trajectory action ranking | `experiment_results/world_model_counterfactual_v144_ranking_confirmation_v1/` |
| Agreement-gated shadow evaluation | `experiment_results/world_model_counterfactual_v145_shadow_confirmation_parallel_v2/` |
| Graph versus parameter-matched flat model boundary | `experiment_results/v150_graph_vs_flat_confirmation_seed17400/` |
| Paired-effect versus absolute-outcome boundary | `experiment_results/v151_paired_vs_absolute_confirmation_seed18400/` |
| Utility-weight sensitivity | `experiment_results/v151_utility_sensitivity_v1/` |
| AnyLogic independent validation | `paper_outputs/anylogic_validation/final/` |

## 4. Numbered Entry Points

The scripts preserve the experiment sequence used during development. The most relevant final-stage entry points are:

| Script | Purpose | Typical cost |
|---|---|---|
| `scripts/106_train_v11_independent_arrival_factorial.ps1` | Train the equal-parameter multistep physics factorial | Long, GPU recommended |
| `scripts/107_evaluate_v11_arrival_v4_factorial.ps1` | Evaluate the factorial on paired episodes | Medium |
| `scripts/121_train_counterfactual_world_model_v141_seeds.ps1` | Train frozen counterfactual action-effect models | Long, GPU recommended |
| `scripts/124_confirm_counterfactual_ranking_v144.ps1` | Independent action-ranking confirmation | Long, GPU recommended |
| `scripts/125_confirm_counterfactual_shadow_v145.ps1` | Frozen shadow/advice confirmation | Long |
| `scripts/133_train_flat_counterfactual_baseline_v150_seeds.ps1` | Train parameter-matched flat baselines | Long, GPU recommended |
| `scripts/135_confirm_v150_architecture_comparison.ps1` | Confirm graph-versus-flat boundary result | Long |
| `scripts/136_train_absolute_outcome_baseline_v151_seeds.ps1` | Train absolute-outcome comparison | Long, GPU recommended |
| `scripts/137_confirm_paired_formulation_v151.ps1` | Confirm paired-effect formulation comparison | Long |
| `scripts/138_run_v151_utility_sensitivity.ps1` | Analyze utility-weight sensitivity | Short |
| `scripts/47_run_anylogic_python_reference.ps1` | Generate Python-side reference trends | Short |
| `scripts/48_analyze_anylogic_validation.ps1` | Compare the AnyLogic export with Python trends | Short |

Run `Get-Help` or open a script before execution to confirm its arguments. Outputs are written under `experiment_results/` or `paper_outputs/` using repository-relative paths.

## 5. Frozen Statistical Protocol

- Environment episodes, not individual decision states, are the resampling unit.
- Where three initialization seeds are present, errors are averaged across model seeds within each paired evaluation episode before episode-level bootstrap resampling.
- Development runs select designs; confirmation seeds and thresholds remain frozen.
- Negative confirmation results are retained and reported rather than replaced by favorable seeds.
- Candidate-ranking claims apply to the frozen feasible candidate-generation protocol, not exhaustive optimization of all joint actions.

## 6. AnyLogic Validation

Open `AGV_DT_AnyLogic_Validation/Manufacturing_AGV_DT_Validation/Manufacturing_AGV_DT_Validation.alpx` in AnyLogic Professional 8.9.8. The project writes `anylogic_validation_results.csv` beside the model file. Run steady and rush scenarios for 1 h, 4 h, and 8 h with seeds 1, 2, and 3, then execute script 48.

The AnyLogic evidence validates capacity and congestion trends only. It does not reproduce the learned controller or independently validate learned world-model accuracy.
