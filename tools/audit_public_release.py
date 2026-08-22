from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".txt", ".json", ".csv", ".xml", ".alpx", ".yml", ".yaml"}
REQUIRED_PATHS = (
    "README.md",
    "REPRODUCIBILITY.md",
    "DATA_MANIFEST.md",
    "CITATION.cff",
    "LICENSE",
    "DATA_LICENSE.md",
    "LICENSE_SCOPE.md",
    "AUTHORITATIVE_EVIDENCE_MAP.md",
    "EXPERIMENT_REGISTRY.csv",
    "agv_dt_env.py",
    "physics_graph_world_model_multistep_v11.py",
    "physics_graph_world_model_counterfactual_v141.py",
    "scripts/106_train_v11_independent_arrival_factorial.ps1",
    "scripts/107_evaluate_v11_arrival_v4_factorial.ps1",
    "scripts/121_train_counterfactual_world_model_v141_seeds.ps1",
    "scripts/124_confirm_counterfactual_ranking_v144.ps1",
    "scripts/125_confirm_counterfactual_shadow_v145.ps1",
    "scripts/135_confirm_v150_architecture_comparison.ps1",
    "scripts/137_confirm_paired_formulation_v151.ps1",
    "agv-test2/simplified_cad_scenario/simplified_scenario_config.json",
    "AGV_DT_AnyLogic_Validation/Manufacturing_AGV_DT_Validation/Manufacturing_AGV_DT_Validation.alpx",
    "experiment_results/v11_physics_factorial_arrival_v4_independent_v2",
    "experiment_results/world_model_counterfactual_v144_ranking_confirmation_v1",
    "experiment_results/world_model_counterfactual_v145_shadow_confirmation_parallel_v2",
    "experiment_results/v150_graph_vs_flat_confirmation_seed17400",
    "experiment_results/v151_paired_vs_absolute_confirmation_seed18400",
    "paper_outputs/anylogic_validation/final",
)
_RESTRICTED_TERMS = (
    "P" + "HDQ",
    "\u4e1c\u963f",
    "\u963f\u80f6",
    "eji" + "aoAGV731V2",
    "eji" + "aoagv713",
)
FORBIDDEN_PATTERNS = {
    "private Windows user path": re.compile(
        r"(?i)[A-Z]:[\\/](?:Users[\\/])?(?:HUA" + r"WEI|qiao" + r"zhiqi)(?:[\\/]|$)"
    ),
    "restricted project marker": re.compile("|".join(map(re.escape, _RESTRICTED_TERMS)), re.IGNORECASE),
}
GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def main() -> int:
    failures: list[str] = []
    ignored_parts = {".git", "__pycache__", ".pytest_cache", ".venv"}

    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            failures.append(f"Missing required path: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.stat().st_size >= GITHUB_FILE_LIMIT:
            failures.append(f"GitHub file limit exceeded: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {relative}")

    if failures:
        print("Public-release audit: FAIL")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1

    included_files = [
        path for path in ROOT.rglob("*") if path.is_file() and not ignored_parts.intersection(path.parts)
    ]
    file_count = len(included_files)
    size_mb = sum(path.stat().st_size for path in included_files) / (1024**2)
    print(f"Public-release audit: PASS ({file_count} files, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
