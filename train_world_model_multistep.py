from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import importlib
import json
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from agv_case_env import AGV_A_Charge_Env
from physics_graph_world_model import collect_world_model_transitions
from physics_graph_world_model_multistep import (
    MODEL_VERSION as V9_MODEL_VERSION,
    MultiStepSequenceDataset,
    PhysicsInformedGraphWorldModelMultiStep,
    build_sequence_samples,
    multistep_world_model_loss,
    save_multistep_world_model,
)
from physics_graph_world_model_multistep_v10 import (
    MODEL_VERSION as V10_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV10,
    multistep_world_model_loss_v10,
    save_multistep_world_model_v10,
)
from physics_graph_world_model_multistep_v11 import (
    MODEL_VERSION as V11_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV11,
    build_physical_graph_features,
    multistep_world_model_loss_v11,
    save_multistep_world_model_v11,
)
from physics_graph_world_model_multistep_v12 import (
    MODEL_VERSION as V12_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV12,
    multistep_world_model_loss_v12,
    save_multistep_world_model_v12,
)
from physics_graph_world_model_multistep_v13 import (
    MODEL_VERSION as V13_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV13,
    annotate_future_congestion_risk,
    future_risk_positive_weights,
    multistep_world_model_loss_v13,
    save_multistep_world_model_v13,
)
from physics_graph_world_model_multistep_v14 import (
    MODEL_VERSION as V14_MODEL_VERSION,
    PhysicsInformedGraphWorldModelMultiStepV14,
    annotate_future_terminal_kpis,
    future_terminal_positive_weights,
    future_terminal_scales,
    multistep_world_model_loss_v14,
    save_multistep_world_model_v14,
)
from train_world_model import metadata_from_sample, split_samples
from jms_parameter_registry import MULTISTEP_WORLD_MODEL_DEFAULTS


ROOT = Path(__file__).resolve().parent
TRANSITION_SCHEMA_VERSION = "assignment_visible_congestion_independent_arrival_streams_v4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an autoregressive physics-informed graph world model."
    )
    parser.add_argument("--episodes", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["episodes"])
    parser.add_argument("--max-steps", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["max_steps"])
    parser.add_argument("--epochs", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["batch_size"])
    parser.add_argument("--learning-rate", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["weight_decay"])
    parser.add_argument("--physics-weight", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["physics_weight"])
    parser.add_argument("--rollout-discount", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["rollout_discount"])
    parser.add_argument("--training-horizon", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["training_horizon"])
    parser.add_argument("--sequence-stride", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["sequence_stride"])
    parser.add_argument("--teacher-forcing-start", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["teacher_forcing_start"])
    parser.add_argument("--teacher-forcing-end", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["teacher_forcing_end"])
    parser.add_argument("--exploration-rate", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["exploration_rate"])
    parser.add_argument("--hidden-dim", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["hidden_dim"])
    parser.add_argument("--planning-horizon", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["planning_horizon"])
    parser.add_argument("--beam-width", type=int, default=MULTISTEP_WORLD_MODEL_DEFAULTS["beam_width"])
    parser.add_argument("--planning-discount", type=float, default=MULTISTEP_WORLD_MODEL_DEFAULTS["planning_discount"])
    parser.add_argument("--agv-count", type=int, default=3)
    parser.add_argument("--env-variant", choices=["ideal", "kinematics", "full"], default="full")
    parser.add_argument("--reward-mode", choices=["individual", "global", "hybrid"], default="hybrid")
    parser.add_argument("--scenario", choices=["steady", "rush"], default="rush")
    parser.add_argument(
        "--dispatch-rule",
        choices=["fcfs", "nearest", "priority", "dt_aware", "dt_marl"],
        default="dt_aware",
    )
    parser.add_argument("--capacity-mode", choices=["baseline", "stress"], default="stress")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="Trajectory-generation seed. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Episode-group split seed. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--transition-cache",
        default=None,
        help="Optional gzip-pickle cache shared across model seeds.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="Maximum PyTorch CPU worker threads used alongside GPU training.",
    )
    parser.add_argument(
        "--low-priority",
        action="store_true",
        help="Use below-normal process priority on Windows to keep the computer responsive.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision. Ignored on CPU.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of silently falling back when CUDA is unavailable.",
    )
    parser.add_argument(
        "--model-variant",
        choices=["v9", "v10", "v11", "v12", "v13", "v14"],
        default="v9",
        help=(
            "V10 adds local actions; V11 adds physical edges, local graph context, and "
            "discrete nodes; V12 separates route blocking from charger-queue congestion; "
            "V13 adds direct long-horizon congestion-risk prediction; V14 adds "
            "action-conditioned terminal energy, throughput, and charge-queue heads."
        ),
    )
    parser.add_argument(
        "--physical-feature-mode",
        choices=["full", "zero"],
        default="full",
        help=(
            "V11 factorial-ablation switch. 'zero' preserves the architecture and "
            "parameter count but replaces all engineered node/edge physical features "
            "with zeros. It does not alter the trajectory data."
        ),
    )
    parser.add_argument(
        "--future-risk-horizon",
        type=int,
        default=80,
        help="Complete low-level transition window used by the V13 future-risk labels.",
    )
    parser.add_argument(
        "--future-terminal-horizon",
        type=int,
        default=80,
        help="Complete action-conditioned window used by V14 terminal KPI labels.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Optional trusted V12 checkpoint for V13, or trusted V13 checkpoint "
            "for V14 initialization."
        ),
    )
    parser.add_argument(
        "--freeze-v13-backbone",
        action="store_true",
        help="Freeze the validated V12 backbone and train only the V13 future-risk head.",
    )
    parser.add_argument(
        "--freeze-v14-backbone",
        action="store_true",
        help="Freeze the validated V13 model and train only the V14 terminal head.",
    )
    parser.add_argument("--output-dir", default="world_model_runs/pi_gwm_multistep_v9_seed42")
    return parser


def select_device(name: str, require_cuda: bool = False) -> str:
    if name == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
        if require_cuda and selected != "cuda":
            raise RuntimeError(
                "CUDA is required but this Python environment has no CUDA-enabled PyTorch."
            )
        return selected
    if name == "cuda" and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build on the GPU computer."
            )
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return name


def validate_args(args: argparse.Namespace) -> None:
    if args.training_horizon < 2:
        raise ValueError("training-horizon must be at least 2")
    if not 0.0 <= args.teacher_forcing_end <= args.teacher_forcing_start <= 1.0:
        raise ValueError("Teacher-forcing ratios must satisfy 0 <= end <= start <= 1")
    if not 0.0 < args.rollout_discount <= 1.0:
        raise ValueError("rollout-discount must be in (0, 1]")
    if args.planning_horizon < 2 or args.beam_width < 1:
        raise ValueError("Planning horizon must be at least 2 and beam width must be positive")
    if not 0.0 < args.planning_discount <= 1.0:
        raise ValueError("planning-discount must be in (0, 1]")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if args.physics_weight < 0.0:
        raise ValueError("physics-weight must be non-negative")
    if args.physical_feature_mode != "full" and args.model_variant != "v11":
        raise ValueError(
            "physical-feature-mode=zero is restricted to the preregistered V11 ablation"
        )
    if args.future_risk_horizon < 1:
        raise ValueError("future-risk-horizon must be positive")
    if args.future_terminal_horizon < 1:
        raise ValueError("future-terminal-horizon must be positive")
    if args.init_checkpoint and args.model_variant not in {"v13", "v14"}:
        raise ValueError("init-checkpoint is supported only for V13 and V14")
    if args.freeze_v13_backbone and args.model_variant != "v13":
        raise ValueError("freeze-v13-backbone is valid only for V13")
    if args.freeze_v14_backbone and args.model_variant != "v14":
        raise ValueError("freeze-v14-backbone is valid only for V14")
    if args.freeze_v13_backbone and args.freeze_v14_backbone:
        raise ValueError("Only one version-specific backbone freeze may be active")


def configure_cpu_runtime(cpu_threads: int, low_priority: bool) -> None:
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    except RuntimeError:
        # PyTorch allows setting inter-op threads only before parallel work starts.
        pass
    if low_priority and os.name == "nt":
        below_normal_priority_class = 0x00004000
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.kernel32.SetPriorityClass(
            process, below_normal_priority_class
        ):
            print("Warning: unable to lower Windows process priority.")


def make_env_factory(args: argparse.Namespace):
    def factory(seed: int) -> AGV_A_Charge_Env:
        return AGV_A_Charge_Env(
            agv_count=args.agv_count,
            env_variant=args.env_variant,
            reward_mode=args.reward_mode,
            scenario=args.scenario,
            dispatch_rule=args.dispatch_rule,
            capacity_mode=args.capacity_mode,
            max_steps=args.max_steps,
            seed=seed,
        )

    return factory


def transition_cache_signature(args: argparse.Namespace, data_seed: int) -> Dict[str, object]:
    return {
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "exploration_rate": args.exploration_rate,
        "agv_count": args.agv_count,
        "env_variant": args.env_variant,
        "reward_mode": args.reward_mode,
        "scenario": args.scenario,
        "dispatch_rule": args.dispatch_rule,
        "capacity_mode": args.capacity_mode,
        "data_seed": data_seed,
    }


def split_v12_congestion_stratified(
    transitions: List[Dict[str, np.ndarray]],
    seed: int,
    train_ratio: float = 0.82,
) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Keep complete episodes while placing charge events in both data splits."""

    episodes: Dict[int, List[Dict[str, np.ndarray]]] = {}
    for transition in transitions:
        episode_id = int(np.asarray(transition["episode_id"]).item())
        episodes.setdefault(episode_id, []).append(transition)
    event_episode_ids = [
        episode_id
        for episode_id, rows in episodes.items()
        if any(float(row.get("congestion_kpi", np.zeros(2))[1]) > 0.0 for row in rows)
    ]
    if len(event_episode_ids) < 2:
        raise ValueError(
            "V12 requires charger-queue observations in at least two complete episodes "
            "so training and validation remain leakage-free."
        )

    rng = np.random.default_rng(seed)
    all_ids = np.asarray(sorted(episodes), dtype=np.int64)
    rng.shuffle(all_ids)
    event_ids = np.asarray(event_episode_ids, dtype=np.int64)
    rng.shuffle(event_ids)
    valid_target = max(1, len(all_ids) - int(len(all_ids) * train_ratio))
    valid_ids = {int(event_ids[0])}
    reserved_train_event = int(event_ids[1])
    for episode_id in all_ids:
        value = int(episode_id)
        if len(valid_ids) >= valid_target:
            break
        if value != reserved_train_event:
            valid_ids.add(value)
    train_ids = set(int(value) for value in all_ids) - valid_ids
    train = [row for episode_id in train_ids for row in episodes[episode_id]]
    valid = [row for episode_id in valid_ids for row in episodes[episode_id]]
    return train, valid


def load_or_collect_transitions(
    args: argparse.Namespace,
    data_seed: int,
) -> tuple[List[Dict[str, np.ndarray]], str]:
    cache_path = Path(args.transition_cache) if args.transition_cache else None
    if cache_path is not None and not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    signature = transition_cache_signature(args, data_seed)
    if cache_path is not None and cache_path.exists():
        print(f"Loading shared transition cache: {cache_path.resolve()}")
        # NumPy 2.x writes private ``numpy._core`` module paths into pickle
        # payloads. NumPy 1.x exposes the same trusted array constructors under
        # ``numpy.core``; these aliases keep frozen local caches portable.
        try:
            importlib.import_module("numpy._core")
            needs_numpy_pickle_alias = False
        except ModuleNotFoundError:
            needs_numpy_pickle_alias = True
        if needs_numpy_pickle_alias:
            module_aliases = {
                "numpy._core": "numpy.core",
                "numpy._core.multiarray": "numpy.core.multiarray",
                "numpy._core.numeric": "numpy.core.numeric",
                "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
            }
            for modern_name, legacy_name in module_aliases.items():
                sys.modules.setdefault(modern_name, importlib.import_module(legacy_name))
        with gzip.open(cache_path, "rb") as stream:
            cached = pickle.load(stream)
        if cached.get("signature") != signature:
            raise ValueError(
                "Transition-cache settings do not match this run. Use a new cache path "
                "or remove the incompatible cache."
            )
        return cached["transitions"], "loaded"

    transitions = collect_world_model_transitions(
        env_factory=make_env_factory(args),
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=data_seed,
        exploration_rate=args.exploration_rate,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving shared transition cache: {cache_path.resolve()}")
        with gzip.open(cache_path, "wb", compresslevel=3) as stream:
            pickle.dump(
                {"signature": signature, "transitions": transitions},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return transitions, "generated"


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    integer_keys = {"actions", "episode_id", "start_transition_id"}
    non_blocking = device == "cuda"
    return {
        key: value.long().to(device, non_blocking=non_blocking)
        if key in integer_keys
        else value.float().to(device, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def teacher_forcing_ratio(args: argparse.Namespace, epoch: int) -> float:
    if args.epochs <= 1:
        return float(args.teacher_forcing_end)
    progress = (epoch - 1) / float(args.epochs - 1)
    return float(
        args.teacher_forcing_start
        + progress * (args.teacher_forcing_end - args.teacher_forcing_start)
    )


def average_parts(totals: Dict[str, float], batches: int) -> Dict[str, float]:
    return {key: value / max(batches, 1) for key, value in totals.items()}


def training_components(model_variant: str) -> tuple[Any, Any, Any, str, str]:
    if model_variant == "v14":
        return (
            PhysicsInformedGraphWorldModelMultiStepV14,
            multistep_world_model_loss_v14,
            save_multistep_world_model_v14,
            V14_MODEL_VERSION,
            "jms_v14_dual_timescale_terminal_efficiency",
        )
    if model_variant == "v13":
        return (
            PhysicsInformedGraphWorldModelMultiStepV13,
            multistep_world_model_loss_v13,
            save_multistep_world_model_v13,
            V13_MODEL_VERSION,
            "jms_v13_multiscale_physical_graph_congestion_risk",
        )
    if model_variant == "v12":
        return (
            PhysicsInformedGraphWorldModelMultiStepV12,
            multistep_world_model_loss_v12,
            save_multistep_world_model_v12,
            V12_MODEL_VERSION,
            "jms_v12_charge_aware_congestion_attribution",
        )
    if model_variant == "v11":
        return (
            PhysicsInformedGraphWorldModelMultiStepV11,
            multistep_world_model_loss_v11,
            save_multistep_world_model_v11,
            V11_MODEL_VERSION,
            "jms_v11_physical_edges_local_graph_discrete_nodes",
        )
    if model_variant == "v10":
        return (
            PhysicsInformedGraphWorldModelMultiStepV10,
            multistep_world_model_loss_v10,
            save_multistep_world_model_v10,
            V10_MODEL_VERSION,
            "jms_v10_action_conditioned_engineering_balanced",
        )
    return (
        PhysicsInformedGraphWorldModelMultiStep,
        multistep_world_model_loss,
        save_multistep_world_model,
        V9_MODEL_VERSION,
        "jms_v9_multistep_interpretable",
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    args: argparse.Namespace,
    loss_function: Any,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with autocast_context(bool(args.amp and device == "cuda")):
                output = model.rollout(batch, teacher_forcing_ratio=0.0)
                _, parts = loss_function(
                    output,
                    batch,
                    physics_weight=args.physics_weight,
                    discount=args.rollout_discount,
                )
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
    return average_parts(totals, batches)


def write_history(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    model_class, loss_function, save_function, model_version, parameter_profile = (
        training_components(args.model_variant)
    )
    configure_cpu_runtime(args.cpu_threads, args.low_priority)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device, require_cuda=args.require_cuda)
    amp_enabled = bool(args.amp and device == "cuda")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected training device: {device}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime reported by PyTorch: {torch.version.cuda}")
        torch.set_float32_matmul_precision("high")
    print(f"CUDA mixed precision enabled: {amp_enabled}")
    print(f"PyTorch CPU thread limit: {torch.get_num_threads()}")
    print(f"Below-normal Windows process priority requested: {args.low_priority}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_seed = args.seed if args.data_seed is None else args.data_seed
    split_seed = args.seed if args.split_seed is None else args.split_seed
    print("Preparing complete DT trajectories for multi-step supervision...")
    print(f"Model seed={args.seed}, data seed={data_seed}, split seed={split_seed}")
    collection_started = time.perf_counter()
    transitions, transition_source = load_or_collect_transitions(args, data_seed)
    if args.model_variant in {"v13", "v14"}:
        transitions = annotate_future_congestion_risk(
            transitions, horizon=args.future_risk_horizon
        )
    if args.model_variant == "v14":
        transitions = annotate_future_terminal_kpis(
            transitions, horizon=args.future_terminal_horizon
        )
    congestion_targets = np.stack(
        [
            transition.get("congestion_kpi", np.zeros(2, dtype=np.float32))
            for transition in transitions
        ]
    )
    congestion_positive_counts = (congestion_targets > 0.0).sum(axis=0)
    print(
        "Congestion supervision events: "
        f"route={int(congestion_positive_counts[0])}, "
        f"charge_queue={int(congestion_positive_counts[1])}"
    )
    if args.model_variant in {"v12", "v13", "v14"} and int(congestion_positive_counts[1]) == 0:
        raise ValueError(
            "V12 requires at least one observed charger-queue event. Increase rush-scenario "
            "trajectory coverage or exploration instead of fitting an unidentifiable head."
        )
    if args.model_variant in {"v12", "v13", "v14"}:
        train_transitions, valid_transitions = split_v12_congestion_stratified(
            transitions, seed=split_seed
        )
    else:
        train_transitions, valid_transitions = split_samples(transitions, seed=split_seed)
    train_sequences = build_sequence_samples(
        train_transitions, horizon=args.training_horizon, stride=args.sequence_stride
    )
    valid_sequences = build_sequence_samples(
        valid_transitions, horizon=args.training_horizon, stride=args.sequence_stride
    )
    print(
        f"Collected {len(transitions)} transitions; constructed "
        f"{len(train_sequences)} train and {len(valid_sequences)} validation sequences."
    )
    collection_time_sec = time.perf_counter() - collection_started
    print(
        f"Trajectory source={transition_source}; data preparation time: "
        f"{collection_time_sec:.1f} s"
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_sampler = None
    future_pos_weight = None
    terminal_scale = None
    terminal_positive_weight = None
    if args.model_variant in {"v13", "v14"}:
        future_pos_weight = future_risk_positive_weights(train_transitions)
        valid_targets = np.stack(
            [row["future_congestion_risk"] for row in valid_transitions]
        )
        valid_masks = np.stack(
            [row["future_congestion_risk_mask"] for row in valid_transitions]
        ) > 0.0
        train_targets = np.stack(
            [row["future_congestion_risk"] for row in train_transitions]
        )
        train_masks = np.stack(
            [row["future_congestion_risk_mask"] for row in train_transitions]
        ) > 0.0
        train_prevalence = [
            float(np.mean(train_targets[train_masks[:, index], index]))
            for index in range(train_targets.shape[1])
        ]
        valid_prevalence = [
            float(np.mean(valid_targets[valid_masks[:, index], index]))
            for index in range(valid_targets.shape[1])
        ]
        print(
            f"{args.model_variant.upper()} future-risk supervision: "
            f"horizon={args.future_risk_horizon}, "
            f"train_prevalence={train_prevalence}, "
            f"valid_prevalence={valid_prevalence}, "
            f"positive_weights={future_pos_weight.tolist()}"
        )
    if args.model_variant == "v14":
        terminal_scale = future_terminal_scales(train_transitions)
        terminal_positive_weight = future_terminal_positive_weights(train_transitions)
        train_terminal_targets = np.stack(
            [row["future_terminal_kpi"] for row in train_transitions]
        )
        train_terminal_masks = np.stack(
            [row["future_terminal_kpi_mask"] for row in train_transitions]
        ) > 0.0
        valid_terminal_masks = np.stack(
            [row["future_terminal_kpi_mask"] for row in valid_transitions]
        ) > 0.0
        print(
            "V14 terminal supervision: "
            f"horizon={args.future_terminal_horizon}, "
            f"training_scales={terminal_scale.tolist()}, "
            f"positive_weights={terminal_positive_weight.tolist()}, "
            f"train_complete={int(np.all(train_terminal_masks, axis=1).sum())}, "
            f"valid_complete={int(np.all(valid_terminal_masks, axis=1).sum())}, "
            f"train_target_mean={np.mean(train_terminal_targets[train_terminal_masks].reshape(-1)):.4f}"
        )
    if args.model_variant == "v12":
        charge_event_mask = np.asarray(
            [
                bool(np.any(sequence["target_congestion_kpi"][:, 1] > 0.0))
                for sequence in train_sequences
            ],
            dtype=bool,
        )
        event_count = int(charge_event_mask.sum())
        non_event_count = int(len(charge_event_mask) - event_count)
        if event_count == 0:
            raise ValueError(
                "V12 training split contains no charger-queue sequence."
            )
        # Data-derived weighting targets 25% rare-event windows and is capped
        # to avoid unstable gradients or unrealistic event prevalence.
        event_weight = min(
            max((0.25 / 0.75) * non_event_count / max(event_count, 1), 1.0),
            25.0,
        )
        sample_weights = np.where(charge_event_mask, event_weight, 1.0)
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_sequences),
            replacement=True,
            generator=generator,
        )
        print(
            "V12 rare-event sampler: "
            f"charge_sequences={event_count}/{len(train_sequences)}, "
            f"event_weight={event_weight:.2f}"
        )
    train_loader = DataLoader(
        MultiStepSequenceDataset(train_sequences),
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        pin_memory=device == "cuda",
    )
    valid_loader = DataLoader(
        MultiStepSequenceDataset(valid_sequences),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
    )

    metadata = metadata_from_sample(transitions[0], hidden_dim=args.hidden_dim)
    if args.model_variant == "v14":
        physical_env = make_env_factory(args)(data_seed)
        node_physical, edge_physical = build_physical_graph_features(physical_env)
        model = model_class(
            metadata,
            node_physical,
            edge_physical,
            future_risk_pos_weight=future_pos_weight,
            future_risk_horizon=args.future_risk_horizon,
            future_terminal_scale=terminal_scale,
            future_terminal_positive_weight=terminal_positive_weight,
            future_terminal_horizon=args.future_terminal_horizon,
        ).to(device)
    elif args.model_variant == "v13":
        physical_env = make_env_factory(args)(data_seed)
        node_physical, edge_physical = build_physical_graph_features(physical_env)
        model = model_class(
            metadata,
            node_physical,
            edge_physical,
            future_risk_pos_weight=future_pos_weight,
            future_risk_horizon=args.future_risk_horizon,
        ).to(device)
    elif args.model_variant in {"v11", "v12"}:
        physical_env = make_env_factory(args)(data_seed)
        node_physical, edge_physical = build_physical_graph_features(physical_env)
        if args.physical_feature_mode == "zero":
            node_physical = np.zeros_like(node_physical)
            edge_physical = np.zeros_like(edge_physical)
            print(
                "V11 physical-feature ablation active: engineered node and edge "
                "features are zeroed; tensor shapes and parameter count are unchanged."
            )
        else:
            print("V11/V12 physical-feature mode: full engineered node and edge features.")
        model = model_class(metadata, node_physical, edge_physical).to(device)
    else:
        model = model_class(metadata).to(device)
    if args.init_checkpoint:
        checkpoint_path = Path(args.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected_version = (
            V12_MODEL_VERSION if args.model_variant == "v13" else V13_MODEL_VERSION
        )
        if checkpoint.get("model_version") != expected_version:
            raise ValueError(
                f"{args.model_variant.upper()} initialization requires a trusted "
                f"{expected_version} checkpoint"
            )
        incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if args.model_variant == "v13":
            allowed_missing = {
                "future_risk_pos_weight",
                "future_risk_head.0.weight",
                "future_risk_head.0.bias",
                "future_risk_head.3.weight",
                "future_risk_head.3.bias",
            }
        else:
            allowed_missing = {
                "future_terminal_scale",
                "future_terminal_positive_weight",
                "future_terminal_head.0.weight",
                "future_terminal_head.0.bias",
                "future_terminal_head.3.weight",
                "future_terminal_head.3.bias",
            }
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                f"Initialization mismatch for {args.model_variant.upper()}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        print(
            f"Initialized {args.model_variant.upper()} backbone from "
            f"{checkpoint_path.resolve()}"
        )
        if args.model_variant == "v14":
            model.future_risk_pos_weight.copy_(
                torch.as_tensor(
                    future_pos_weight,
                    dtype=model.future_risk_pos_weight.dtype,
                    device=model.future_risk_pos_weight.device,
                )
            )
            print("Restored V14 future-risk class weights from the current training split.")
    if args.freeze_v13_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.future_risk_head.parameters():
            parameter.requires_grad = True
        print("Frozen V12 backbone; training only the V13 future charge-risk head.")
    if args.freeze_v14_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.future_terminal_head.parameters():
            parameter.requires_grad = True
        print("Frozen V13 model; training only the V14 terminal-efficiency head.")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}/{total_parameters:,}")
    print(f"Training batches per epoch: {len(train_loader)}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(amp_enabled)
    history: List[Dict[str, float]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        ratio = teacher_forcing_ratio(args, epoch)
        totals: Dict[str, float] = {}
        batches = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            with autocast_context(amp_enabled):
                output = model.rollout(batch, teacher_forcing_ratio=ratio)
                loss, parts = loss_function(
                    output,
                    batch,
                    physics_weight=args.physics_weight,
                    discount=args.rollout_discount,
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1

        train_time_sec = time.perf_counter() - epoch_started
        train_parts = {
            f"train_{key}": value for key, value in average_parts(totals, batches).items()
        }
        valid_parts = {
            f"valid_{key}": value
            for key, value in evaluate(model, valid_loader, device, args, loss_function).items()
        }
        row = {
            "epoch": float(epoch),
            "teacher_forcing_ratio": ratio,
            "epoch_time_sec": time.perf_counter() - epoch_started,
            "train_time_sec": train_time_sec,
            "train_batches_per_sec": batches / max(train_time_sec, 1.0e-9),
            **train_parts,
            **valid_parts,
        }
        history.append(row)
        selection_metric = {
            "v13": "valid_future_risk_loss",
            "v14": "valid_future_terminal_loss",
        }.get(args.model_variant, "valid_loss")
        if row[selection_metric] < best_valid_loss:
            best_valid_loss = float(row[selection_metric])
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 8) == 0:
            print(
                f"epoch {epoch:03d} | teacher_forcing={ratio:.3f} | "
                f"train_loss={row['train_loss']:.6f} | "
                f"open_loop_valid_loss={row['valid_loss']:.6f} | "
                f"valid_physics={row['valid_physics_loss']:.6f} | "
                f"epoch_time={row['epoch_time_sec']:.1f}s"
            )

    if best_state is None:
        raise RuntimeError("Multi-step training produced no valid checkpoint")
    model.load_state_dict(best_state)

    model_path = output_dir / "physics_graph_world_model_multistep.pt"
    train_episode_ids = {
        int(np.asarray(sample["episode_id"]).item()) for sample in train_transitions
    }
    valid_episode_ids = {
        int(np.asarray(sample["episode_id"]).item()) for sample in valid_transitions
    }
    args_dict = vars(args) | {
        "model_version": model_version,
        "parameter_profile": parameter_profile,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "validation_split": "episode_grouped_before_windowing",
        "validation_rollout": "fully_open_loop",
        "selected_device": device,
        "amp_enabled": amp_enabled,
        "selected_epoch": best_epoch,
        "selected_valid_loss": best_valid_loss,
        "transition_count": len(transitions),
        "trajectory_collection_time_sec": collection_time_sec,
        "trajectory_source": transition_source,
        "resolved_data_seed": data_seed,
        "resolved_split_seed": split_seed,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "selection_metric": {
            "v13": "valid_future_risk_loss",
            "v14": "valid_future_terminal_loss",
        }.get(args.model_variant, "valid_loss"),
        "train_sequence_count": len(train_sequences),
        "valid_sequence_count": len(valid_sequences),
        "train_episode_count": len(train_episode_ids),
        "valid_episode_count": len(valid_episode_ids),
    }
    save_function(model_path, model, metadata, history, args_dict)
    write_history(output_dir / "training_history.csv", history)
    (output_dir / "training_args.json").write_text(
        json.dumps(args_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_summary.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in args_dict.items()),
        encoding="utf-8",
    )
    selection_metric = {
        "v13": "valid_future_risk_loss",
        "v14": "valid_future_terminal_loss",
    }.get(args.model_variant, "valid_loss")
    print(
        f"Selected epoch {best_epoch} by {selection_metric}={best_valid_loss:.6f}."
    )
    print(f"Multi-step world model saved to {model_path.resolve()}")
    return model_path


if __name__ == "__main__":
    train(build_parser().parse_args())
