from __future__ import annotations

import argparse
import gzip
import json
import pickle
from dataclasses import asdict
from pathlib import Path

from counterfactual_rollout_v143 import (
    ExpectedEffectCollectionConfig,
    collect_expected_effect_samples,
    summarize_expected_effect_samples,
)


SCHEMA = "v143_queue_complete_expected_effect_pairs_v1"


def collect(args: argparse.Namespace) -> Path:
    config = ExpectedEffectCollectionConfig(
        episodes=args.episodes,
        behavior_steps=args.behavior_steps,
        warmup_steps=args.warmup_steps,
        sample_stride=args.sample_stride,
        candidates_per_state=args.candidates_per_state,
        future_replications=args.future_replications,
        minimum_valid_replications=args.minimum_valid_replications,
        behavior_seed=args.behavior_seed,
        future_seed=args.future_seed,
    )
    samples = collect_expected_effect_samples(config)
    summary = summarize_expected_effect_samples(samples)
    payload = {
        "signature": {
            "schema": SCHEMA,
            "config": asdict(config),
            "queue_complete_auxiliary_state": True,
            "common_random_numbers_within_each_future_replication": True,
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wb", compresslevel=4) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"V14.3 expected-effect cache saved to {args.output}")
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--behavior-steps", type=int, default=2400)
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--candidates-per-state", type=int, default=3)
    parser.add_argument("--future-replications", type=int, default=2)
    parser.add_argument("--minimum-valid-replications", type=int, default=2)
    parser.add_argument("--behavior-seed", type=int, default=14300)
    parser.add_argument("--future-seed", type=int, default=14350)
    return parser


if __name__ == "__main__":
    collect(build_parser().parse_args())
