#!/usr/bin/env python3
"""Local SPSA-style runner for BulletOu fine-tuning experiments.

This script does not change BulletOu's training algorithm.  It repeatedly
branches from a checkpoint, runs two short trials with slightly different
factorizer hyperparameters along opposite random directions, estimates an SPSA
direction from the two probe losses, and updates the hyperparameters by a small
fraction of that perturbation width.
The model weights are continued from the better short trial checkpoint.

Typical use:

    python spsa_local_runner.py ^
      --exe .\target\release\examples\bulletou.exe ^
      --base-checkpoint C:\...\0033 ^
      --teacher D:\sojoteam_datasets ^
      --test-teacher C:\shogi\teacher\test\test.hcpe ^
      --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 ^
      --bucket-counts D:\sojo_counts\...\count-all.bin ^
      --output-folder D:\BulletOu-snapshots\20260820 ^
      --tag-prefix spsa-hp4 ^
      --iterations 20 ^
      --sb-per-trial 8 ^
      --theta shared=1,axis=1,pair=0.3,residual_count=1,axis_count=1,pair_count=10,king_axis_count=4 ^
      --positions-per-superbatch 40000000 ^
      -- --wrm-in-offset 0 --wrm-target-offset 0 --lr 0.000300 --lr-min 0.000300

Arguments after `--` are passed through to bulletou.exe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_THETA: dict[str, float] = {
    "shared_alpha": 1.0,
    "king_axis_alpha": 1.0,
    "hand_axis_alpha": 1.0,
    "progress_axis_alpha": 1.0,
    "pair_alpha": 1.0,
    "residual_count_confidence": 0.0,
    "king_axis_count_confidence": 0.0,
    "hand_axis_count_confidence": 0.0,
    "progress_axis_count_confidence": 0.0,
    "king_hand_pair_count_confidence": 0.0,
    "king_progress_pair_count_confidence": 0.0,
    "hand_progress_pair_count_confidence": 0.0,
}


AXIS_ALPHA_KEYS = [
    "king_axis_alpha",
    "hand_axis_alpha",
    "progress_axis_alpha",
]


AXIS_COUNT_KEYS = [
    "king_axis_count_confidence",
    "hand_axis_count_confidence",
    "progress_axis_count_confidence",
]


PAIR_COUNT_KEYS = [
    "king_hand_pair_count_confidence",
    "king_progress_pair_count_confidence",
    "hand_progress_pair_count_confidence",
]


ANSI_COLORS = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}


DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "shared_alpha": (0.05, 10.0),
    "king_axis_alpha": (0.05, 10.0),
    "hand_axis_alpha": (0.0, 10.0),
    "progress_axis_alpha": (0.0, 10.0),
    "pair_alpha": (0.0, 10.0),
    "residual_count_confidence": (0.0, 20.0),
    "king_axis_count_confidence": (0.0, 100.0),
    "hand_axis_count_confidence": (0.0, 100.0),
    "progress_axis_count_confidence": (0.0, 100.0),
    "king_hand_pair_count_confidence": (0.0, 200.0),
    "king_progress_pair_count_confidence": (0.0, 200.0),
    "hand_progress_pair_count_confidence": (0.0, 200.0),
}


ALPHA_KEYS = [
    "shared_alpha",
    "king_axis_alpha",
    "hand_axis_alpha",
    "progress_axis_alpha",
    "pair_alpha",
]


CONFIDENCE_FLAG_BY_KEY = {
    "residual_count_confidence": "--sfnn-residual-count-confidence",
    "king_axis_count_confidence": "--sfnn-king-axis-count-confidence",
    "hand_axis_count_confidence": "--sfnn-hand-axis-count-confidence",
    "progress_axis_count_confidence": "--sfnn-progress-axis-count-confidence",
    "king_hand_pair_count_confidence": "--sfnn-king-hand-pair-count-confidence",
    "king_progress_pair_count_confidence": "--sfnn-king-progress-pair-count-confidence",
    "hand_progress_pair_count_confidence": "--sfnn-hand-progress-pair-count-confidence",
}


QUANTIZED_TEST_VALUE_FLAGS = {
    "--test-positions",
    "--test-sample",
    "--test-seed",
    "--score-drop-abs",
    "--fv-scale",
    "--sfnn-ft-shift",
    "--lambda",
    "--scale",
    "--loss-pow-exp",
    "--wrm-nnue2score",
    "--wrm-in-offset",
    "--wrm-in-scaling",
    "--wrm-target-offset",
    "--wrm-target-scaling",
    "--loader-threads",
    "--batch-queue-size",
    "--teacher-shuffle-buffer-batches",
    "--teacher-shuffle-seed",
}


QUANTIZED_TEST_BOOL_FLAGS = {
    "--win-rate-model",
    "--loss-sigmoid-mse",
}


PROBE_A = "probe_a"
PROBE_B = "probe_b"
LEGACY_PROBE_A = {"probe_a", "plus"}
LEGACY_PROBE_B = {"probe_b", "minus"}
TRIAL_NO_INTERVAL_SAVE_RATE = 9999


ACCEPTED_SUMMARY_FIELDS = [
    "iteration",
    "accepted_sbs",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "test_value_accuracy",
    "test_value_loss",
    "saved_checkpoint",
    "update_mode",
    "step_scale_used",
    "step_scale_next",
    "spsa_move_ratio",
    "theta_change",
    "theta_before_json",
    "theta_delta_json",
    "theta_json",
    "reason",
]


HISTORY_FIELDS = [
    "iteration",
    "retry",
    "reason",
    "base_score_before",
    "probe_a_score",
    "probe_b_score",
    "probe_score_diff",
    "best_probe_score",
    "new_base_score",
    "new_base_checkpoint",
    "accepted_sbs",
    "public_checkpoint",
    "retry_best_score",
    "retry_best_checkpoint",
    "update_mode",
    "step_scale_used",
    "step_scale_next",
    "spsa_move_ratio",
    "theta_change",
    "theta_before_json",
    "theta_delta_json",
    "theta_json",
    "theta_candidate_json",
    "spsa_gradient_log_json",
    "spsa_log_update_json",
    "delta_json",
]


TRIAL_SUMMARY_FIELDS = [
    "iteration",
    "retry",
    "result",
    "reason",
    "accepted_sbs",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "test_value_accuracy",
    "test_value_loss",
    "trial_tag",
    "trial_output_dir",
    "checkpoint",
    "saved_checkpoint",
    "update_mode",
    "step_scale_used",
    "step_scale_next",
    "spsa_move_ratio",
    "theta_change",
    "theta_json",
]


@dataclass(frozen=True)
class Metric:
    qloss: float | None
    qacc: float | None
    test_loss: float | None
    test_acc: float | None

    def score(self, metric_name: str) -> float:
        if metric_name == "quantized_value_loss":
            if self.qloss is not None:
                return self.qloss
            if self.test_loss is not None:
                return self.test_loss
        elif metric_name == "test_value_loss":
            if self.test_loss is not None:
                return self.test_loss
            if self.qloss is not None:
                return self.qloss
        raise ValueError(f"metric {metric_name!r} is unavailable in summary row")


@dataclass(frozen=True)
class TrialResult:
    side: str
    tag: str
    output_dir: Path
    checkpoint_dir: Path
    metric: Metric
    score: float
    summary_row: dict[str, str]


@dataclass(frozen=True)
class RetryBest:
    result: TrialResult
    theta: dict[str, float]


@dataclass(frozen=True)
class SpsaUpdate:
    theta: dict[str, float]
    gradient_log: dict[str, float]
    log_update: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired randomized BulletOu probes and tune factorizer hyperparameters."
    )
    parser.add_argument("--exe", required=True, help="Path to bulletou.exe")
    parser.add_argument("--base-checkpoint", type=Path, default=None, help="Checkpoint directory containing state.bin")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--test-teacher", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--bucket-counts", type=Path, default=None)
    parser.add_argument("--output-folder", required=True, type=Path, help="Root folder for runner outputs")
    parser.add_argument("--runner-dir", type=Path, default=None, help="Exact runner directory. Default: output-folder/spsa-<tag-prefix>")
    parser.add_argument(
        "--resume",
        "--resume-runner",
        dest="resume_runner",
        action="store_true",
        help="Resume an existing runner resolved from --output-folder and --tag-prefix, or from --runner-dir if specified.",
    )
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--factorizer", default="pair")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sb-per-trial", type=int, default=8)
    parser.add_argument(
        "--epoch-sbs",
        type=int,
        default=32,
        help="Number of accepted superbatches treated as one runner epoch. Used for accepted checkpoint names.",
    )
    parser.add_argument(
        "--save-rate",
        type=int,
        default=None,
        help="Save the accepted path every N accepted trials. 1 saves after every accept; 0 disables public checkpoints.",
    )
    parser.add_argument(
        "--accepted-save-rate-sbs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-trials",
        action="store_true",
        help="Keep probe trial output directories. Default deletes trial outputs after the decision.",
    )
    parser.add_argument(
        "--trial-validation-rate-sbs",
        type=int,
        default=1,
        help="Run normal validation every N sb inside each trial. Default: 1, so stdout shows loss every sb.",
    )
    parser.add_argument(
        "--trial-quantized-validation-rate-sbs",
        type=int,
        default=1,
        help="Run quantized validation every N sb inside each trial. Default: 1, so stdout shows qloss every sb.",
    )
    parser.add_argument("--positions-per-superbatch", type=int, default=40_000_000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-score", type=float, default=None, help="Base score override. Lower is better.")
    parser.add_argument("--metric", choices=["quantized_value_loss", "test_value_loss"], default="quantized_value_loss")
    parser.add_argument(
        "--base-metric-source",
        choices=["quantized-test", "summary"],
        default="quantized-test",
        help="How to obtain the initial base metric. Default runs bulletou quantized-test on base nn.bin.",
    )
    parser.add_argument(
        "--base-quantized-test-mode",
        choices=["gpu", "cpu-exact"],
        default="gpu",
        help="quantized-test mode used for the initial base metric.",
    )
    parser.add_argument(
        "--theta",
        action="append",
        default=[],
        help=(
            "Initial parameter override list, e.g. "
            "shared=1,axis=1,pair=0.3,residual_count=1,axis_count=1,pair_count=10,king_axis_count=4. "
            "Repeatable."
        ),
    )
    parser.add_argument("--theta-json", type=Path, default=None, help="Initial theta JSON file")
    parser.add_argument("--bounds-json", type=Path, default=None, help="Bounds JSON file: {name:[min,max], ...}")
    parser.add_argument(
        "--tune",
        action="append",
        default=None,
        help=(
            "Parameter or group to tune. Repeatable. Groups: alpha, axis, pair, count, "
            "axis_count, pair_count. `pair` means pair_alpha only; `pair_count` means "
            "king_hand/king_progress/hand_progress pair count confidences. Default: "
            "non-zero parameters except shared_alpha."
        ),
    )
    parser.add_argument("--fixed", action="append", default=[], help="Parameter to keep fixed. Repeatable.")
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.03,
        help="Initial multiplicative perturbation scale. 1.03 moves 0.300 to about 0.309/0.291.",
    )
    parser.add_argument("--step-grow", type=float, default=1.01, help="Grow factor after an improving acceptance")
    parser.add_argument("--min-step-scale", type=float, default=1.005)
    parser.add_argument("--max-step-scale", type=float, default=1.10)
    parser.add_argument(
        "--update-mode",
        choices=["spsa", "winner"],
        default="spsa",
        help=(
            "How to update theta after paired probes. `spsa` uses the two-probe "
            "loss difference as a gradient estimate. `winner` jumps directly to "
            "the better observed probe theta."
        ),
    )
    parser.add_argument(
        "--spsa-move-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of the paired perturbation used for the theta update. "
            "Default 0.1 means move theta by 10%% of the current delta width."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Force-accept the best retry trial after this many non-improving probe pairs. Default: 2.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-stream-child-output",
        action="store_true",
        help="Do not mirror bulletou.exe stdout to the console. Logs are still saved.",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color runner event lines. Default: auto.",
    )
    parser.add_argument(
        "--use-worker",
        action="store_true",
        help=(
            "Run one GPU-resident bulletou.exe worker session. Probe trials use GPU snapshot/restore "
            "and do not write trial checkpoints; accepted states are saved only at runner --save-rate boundaries."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to bulletou.exe")
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if not args.resume_runner and args.base_checkpoint is None:
        parser.error("--base-checkpoint is required unless --resume is specified")
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    if args.sb_per_trial <= 0:
        parser.error("--sb-per-trial must be > 0")
    if args.epoch_sbs <= 0:
        parser.error("--epoch-sbs must be > 0")
    if args.epoch_sbs % args.sb_per_trial != 0:
        parser.error("--epoch-sbs must be divisible by --sb-per-trial")
    if args.save_rate is not None and args.accepted_save_rate_sbs is not None:
        parser.error("use runner --save-rate, not both --save-rate and --accepted-save-rate-sbs")
    if args.accepted_save_rate_sbs is not None:
        if args.accepted_save_rate_sbs < 0:
            parser.error("--accepted-save-rate-sbs must be >= 0")
        if args.accepted_save_rate_sbs and args.accepted_save_rate_sbs % args.sb_per_trial != 0:
            parser.error("--accepted-save-rate-sbs must be divisible by --sb-per-trial")
        args.save_rate = 0 if args.accepted_save_rate_sbs == 0 else args.accepted_save_rate_sbs // args.sb_per_trial
        print(
            "WARN: --accepted-save-rate-sbs is deprecated; use "
            f"--save-rate {args.save_rate} instead.",
            flush=True,
        )
    if args.save_rate is None:
        args.save_rate = 4
    if args.save_rate < 0:
        parser.error("--save-rate must be >= 0")
    if args.trial_validation_rate_sbs <= 0:
        parser.error("--trial-validation-rate-sbs must be > 0")
    if args.trial_quantized_validation_rate_sbs <= 0:
        parser.error("--trial-quantized-validation-rate-sbs must be > 0")
    if args.positions_per_superbatch <= 0:
        parser.error("--positions-per-superbatch must be > 0")
    for name in ["step_scale", "step_grow", "min_step_scale", "max_step_scale"]:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.step_scale <= 1.0:
        parser.error("--step-scale should be > 1")
    if args.min_step_scale <= 1.0:
        parser.error("--min-step-scale should be > 1")
    if args.max_step_scale < args.min_step_scale:
        parser.error("--max-step-scale must be >= --min-step-scale")
    if not math.isfinite(args.spsa_move_ratio) or args.spsa_move_ratio < 0.0:
        parser.error("--spsa-move-ratio must be finite and >= 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def metric_to_json(metric: Metric) -> dict[str, float | None]:
    return {
        "qloss": metric.qloss,
        "qacc": metric.qacc,
        "test_loss": metric.test_loss,
        "test_acc": metric.test_acc,
    }


def metric_from_json(raw: Any) -> Metric:
    if not isinstance(raw, dict):
        raise ValueError("metric state must be an object")
    return Metric(
        qloss=parse_float(str(raw["qloss"])) if raw.get("qloss") is not None else None,
        qacc=parse_float(str(raw["qacc"])) if raw.get("qacc") is not None else None,
        test_loss=parse_float(str(raw["test_loss"])) if raw.get("test_loss") is not None else None,
        test_acc=parse_float(str(raw["test_acc"])) if raw.get("test_acc") is not None else None,
    )


def split_assignments(text: str) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"expected key=value in {text!r}, got {part!r}")
        key, value = part.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"invalid numeric value in {part!r}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"value for {key!r} must be finite")
        out.append((key, parsed))
    return out


def apply_theta_alias(theta: dict[str, float], key: str, value: float) -> None:
    direct = {
        "shared_alpha": ["shared_alpha", "shared"],
        "king_axis_alpha": ["king_axis_alpha", "king_axis", "king", "k"],
        "hand_axis_alpha": ["hand_axis_alpha", "hand_axis", "hand", "h"],
        "progress_axis_alpha": ["progress_axis_alpha", "progress_axis", "progress", "prog", "p"],
        "pair_alpha": ["pair_alpha", "pair"],
        "residual_count_confidence": [
            "residual_count_confidence",
            "residual_count",
            "residual_confidence",
            "residual",
        ],
        "king_axis_count_confidence": [
            "king_axis_count_confidence",
            "king_axis_count",
            "king_count",
            "k_count",
        ],
        "hand_axis_count_confidence": [
            "hand_axis_count_confidence",
            "hand_axis_count",
            "hand_count",
            "h_count",
        ],
        "progress_axis_count_confidence": [
            "progress_axis_count_confidence",
            "progress_axis_count",
            "progress_count",
            "prog_count",
            "p_count",
        ],
        "king_hand_pair_count_confidence": [
            "king_hand_pair_count_confidence",
            "king_hand_pair_count",
            "king_hand_count",
            "kh_pair_count",
            "kh_count",
        ],
        "king_progress_pair_count_confidence": [
            "king_progress_pair_count_confidence",
            "king_progress_pair_count",
            "king_progress_count",
            "kp_pair_count",
            "kp_count",
        ],
        "hand_progress_pair_count_confidence": [
            "hand_progress_pair_count_confidence",
            "hand_progress_pair_count",
            "hand_progress_count",
            "hp_pair_count",
            "hp_count",
        ],
    }
    alias_to_key = {alias: canonical for canonical, aliases in direct.items() for alias in aliases}
    if key in alias_to_key:
        theta[alias_to_key[key]] = value
        return
    if key in ("all_alpha", "all"):
        for name in ALPHA_KEYS:
            theta[name] = value
        return
    if key == "axis":
        for name in AXIS_ALPHA_KEYS:
            theta[name] = value
        return
    if key in ("axis_count", "axis_confidence"):
        for name in AXIS_COUNT_KEYS:
            theta[name] = value
        return
    if key in ("pair_count", "pair_confidence"):
        for name in PAIR_COUNT_KEYS:
            theta[name] = value
        return
    if key in ("count", "count_confidence"):
        theta["residual_count_confidence"] = value
        for name in AXIS_COUNT_KEYS:
            theta[name] = value
        for name in PAIR_COUNT_KEYS:
            theta[name] = value
        return
    raise ValueError(f"unknown theta key or alias {key!r}")


def load_theta(path: Path | None, overrides: list[str]) -> dict[str, float]:
    theta = dict(DEFAULT_THETA)
    if path is not None:
        raw = load_json_object(path)
        for key, value in raw.items():
            if key not in theta:
                raise ValueError(f"unknown theta key {key!r} in {path}")
            theta[key] = float(value)
    for override in overrides:
        for key, value in split_assignments(override):
            apply_theta_alias(theta, key, value)
    return theta


def load_bounds(path: Path | None) -> dict[str, tuple[float, float]]:
    bounds = dict(DEFAULT_BOUNDS)
    if path is not None:
        raw = load_json_object(path)
        for key, value in raw.items():
            if key not in bounds:
                raise ValueError(f"unknown bounds key {key!r} in {path}")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"bounds for {key!r} must be [min, max]")
            lo, hi = float(value[0]), float(value[1])
            if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo <= hi):
                raise ValueError(f"invalid bounds for {key!r}: {value!r}")
            bounds[key] = (lo, hi)
    return bounds


def clamp_theta(theta: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = {}
    for key, value in theta.items():
        if key not in bounds:
            raise ValueError(f"no bounds for theta key {key!r}")
        lo, hi = bounds[key]
        if not math.isfinite(value):
            raise ValueError(f"theta {key!r} must be finite")
        out[key] = min(max(value, lo), hi)
    return out


def expand_tune_name(name: str, theta: dict[str, float]) -> list[str]:
    key = name.strip().lower().replace("-", "_")
    if key in theta:
        return [key]
    groups = {
        "alpha": ALPHA_KEYS,
        "all_alpha": ALPHA_KEYS,
        "axis": AXIS_ALPHA_KEYS,
        "pair": ["pair_alpha"],
        "count": ["residual_count_confidence", *AXIS_COUNT_KEYS, *PAIR_COUNT_KEYS],
        "axis_count": AXIS_COUNT_KEYS,
        "pair_count": PAIR_COUNT_KEYS,
        "shared": ["shared_alpha"],
        "king_axis": ["king_axis_alpha"],
        "hand_axis": ["hand_axis_alpha"],
        "progress_axis": ["progress_axis_alpha"],
        "king_axis_count": ["king_axis_count_confidence"],
        "hand_axis_count": ["hand_axis_count_confidence"],
        "progress_axis_count": ["progress_axis_count_confidence"],
        "king_hand_pair_count": ["king_hand_pair_count_confidence"],
        "king_progress_pair_count": ["king_progress_pair_count_confidence"],
        "hand_progress_pair_count": ["hand_progress_pair_count_confidence"],
    }
    if key not in groups:
        raise ValueError(f"unknown tune key or group {name!r}")
    return list(groups[key])


def expand_key_list(names: list[str], theta: dict[str, float]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        for key in expand_tune_name(name, theta):
            if key not in seen:
                out.append(key)
                seen.add(key)
    return out


def tuned_keys(args: argparse.Namespace, theta: dict[str, float]) -> list[str]:
    if args.tune:
        keys = expand_key_list(args.tune, theta)
    else:
        keys = [key for key in theta if key != "shared_alpha" and theta[key] != 0.0]
    fixed = set(expand_key_list(args.fixed, theta))
    return [key for key in keys if key not in fixed]


def theta_log_offset(lo: float, hi: float) -> float:
    return max(1.0e-6, min(0.01, (hi - lo) * 1.0e-4))


def theta_to_log(value: float, lo: float, hi: float) -> float:
    return math.log(max(value, lo) + theta_log_offset(lo, hi))


def theta_from_log(value: float, lo: float, hi: float) -> float:
    return min(max(math.exp(value) - theta_log_offset(lo, hi), lo), hi)


def perturb_value(value: float, sign: int, step_scale: float, lo: float, hi: float) -> float:
    if lo == hi:
        return lo
    y = theta_to_log(value, lo, hi)
    y += sign * math.log(step_scale)
    return theta_from_log(y, lo, hi)


def make_variant(
    theta: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    delta: dict[str, int],
    side_sign: int,
    step_scale: float,
) -> dict[str, float]:
    out = dict(theta)
    for key, direction in delta.items():
        lo, hi = bounds[key]
        out[key] = perturb_value(theta[key], side_sign * direction, step_scale, lo, hi)
    return out


def probe_direction(seed: int, iteration: int, retry: int, keys: list[str]) -> dict[str, int]:
    mixed = (
        (seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((iteration * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        ^ ((retry * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF)
    )
    rng = random.Random(mixed)
    return {key: rng.choice([-1, 1]) for key in keys}


def make_spsa_update(
    theta: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    keys: list[str],
    delta: dict[str, int],
    probe_a_score: float,
    probe_b_score: float,
    step_scale: float,
    move_ratio: float,
) -> SpsaUpdate:
    if not (math.isfinite(probe_a_score) and math.isfinite(probe_b_score)):
        raise ValueError("SPSA update requires finite paired probe scores")
    c = math.log(step_scale)
    if c <= 0.0:
        raise ValueError("--step-scale must be > 1 for SPSA update")
    if probe_a_score < probe_b_score:
        winner_sign = +1
    elif probe_b_score < probe_a_score:
        winner_sign = -1
    else:
        winner_sign = 0
    next_theta = dict(theta)
    gradient_log: dict[str, float] = {}
    log_update: dict[str, float] = {}
    for key in keys:
        lo, hi = bounds[key]
        direction = delta[key]
        grad = (probe_a_score - probe_b_score) / (2.0 * c * direction)
        update = winner_sign * direction * move_ratio * c
        z = theta_to_log(theta[key], lo, hi)
        next_theta[key] = theta_from_log(z + update, lo, hi)
        gradient_log[key] = grad
        log_update[key] = update
    return SpsaUpdate(next_theta, gradient_log, log_update)


def alpha_arg(theta: dict[str, float]) -> str:
    return ",".join(
        [
            f"shared={theta['shared_alpha']:.9g}",
            f"king_axis={theta['king_axis_alpha']:.9g}",
            f"hand_axis={theta['hand_axis_alpha']:.9g}",
            f"progress_axis={theta['progress_axis_alpha']:.9g}",
            f"pair={theta['pair_alpha']:.9g}",
        ]
    )


def theta_args(theta: dict[str, float], bucket_counts: Path | None) -> list[str]:
    out = ["--sfnn-factorizer-alpha", alpha_arg(theta)]
    any_count = False
    for key, flag in CONFIDENCE_FLAG_BY_KEY.items():
        value = theta[key]
        out.extend([flag, f"{value:.9g}"])
        any_count = any_count or value != 0.0
    if any_count:
        if bucket_counts is None:
            raise ValueError("count-confidence is non-zero, but --bucket-counts was not specified")
        out.extend(["--sfnn-bucket-counts", str(bucket_counts)])
    return out


def trial_no_interval_save_rate(sb_per_trial: int) -> int:
    return max(TRIAL_NO_INTERVAL_SAVE_RATE, sb_per_trial + 1)


def base_command(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    tag: str,
    theta: dict[str, float],
    trial_output_folder: Path,
) -> list[str]:
    cmd = [
        str(args.exe),
        "--backend",
        "cuda-cpp",
        "--teacher",
        str(args.teacher),
        "--test-teacher",
        str(args.test_teacher),
        "--arch",
        args.arch,
        "--initial-state",
        str(checkpoint_dir / "state.bin"),
        "--initial-dataloader-pos",
        str(checkpoint_dir / "dataloader_pos.txt"),
        "--output-folder",
        str(trial_output_folder),
        "--tag",
        tag,
        "--sfnn-factorizer",
        args.factorizer,
        "--positions-per-superbatch",
        str(args.positions_per_superbatch),
        "--superbatches",
        str(args.sb_per_trial),
        "--max-epochs",
        "1",
        "--save-rate",
        str(trial_no_interval_save_rate(args.sb_per_trial)),
        "--validation-rate",
        str(args.trial_validation_rate_sbs),
        "--quantized-validation-rate",
        str(args.trial_quantized_validation_rate_sbs),
    ]
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    cmd.extend(theta_args(theta, args.bucket_counts))
    cmd.extend(args.extra_args)
    return cmd


def find_output_dir(trial_output_folder: Path, tag: str) -> Path:
    matches = [path for path in trial_output_folder.iterdir() if path.is_dir() and path.name.endswith(f"-{tag}")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one output dir ending with -{tag!r}, found {len(matches)}")
    return matches[0]


def parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text or text == "-":
        return None
    value = float(text)
    if math.isnan(value):
        return None
    return value


def metric_from_row(row: dict[str, str]) -> Metric:
    return Metric(
        qloss=parse_float(row.get("quantized_value_loss")),
        qacc=parse_float(row.get("quantized_value_accuracy")),
        test_loss=parse_float(row.get("test_value_loss")),
        test_acc=parse_float(row.get("test_value_accuracy")),
    )


def last_summary_row(summary_path: Path) -> dict[str, str]:
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist")
    last: dict[str, str] | None = None
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if last is None:
        raise ValueError(f"{summary_path} has no rows")
    return last


def checkpoint_metric(checkpoint_dir: Path, metric_name: str) -> Metric:
    summary_path = checkpoint_dir.parent / "summary-learn.log"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist")
    found: Metric | None = None
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("checkpoint") or "").strip() == checkpoint_dir.name:
                found = metric_from_row(row)
    if found is None:
        raise ValueError(f"no summary row for checkpoint {checkpoint_dir.name!r} in {summary_path}")
    found.score(metric_name)
    return found


def quantized_test_passthrough_args(extra_args: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token in QUANTIZED_TEST_VALUE_FLAGS:
            if i + 1 >= len(extra_args):
                raise ValueError(f"{token} in passthrough args requires a value")
            out.extend([token, extra_args[i + 1]])
            i += 2
        elif token in QUANTIZED_TEST_BOOL_FLAGS:
            out.append(token)
            i += 1
        else:
            i += 1
    return out


def run_process_tee(cmd: list[str], log_path: Path, stream: bool) -> tuple[int, float, list[str]]:
    lines: list[str] = []
    start = time.time()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            if stream:
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = proc.wait()
    elapsed = time.time() - start
    return returncode, elapsed, lines


class BulletOuWorker:
    def __init__(self, exe: str, log_path: Path, stream: bool) -> None:
        self.exe = exe
        self.stream = stream
        self.next_id = 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.main_log = log_path.open("a", encoding="utf-8", newline="")
        self.proc = subprocess.Popen(
            [exe, "worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open bulletou worker pipes")

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.request("quit", {}, Path(os_devnull_log_name()), stream=False)
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        self.proc.terminate()
                    except Exception:
                        pass
        finally:
            try:
                self.main_log.close()
            except Exception:
                pass

    def request(
        self,
        cmd: str,
        payload: dict[str, Any],
        log_path: Path,
        stream: bool,
    ) -> tuple[dict[str, Any], float, list[str]]:
        if self.proc.poll() is not None:
            raise RuntimeError(f"bulletou worker already exited with code {self.proc.returncode}")
        request_id = self.next_id
        self.next_id += 1
        request = {"id": request_id, "cmd": cmd}
        request.update(payload)
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        lines: list[str] = []
        started = time.time()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="") as log:
            request_line = json.dumps(request, ensure_ascii=False)
            self.main_log.write(f">>> {request_line}\n")
            self.main_log.flush()
            self.proc.stdin.write(request_line + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    returncode = self.proc.poll()
                    if returncode is None:
                        continue
                    raise RuntimeError(f"bulletou worker exited while waiting for `{cmd}` response: code={returncode}")
                lines.append(line)
                log.write(line)
                log.flush()
                self.main_log.write(line)
                self.main_log.flush()
                if stream:
                    sys.stdout.write(line)
                    sys.stdout.flush()

                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(response, dict) or response.get("id") != request_id or "ok" not in response:
                    continue
                elapsed = time.time() - started
                if not response.get("ok"):
                    error = response.get("error", "unknown worker error")
                    raise RuntimeError(f"worker `{cmd}` failed: {error}")
                payload_value = response.get("payload", {})
                if not isinstance(payload_value, dict):
                    raise RuntimeError(f"worker `{cmd}` returned non-object payload")
                return payload_value, elapsed, lines


def os_devnull_log_name() -> str:
    return "NUL" if sys.platform.startswith("win") else "/dev/null"


def use_color(args: argparse.Namespace) -> bool:
    if args.color == "always":
        return True
    if args.color == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def color_text(args: argparse.Namespace, text: str, color: str, *, bold: bool = False) -> str:
    if not use_color(args):
        return text
    prefix = ""
    if bold:
        prefix += ANSI_COLORS["bold"]
    prefix += ANSI_COLORS[color]
    return f"{prefix}{text}{ANSI_COLORS['reset']}"


def event_line(args: argparse.Namespace, label: str, message: str, color: str, *, bold: bool = False) -> None:
    print(f"{color_text(args, f'[{label}]', color, bold=bold)} {message}", flush=True)


def objective_label(metric_name: str) -> str:
    if metric_name == "quantized_value_loss":
        return "qloss"
    if metric_name == "test_value_loss":
        return "test_loss"
    return metric_name


def accept_threshold_text(args: argparse.Namespace, base_score: float, base_metric: Metric) -> str:
    label = objective_label(args.metric)
    text = color_text(args, f"accept_if {label} < {base_score:.9f}", "cyan", bold=True)
    if args.metric != "quantized_value_loss":
        text += f"  base_qloss={fmt_metric(base_metric.qloss)}"
    text += (
        f"  base_qacc={fmt_metric(base_metric.qacc)}"
        f"  base_test_loss={fmt_metric(base_metric.test_loss)}"
        f"  base_test_acc={fmt_metric(base_metric.test_acc)}"
    )
    return text


def trial_start_line(
    args: argparse.Namespace,
    *,
    trial_number: int,
    side: str,
    tag: str,
    base_score: float,
    base_metric: Metric,
) -> None:
    event_line(
        args,
        f"TRIAL {trial_number} START",
        (
            f"{probe_label(side)}  tag={tag}  "
            f"{accept_threshold_text(args, base_score, base_metric)}"
        ),
        "cyan",
        bold=True,
    )


def score_compare_text(args: argparse.Namespace, score: float, base_score: float, *, value_name: str) -> str:
    label = objective_label(args.metric)
    return (
        f"{value_name}_{label}={score:.9f} "
        f"start_{label}={base_score:.9f} "
        f"delta={score - base_score:+.9f}"
    )


def trial_threshold_line(
    args: argparse.Namespace,
    trial: TrialResult,
    base_score: float,
    *,
    trial_number: int,
) -> None:
    beats_target = trial.score < base_score
    status = "beats_target" if beats_target else "misses_target"
    color = "green" if beats_target else "yellow"
    event_line(
        args,
        f"TRIAL {trial_number} END",
        (
            f"{probe_label(trial.side)} {score_compare_text(args, trial.score, base_score, value_name='final')} "
            f"qacc={fmt_metric(trial.metric.qacc)} result={status}"
        ),
        color,
        bold=True,
    )


def probe_decision_line(args: argparse.Namespace, best_probe: TrialResult, base_score: float) -> None:
    beats_target = best_probe.score < base_score
    color = "green" if beats_target else "yellow"
    if beats_target:
        if args.update_mode == "spsa":
            decision = "continue from lower-loss NN weights; theta_update=spsa_step"
        else:
            decision = "continue from lower-loss NN weights; theta_update=winner_theta"
    else:
        decision = "retry because lower-loss trial did not beat start"
    event_line(
        args,
        "DECISION",
        (
            f"lower_loss_trial={probe_label(best_probe.side)} "
            f"{score_compare_text(args, best_probe.score, base_score, value_name='lower_loss_trial')} "
            f"decision={decision}"
        ),
        color,
        bold=True,
    )


def colored_metric_value(args: argparse.Namespace, value: float | None, color: str) -> str:
    return color_text(args, fmt_metric(value), color, bold=True)


def lower_is_better_color(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "cyan"
    return "green" if value < baseline else "yellow"


def higher_is_better_color(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None:
        return "cyan"
    return "green" if value > baseline else "yellow"


def done_metric_text(
    args: argparse.Namespace,
    metric: Metric,
    score: float,
    base_score: float | None,
    base_metric: Metric | None,
) -> str:
    parts: list[str] = []
    label = objective_label(args.metric)
    if label != "qloss" and label != "test_loss":
        parts.append(
            f"{label}={color_text(args, f'{score:.9f}', lower_is_better_color(score, base_score), bold=True)}"
        )
    if label == "test_loss":
        parts.append(
            "test_loss="
            + colored_metric_value(
                args,
                metric.test_loss,
                lower_is_better_color(metric.test_loss, base_metric.test_loss if base_metric is not None else None),
            )
        )
    qloss_baseline = base_metric.qloss if base_metric is not None else (base_score if label == "qloss" else None)
    parts.append(
        "qloss=" + colored_metric_value(args, metric.qloss, lower_is_better_color(metric.qloss, qloss_baseline))
    )
    parts.append(
        "qacc="
        + colored_metric_value(
            args,
            metric.qacc,
            higher_is_better_color(metric.qacc, base_metric.qacc if base_metric is not None else None),
        )
    )
    if label != "test_loss" and metric.test_loss is not None:
        parts.append(f"test_loss={fmt_metric(metric.test_loss)}")
    if metric.test_acc is not None:
        parts.append(f"test_acc={fmt_metric(metric.test_acc)}")
    return " ".join(parts)


def accepted_count(args: argparse.Namespace, accepted_sbs: int) -> int:
    return accepted_sbs // args.sb_per_trial


def should_save_accepted_checkpoint(args: argparse.Namespace, accepted_sbs: int) -> bool:
    if args.save_rate <= 0:
        return False
    count = accepted_count(args, accepted_sbs)
    return count > 0 and count % args.save_rate == 0


def next_save_accept_count(args: argparse.Namespace, accepted_sbs: int) -> int | None:
    if args.save_rate <= 0:
        return None
    count = accepted_count(args, accepted_sbs)
    remainder = count % args.save_rate
    if remainder == 0:
        return count
    return count + (args.save_rate - remainder)


def parse_quantized_test_metric(lines: list[str]) -> Metric:
    qacc: float | None = None
    qloss: float | None = None
    test_loss: float | None = None
    for line in lines:
        text = line.strip()
        if text.startswith("accuracy"):
            match = re.search(r"=\s*([0-9.+\-eE]+)%", text)
            if match:
                qacc = float(match.group(1)) / 100.0
        elif text.startswith("loss_engine_scale"):
            match = re.search(r"=\s*([0-9.+\-eE]+)", text)
            if match:
                qloss = float(match.group(1))
        elif text.startswith("loss_train_scale"):
            match = re.search(r"=\s*([0-9.+\-eE]+)", text)
            if match:
                test_loss = float(match.group(1))
    if qloss is None:
        raise ValueError("quantized-test output did not contain loss_engine_scale")
    return Metric(qloss=qloss, qacc=qacc, test_loss=test_loss, test_acc=None)


def metric_from_worker_quantized_payload(payload: dict[str, Any]) -> Metric:
    return Metric(
        qloss=parse_float(str(payload.get("quantized_value_loss")) if payload.get("quantized_value_loss") is not None else None),
        qacc=parse_float(str(payload.get("quantized_value_accuracy")) if payload.get("quantized_value_accuracy") is not None else None),
        test_loss=parse_float(str(payload.get("train_scale_loss")) if payload.get("train_scale_loss") is not None else None),
        test_acc=None,
    )


def metric_from_worker_trial_payload(payload: dict[str, Any]) -> Metric:
    return Metric(
        qloss=parse_float(str(payload.get("quantized_value_loss")) if payload.get("quantized_value_loss") is not None else None),
        qacc=parse_float(str(payload.get("quantized_value_accuracy")) if payload.get("quantized_value_accuracy") is not None else None),
        test_loss=parse_float(str(payload.get("test_value_loss")) if payload.get("test_value_loss") is not None else None),
        test_acc=parse_float(str(payload.get("test_value_accuracy")) if payload.get("test_value_accuracy") is not None else None),
    )


def worker_open_session(
    args: argparse.Namespace,
    worker: BulletOuWorker,
    checkpoint_dir: Path,
    theta: dict[str, float],
    runner_dir: Path,
    log_dir: Path,
) -> None:
    cmd = base_command(args, checkpoint_dir, "worker-session-open", theta, runner_dir / "session-open")
    print("[worker-open] " + subprocess.list2cmdline(cmd), flush=True)
    payload, elapsed, _lines = worker.request(
        "open",
        {"args": cmd[1:]},
        log_dir / "worker-open.stdout.log",
        stream=not args.no_stream_child_output,
    )
    print(
        f"[worker-open-done] arch={payload.get('arch')} batch_size={payload.get('batch_size')} "
        f"completed_steps={payload.get('completed_steps')} optimizer_steps={payload.get('optimizer_steps')} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def probe_label(side: str) -> str:
    if side in LEGACY_PROBE_A:
        return "probe A"
    if side in LEGACY_PROBE_B:
        return "probe B"
    return side.replace("_", " ")


def worker_accept_cached_trial(
    args: argparse.Namespace,
    worker: BulletOuWorker,
    result: TrialResult,
    log_dir: Path,
) -> TrialResult:
    log_path = log_dir / f"{result.tag}.accept.stdout.log"
    payload, elapsed, _lines = worker.request(
        "accept-cached-trial",
        {"cache_key": result.tag},
        log_path,
        stream=not args.no_stream_child_output,
    )
    metric = metric_from_worker_trial_payload(payload)
    score = metric.score(args.metric)
    row = {
        "test_value_accuracy": fmt_metric(metric.test_acc),
        "test_value_loss": fmt_metric(metric.test_loss),
        "quantized_value_accuracy": fmt_metric(metric.qacc),
        "quantized_value_loss": fmt_metric(metric.qloss),
        "checkpoint": "",
        "worker_trial_theta_json": result.summary_row.get("worker_trial_theta_json", ""),
    }
    event_line(
        args,
        "WEIGHTS",
        (
            f"restored selected trial NN weights ({probe_label(result.side)}) tag={result.tag}; "
            f"theta is updated separately by {args.update_mode}  "
            f"{done_metric_text(args, metric, score, None, None)} elapsed={elapsed:.1f}s"
        ),
        "green",
        bold=True,
    )
    return TrialResult(result.side, result.tag, result.output_dir, result.checkpoint_dir, metric, score, row)


def worker_drop_cached_trials(
    worker: BulletOuWorker | None,
    keys: list[str],
    log_dir: Path,
    *,
    stream: bool,
) -> None:
    if worker is None or not keys:
        return
    worker.request(
        "drop-cached-trials",
        {"cache_keys": keys},
        log_dir / "worker-drop-cached-trials.stdout.log",
        stream=stream,
    )


def worker_save_checkpoint(
    args: argparse.Namespace,
    worker: BulletOuWorker,
    checkpoint_hint: Path,
    theta: dict[str, float],
    save_dir: Path,
    accepted_sbs: int,
    log_dir: Path,
) -> Path:
    tag = f"worker-save-{accepted_checkpoint_name(accepted_sbs, args.epoch_sbs)}"
    cmd = base_command(args, checkpoint_hint, tag, theta, save_dir.parent)
    epoch = max(1, accepted_sbs // args.epoch_sbs)
    superbatch = args.epoch_sbs if accepted_sbs % args.epoch_sbs == 0 else accepted_sbs % args.epoch_sbs
    payload, elapsed, _lines = worker.request(
        "save",
        {"args": cmd[1:], "dir": str(save_dir), "epoch": epoch, "superbatch": superbatch},
        log_dir / f"{tag}.stdout.log",
        stream=not args.no_stream_child_output,
    )
    saved = Path(str(payload.get("checkpoint_dir", save_dir))).resolve()
    event_line(args, "SAVE", f"accepted_sbs={accepted_sbs} checkpoint={saved} elapsed={elapsed:.1f}s", "cyan", bold=True)
    return saved


def run_base_quantized_test(
    args: argparse.Namespace,
    base_checkpoint: Path,
    log_dir: Path,
    worker: BulletOuWorker | None = None,
) -> Metric:
    nn_bin = base_checkpoint / "nn.bin"
    if not nn_bin.exists():
        raise FileNotFoundError(f"{nn_bin} does not exist; use --base-metric-source summary or provide a checkpoint with nn.bin")
    qt_args = [
        "--arch",
        args.arch,
        "--nn-bin",
        str(nn_bin),
        "--test-teacher",
        str(args.test_teacher),
        "--mode",
        args.base_quantized_test_mode,
    ]
    qt_args.extend(quantized_test_passthrough_args(args.extra_args))
    cmd = [str(args.exe), "quantized-test", *qt_args]
    log_path = log_dir / "base-quantized-test.stdout.log"
    if worker is not None:
        print("[base-run worker] quantized-test " + subprocess.list2cmdline(qt_args), flush=True)
        payload, elapsed, _lines = worker.request(
            "quantized-test",
            {"args": qt_args},
            log_path,
            stream=not args.no_stream_child_output,
        )
        metric = metric_from_worker_quantized_payload(payload)
    else:
        print("[base-run] " + subprocess.list2cmdline(cmd), flush=True)
        returncode, elapsed, lines = run_process_tee(cmd, log_path, stream=not args.no_stream_child_output)
        if returncode != 0:
            raise RuntimeError(f"base quantized-test failed with exit code {returncode}; see {log_path}")
        metric = parse_quantized_test_metric(lines)
    print(
        f"[base-done] source=quantized-test mode={args.base_quantized_test_mode} "
        f"qloss={fmt_metric(metric.qloss)} qacc={fmt_metric(metric.qacc)} "
        f"loss_train_scale={fmt_metric(metric.test_loss)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return metric


def run_trial(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    tag: str,
    side: str,
    trial_number: int,
    base_score: float,
    base_metric: Metric,
    theta: dict[str, float],
    trial_output_folder: Path,
    log_dir: Path,
    worker: BulletOuWorker | None = None,
) -> TrialResult:
    cmd = base_command(args, checkpoint_dir, tag, theta, trial_output_folder)
    log_path = log_dir / f"{tag}.stdout.log"
    trial_start_line(
        args,
        trial_number=trial_number,
        side=side,
        tag=tag,
        base_score=base_score,
        base_metric=base_metric,
    )
    print("      " + subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        return TrialResult(side, tag, log_dir, checkpoint_dir, Metric(None, None, None, None), float("inf"), {})
    prune_trial_output_by_tag(trial_output_folder, tag)
    if worker is not None:
        payload, elapsed, _lines = worker.request(
            "trial",
            {"args": cmd[1:], "cache_key": tag},
            log_path,
            stream=not args.no_stream_child_output,
        )
        metric = metric_from_worker_trial_payload(payload)
        score = metric.score(args.metric)
        row = {
            "test_value_accuracy": fmt_metric(metric.test_acc),
            "test_value_loss": fmt_metric(metric.test_loss),
            "quantized_value_accuracy": fmt_metric(metric.qacc),
            "quantized_value_loss": fmt_metric(metric.qloss),
            "checkpoint": "",
            "worker_trial_theta_json": json.dumps(theta, ensure_ascii=False, sort_keys=True),
        }
        print(
            f"[done] {probe_label(side)} tag={tag} "
            f"{done_metric_text(args, metric, score, base_score, base_metric)} elapsed={elapsed:.1f}s",
            flush=True,
        )
        return TrialResult(side, tag, trial_output_folder / tag, checkpoint_dir, metric, score, row)
    else:
        returncode, elapsed, _lines = run_process_tee(cmd, log_path, stream=not args.no_stream_child_output)
        if returncode != 0:
            raise RuntimeError(f"trial {tag} failed with exit code {returncode}; see {log_path}")
    output_dir = find_output_dir(trial_output_folder, tag)
    row = last_summary_row(output_dir / "summary-learn.log")
    metric = metric_from_row(row)
    score = metric.score(args.metric)
    checkpoint_name = (row.get("checkpoint") or "").strip()
    if not checkpoint_name or checkpoint_name == "-":
        raise ValueError(f"{output_dir / 'summary-learn.log'} final row has no checkpoint column")
    next_checkpoint = output_dir / checkpoint_name
    if not (next_checkpoint / "state.bin").exists():
        raise FileNotFoundError(f"{next_checkpoint / 'state.bin'} does not exist")
    if not (next_checkpoint / "dataloader_pos.txt").exists():
        raise FileNotFoundError(f"{next_checkpoint / 'dataloader_pos.txt'} does not exist")
    print(
        f"[done] {probe_label(side)} tag={tag} "
        f"{done_metric_text(args, metric, score, base_score, base_metric)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return TrialResult(side, tag, output_dir, next_checkpoint, metric, score, row)


def fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.9f}"


def ensure_inside(path: Path, root: Path) -> tuple[Path, Path]:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ValueError(f"refusing to modify {resolved}; it is not inside {resolved_root}")
    return resolved, resolved_root


def safe_rmtree(path: Path, root: Path) -> None:
    resolved, _ = ensure_inside(path, root)
    if resolved.exists():
        shutil.rmtree(resolved)


def move_checkpoint_to_current(src: Path, current_dir: Path, runner_dir: Path) -> Path:
    ensure_inside(current_dir, runner_dir)
    temp_dir = runner_dir / "current.new"
    safe_rmtree(temp_dir, runner_dir)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    shutil.move(str(src), str(temp_dir))
    safe_rmtree(current_dir, runner_dir)
    temp_dir.rename(current_dir)
    return current_dir


def accepted_checkpoint_name(accepted_sbs: int, epoch_sbs: int) -> str:
    if accepted_sbs % epoch_sbs == 0:
        return f"{accepted_sbs // epoch_sbs:04d}"
    return f"sb{accepted_sbs:08d}"


def copy_public_checkpoint(src: Path, accepted_root: Path, accepted_sbs: int, epoch_sbs: int) -> Path:
    accepted_root.mkdir(parents=True, exist_ok=True)
    dst = accepted_root / accepted_checkpoint_name(accepted_sbs, epoch_sbs)
    if dst.exists():
        raise FileExistsError(f"accepted checkpoint already exists: {dst}")
    shutil.copytree(src, dst)
    return dst


def prune_trial_outputs(results: list[TrialResult], trial_output_folder: Path) -> None:
    for result in results:
        safe_rmtree(result.output_dir, trial_output_folder)


def prune_trial_output_by_tag(trial_output_folder: Path, tag: str) -> None:
    if not trial_output_folder.exists():
        return
    for path in trial_output_folder.iterdir():
        if path.is_dir() and path.name.endswith(f"-{tag}"):
            safe_rmtree(path, trial_output_folder)


def theta_for_result(result: TrialResult, probe_a_theta: dict[str, float], probe_b_theta: dict[str, float]) -> dict[str, float]:
    if result.side in LEGACY_PROBE_A:
        return probe_a_theta
    if result.side in LEGACY_PROBE_B:
        return probe_b_theta
    raise ValueError(f"unknown trial side {result.side!r}")


def stash_retry_best(
    result: TrialResult,
    theta: dict[str, float],
    retry_best_dir: Path,
    runner_dir: Path,
    keep_trials: bool,
    dry_run: bool,
) -> RetryBest:
    if keep_trials or dry_run:
        return RetryBest(result, dict(theta))
    safe_rmtree(retry_best_dir, runner_dir)
    if not result.checkpoint_dir.exists():
        raise FileNotFoundError(f"{result.checkpoint_dir} does not exist")
    shutil.move(str(result.checkpoint_dir), str(retry_best_dir))
    return RetryBest(replace(result, checkpoint_dir=retry_best_dir), dict(theta))


def retry_best_to_json(retry_best: RetryBest | None) -> dict[str, Any] | None:
    if retry_best is None:
        return None
    result = retry_best.result
    return {
        "theta": retry_best.theta,
        "result": {
            "side": result.side,
            "tag": result.tag,
            "output_dir": str(result.output_dir),
            "checkpoint_dir": str(result.checkpoint_dir),
            "metric": metric_to_json(result.metric),
            "score": result.score,
            "summary_row": result.summary_row,
        },
    }


def retry_best_from_json(raw: Any) -> RetryBest | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("result"), dict):
        raise ValueError("retry_best state must be null or an object")
    result_raw = raw["result"]
    metric = metric_from_json(result_raw["metric"])
    result = TrialResult(
        side=str(result_raw["side"]),
        tag=str(result_raw["tag"]),
        output_dir=Path(str(result_raw["output_dir"])),
        checkpoint_dir=Path(str(result_raw["checkpoint_dir"])),
        metric=metric,
        score=float(result_raw["score"]),
        summary_row=dict(result_raw.get("summary_row", {})),
    )
    theta_raw = raw.get("theta")
    if not isinstance(theta_raw, dict):
        raise ValueError("retry_best theta must be an object")
    return RetryBest(result, {str(key): float(value) for key, value in theta_raw.items()})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def theta_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in before}


def theta_change_text(before: dict[str, float], after: dict[str, float], keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        old = before[key]
        new = after[key]
        diff = new - old
        if diff == 0.0:
            continue
        parts.append(f"{key}:{old:.9g}->{new:.9g}({diff:+.9g})")
    return ";".join(parts)


def schema_backup_path(path: Path) -> Path:
    base = path.with_name(path.name + ".old-schema")
    if not base.exists():
        return base
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}.old-schema{index}")
        if not candidate.exists():
            return candidate
        index += 1


def migrate_csv_schema(path: Path, backup: Path, fieldnames: list[str]) -> None:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    path.replace(backup)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[log-schema] migrated CSV schema: {path} (backup={backup})", flush=True)


def ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        if header == fieldnames:
            return
        backup = schema_backup_path(path)
        migrate_csv_schema(path, backup, fieldnames)
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    ensure_csv_header(path, fieldnames)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def read_csv_rows(path: Path, fieldnames: list[str]) -> list[dict[str, str]]:
    ensure_csv_header(path, fieldnames)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    append_csv_row(path, HISTORY_FIELDS, row)


def append_accepted_summary(path: Path, row: dict[str, Any]) -> None:
    append_csv_row(path, ACCEPTED_SUMMARY_FIELDS, row)


def append_trial_summary(path: Path, row: dict[str, Any]) -> None:
    append_csv_row(path, TRIAL_SUMMARY_FIELDS, row)


def trial_checkpoint_text(result: TrialResult) -> str:
    checkpoint = result.summary_row.get("checkpoint", "").strip()
    if checkpoint and checkpoint != "-":
        return str(result.output_dir / checkpoint)
    return str(result.checkpoint_dir)


def make_trial_summary_row(
    *,
    iteration: int,
    retry: int,
    result_label: str,
    reason: str,
    accepted_sbs: int,
    trial: TrialResult,
    trial_theta: dict[str, float],
    theta_before: dict[str, float],
    keys: list[str],
    saved_checkpoint: str,
    update_mode: str,
    step_scale_used: float,
    step_scale_next: float,
    spsa_move_ratio: float,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "retry": retry,
        "result": result_label,
        "reason": reason,
        "accepted_sbs": accepted_sbs,
        "quantized_value_loss": fmt_metric(trial.metric.qloss),
        "quantized_value_accuracy": fmt_metric(trial.metric.qacc),
        "test_value_loss": fmt_metric(trial.metric.test_loss),
        "test_value_accuracy": fmt_metric(trial.metric.test_acc),
        "trial_tag": trial.tag,
        "trial_output_dir": str(trial.output_dir),
        "checkpoint": trial_checkpoint_text(trial),
        "saved_checkpoint": saved_checkpoint,
        "update_mode": update_mode,
        "step_scale_used": f"{step_scale_used:.9f}",
        "step_scale_next": f"{step_scale_next:.9f}",
        "spsa_move_ratio": f"{spsa_move_ratio:.9f}",
        "theta_change": theta_change_text(theta_before, trial_theta, keys),
        "theta_json": json.dumps(trial_theta, ensure_ascii=False, sort_keys=True),
    }


def append_trial_summary_row(path: Path, **kwargs: Any) -> None:
    append_trial_summary(path, make_trial_summary_row(**kwargs))


def retry_window_rows(path: Path, iteration: int) -> list[dict[str, str]]:
    return [row for row in read_csv_rows(path, TRIAL_SUMMARY_FIELDS) if row.get("iteration") == str(iteration)]


def rewrite_trial_summary_iteration(path: Path, iteration: int, iteration_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in read_csv_rows(path, TRIAL_SUMMARY_FIELDS) if row.get("iteration") != str(iteration)]
    rows.extend(iteration_rows)
    write_csv_rows(path, TRIAL_SUMMARY_FIELDS, rows)


def default_runner_dir(output_folder: Path, tag_prefix: str) -> Path:
    return output_folder / f"spsa-{tag_prefix}"


def resolve_runner_dir(args: argparse.Namespace) -> Path:
    if args.runner_dir is not None:
        runner_dir = args.runner_dir
        if not args.resume_runner and (runner_dir / "state.json").exists():
            raise FileExistsError(f"{runner_dir} already has state.json; use --resume or choose a new --tag-prefix/--runner-dir")
        return runner_dir

    runner_dir = default_runner_dir(args.output_folder, args.tag_prefix)
    if args.resume_runner:
        if (runner_dir / "state.json").exists():
            return runner_dir
        legacy = [
            path
            for path in args.output_folder.glob(f"spsa-{args.tag_prefix}-*")
            if path.is_dir() and (path / "state.json").exists()
        ]
        if len(legacy) == 1:
            return legacy[0]
        if len(legacy) > 1:
            names = "\n  ".join(str(path) for path in legacy)
            raise ValueError(
                "multiple legacy runner directories match this tag-prefix; "
                "refusing to choose by timestamp. Specify --runner-dir explicitly:\n  " + names
            )
        raise FileNotFoundError(f"{runner_dir / 'state.json'} does not exist; start a new run without --resume")

    if (runner_dir / "state.json").exists():
        raise FileExistsError(f"{runner_dir} already has state.json; use --resume or choose a new --tag-prefix")
    return runner_dir


def write_runner_state(
    path: Path,
    *,
    phase: str,
    iteration: int,
    next_iteration: int,
    base_checkpoint: Path,
    base_score: float,
    base_metric: Metric,
    theta: dict[str, float],
    step_scale: float,
    failed_retries: int,
    accepted_sbs: int,
    retry_best: RetryBest | None,
    update_mode: str,
    complete: bool = False,
) -> None:
    write_json(
        path,
        {
            "phase": phase,
            "iteration": iteration,
            "next_iteration": next_iteration,
            "base_checkpoint": str(base_checkpoint),
            "base_score": base_score,
            "base_metric": metric_to_json(base_metric),
            "theta": theta,
            "step_scale": step_scale,
            "failed_retries": failed_retries,
            "accepted_sbs": accepted_sbs,
            "retry_best": retry_best_to_json(retry_best),
            "update_mode": update_mode,
            "complete": complete,
        },
    )


def main() -> int:
    args = parse_args()
    worker: BulletOuWorker | None = None
    try:
        bounds = load_bounds(args.bounds_json)

        runner_dir = resolve_runner_dir(args)
        trial_output_folder = runner_dir / "trials"
        log_dir = runner_dir / "logs"
        runner_dir.mkdir(parents=True, exist_ok=True)
        trial_output_folder.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        current_dir = runner_dir / "current"
        accepted_root = runner_dir / "accepted-checkpoints"
        retry_best_dir = runner_dir / "retry-best"

        state_path = runner_dir / "state.json"
        history_path = runner_dir / "history.csv"
        trial_summary_path = runner_dir / "summary-learn.log"
        accepted_summary_path = runner_dir / "accepted-summary-learn.log"
        ensure_csv_header(trial_summary_path, TRIAL_SUMMARY_FIELDS)
        ensure_csv_header(accepted_summary_path, ACCEPTED_SUMMARY_FIELDS)
        ensure_csv_header(history_path, HISTORY_FIELDS)

        if args.use_worker and not args.dry_run:
            worker = BulletOuWorker(str(args.exe), log_dir / "worker.stdout.log", stream=not args.no_stream_child_output)
            hello_payload, _hello_elapsed, _hello_lines = worker.request(
                "hello",
                {},
                log_dir / "worker-hello.stdout.log",
                stream=not args.no_stream_child_output,
            )
            print(
                f"[worker] started protocol={hello_payload.get('protocol_version')} "
                f"capabilities={hello_payload.get('capabilities')}",
                flush=True,
            )

        if args.resume_runner:
            if not state_path.exists():
                raise FileNotFoundError(f"{state_path} does not exist")
            state = load_json_object(state_path)
            accepted_checkpoint = Path(str(state["base_checkpoint"])).resolve()
            base_metric = metric_from_json(state["base_metric"])
            base_score = float(state["base_score"])
            theta_raw = state.get("theta")
            if not isinstance(theta_raw, dict):
                raise ValueError("state theta must be an object")
            if args.theta or args.theta_json is not None:
                print(
                    "WARN: --resume loads theta from runner state.json; ignoring --theta/--theta-json from the command line.",
                    flush=True,
                )
            theta = clamp_theta({str(key): float(value) for key, value in theta_raw.items()}, bounds)
            keys = tuned_keys(args, theta)
            step_scale = float(state["step_scale"])
            failed_retries = int(state.get("failed_retries", 0))
            accepted_sbs = int(state.get("accepted_sbs", 0))
            retry_best = retry_best_from_json(state.get("retry_best"))
            phase = str(state.get("phase", "complete" if state.get("complete") else "running_trials"))
            if phase == "between_retries":
                start_iteration = int(state.get("next_iteration", state["iteration"]))
            elif phase == "between_iterations" and failed_retries > 0:
                # Old runner versions wrote phase=between_iterations even when
                # the logical iteration was still retrying the same target.
                # Keep that retry under the same iteration number.
                start_iteration = int(state["iteration"])
            elif phase in ("between_iterations", "complete"):
                start_iteration = int(state.get("next_iteration", int(state["iteration"]) + 1))
            else:
                start_iteration = int(state.get("next_iteration", state["iteration"]))
            base_source = f"{state_path} (--resume)"
            print(
                f"[resume] runner_dir={runner_dir} phase={phase} start_iteration={start_iteration} "
                f"accepted_sbs={accepted_sbs} checkpoint={accepted_checkpoint}",
                flush=True,
            )
        else:
            theta = clamp_theta(load_theta(args.theta_json, args.theta), bounds)
            keys = tuned_keys(args, theta)
            assert args.base_checkpoint is not None
            accepted_checkpoint = args.base_checkpoint.resolve()
            if not (accepted_checkpoint / "state.bin").exists():
                raise FileNotFoundError(f"{accepted_checkpoint / 'state.bin'} does not exist")
            if not (accepted_checkpoint / "dataloader_pos.txt").exists():
                raise FileNotFoundError(f"{accepted_checkpoint / 'dataloader_pos.txt'} does not exist")

            if args.base_score is not None:
                base_metric = Metric(None, None, None, None)
                base_score = args.base_score
                base_source = "--base-score"
            elif args.base_metric_source == "quantized-test" and not args.dry_run:
                base_metric = run_base_quantized_test(args, accepted_checkpoint, log_dir, worker)
                base_score = base_metric.score(args.metric)
                base_source = str(log_dir / "base-quantized-test.stdout.log")
            else:
                base_metric = checkpoint_metric(accepted_checkpoint, args.metric)
                base_score = base_metric.score(args.metric)
                base_source = str(accepted_checkpoint.parent / "summary-learn.log")
                if args.base_metric_source == "quantized-test" and args.dry_run:
                    base_source += " (dry-run fallback)"
            write_json(
                runner_dir / "config.json",
                {
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "extra_args"},
                    "extra_args": args.extra_args,
                    "initial_theta": theta,
                    "bounds": bounds,
                    "tuned_keys": keys,
                    "initial_base_checkpoint": str(accepted_checkpoint),
                    "initial_base_score": base_score,
                    "accepted_checkpoint_policy": {
                        "keep_trials": args.keep_trials,
                        "epoch_sbs": args.epoch_sbs,
                        "save_rate_accepts": args.save_rate,
                        "accepted_root": str(accepted_root),
                        "current_dir": str(current_dir),
                        "retry_best_dir": str(retry_best_dir),
                    },
                },
            )
            step_scale = args.step_scale
            failed_retries = 0
            accepted_sbs = 0
            retry_best = None
            start_iteration = 1

        print(
            "[theta] loaded "
            + json.dumps({key: theta[key] for key in sorted(theta)}, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(f"[theta] tuned_keys={','.join(keys) if keys else '-'}", flush=True)

        if not (accepted_checkpoint / "state.bin").exists():
            raise FileNotFoundError(f"{accepted_checkpoint / 'state.bin'} does not exist")
        if not (accepted_checkpoint / "dataloader_pos.txt").exists():
            raise FileNotFoundError(f"{accepted_checkpoint / 'dataloader_pos.txt'} does not exist")

        event_line(
            args,
            "BASE",
            (
                f"{accept_threshold_text(args, base_score, base_metric)} "
                f"checkpoint={accepted_checkpoint} source={base_source}"
            ),
            "cyan",
            bold=True,
        )

        if worker is not None and not args.dry_run:
            worker_open_session(args, worker, accepted_checkpoint, theta, runner_dir, log_dir)

        if start_iteration > args.iterations:
            print(
                f"[complete] runner_dir={runner_dir} already reached iteration {start_iteration - 1}; "
                f"--iterations={args.iterations}",
                flush=True,
            )
            return 0

        iteration = start_iteration
        while iteration <= args.iterations:
            old_base_score = base_score
            theta_before = dict(theta)
            step_scale_used = step_scale
            retry = failed_retries + 1
            delta = probe_direction(args.seed, iteration, retry, keys)
            probe_a_theta = make_variant(theta, bounds, delta, +1, step_scale)
            probe_b_theta = make_variant(theta, bounds, delta, -1, step_scale)
            tag_base = f"{args.tag_prefix}-i{iteration:03d}-r{retry:02d}"

            event_line(
                args,
                "TARGET",
                (
                    f"iteration={iteration}/{args.iterations} retry={retry} "
                    f"{accept_threshold_text(args, base_score, base_metric)} "
                    f"step_scale={step_scale:.4f} failed_retries={failed_retries}"
                ),
                "cyan",
                bold=True,
            )
            write_runner_state(
                state_path,
                phase="running_trials",
                iteration=iteration,
                next_iteration=iteration,
                base_checkpoint=accepted_checkpoint,
                base_score=base_score,
                base_metric=base_metric,
                theta=theta,
                step_scale=step_scale,
                failed_retries=failed_retries,
                accepted_sbs=accepted_sbs,
                retry_best=retry_best,
                update_mode=args.update_mode,
            )

            probe_a_trial_number = (retry - 1) * 2 + 1
            probe_b_trial_number = (retry - 1) * 2 + 2

            probe_a = run_trial(
                args,
                accepted_checkpoint,
                f"{tag_base}-probe-a",
                PROBE_A,
                probe_a_trial_number,
                old_base_score,
                base_metric,
                probe_a_theta,
                trial_output_folder,
                log_dir,
                worker,
            )
            trial_threshold_line(args, probe_a, old_base_score, trial_number=probe_a_trial_number)
            probe_b = run_trial(
                args,
                accepted_checkpoint,
                f"{tag_base}-probe-b",
                PROBE_B,
                probe_b_trial_number,
                old_base_score,
                base_metric,
                probe_b_theta,
                trial_output_folder,
                log_dir,
                worker,
            )
            trial_threshold_line(args, probe_b, old_base_score, trial_number=probe_b_trial_number)
            candidates = [probe_a, probe_b]
            best_probe = min(candidates, key=lambda item: item.score)
            best_probe_theta = theta_for_result(best_probe, probe_a_theta, probe_b_theta)
            theta_candidate = best_probe_theta
            probe_score_diff = probe_a.score - probe_b.score
            spsa_gradient_log: dict[str, float] = {}
            spsa_log_update: dict[str, float] = {}
            if args.update_mode == "spsa":
                if math.isfinite(probe_a.score) and math.isfinite(probe_b.score):
                    update = make_spsa_update(
                        theta,
                        bounds,
                        keys,
                        delta,
                        probe_a.score,
                        probe_b.score,
                        step_scale,
                        args.spsa_move_ratio,
                    )
                    theta_candidate = update.theta
                    spsa_gradient_log = update.gradient_log
                    spsa_log_update = update.log_update
                    max_abs_log_update = max((abs(value) for value in spsa_log_update.values()), default=0.0)
                    print(
                        f"[theta] mode=spsa probe_{objective_label(args.metric)}_diff={probe_score_diff:+.9f} "
                        f"move_ratio={args.spsa_move_ratio:.6g} max_abs_log_update={max_abs_log_update:.6g}",
                        flush=True,
                    )
                else:
                    theta_candidate = dict(theta)
                    print("[theta] mode=spsa skipped because trial scores are not finite", flush=True)
            else:
                print("[theta] mode=winner next=best observed probe theta", flush=True)
            probe_decision_line(args, best_probe, old_base_score)
            best = best_probe
            history_retry_best = retry_best
            previous_retry_best_tag = retry_best.result.tag if retry_best is not None else ""

            if best_probe.score < base_score:
                reason = "improved"
                theta = theta_candidate
                if worker is not None and not args.dry_run:
                    best = worker_accept_cached_trial(args, worker, best_probe, log_dir)
                    safe_rmtree(retry_best_dir, runner_dir)
                elif not args.dry_run and not args.keep_trials:
                    accepted_checkpoint = move_checkpoint_to_current(best_probe.checkpoint_dir, current_dir, runner_dir)
                    safe_rmtree(retry_best_dir, runner_dir)
                else:
                    accepted_checkpoint = best_probe.checkpoint_dir
                retry_best = None
                history_retry_best = None
                base_metric = best_probe.metric
                base_score = best_probe.score
                failed_retries = 0
                step_scale = min(args.max_step_scale, max(args.min_step_scale, step_scale * args.step_grow))
                accepted_sbs += args.sb_per_trial
            else:
                if retry_best is None or best_probe.score < retry_best.result.score:
                    if worker is not None and not args.dry_run:
                        retry_best = RetryBest(best_probe, theta_candidate)
                    else:
                        retry_best = stash_retry_best(
                            best_probe,
                            theta_candidate,
                            retry_best_dir,
                            runner_dir,
                            args.keep_trials,
                            args.dry_run,
                        )
                    history_retry_best = retry_best
                    print(
                        f"[retry-best] retry={retry} {objective_label(args.metric)}={best_probe.score:.9f} "
                        f"checkpoint={retry_best.result.checkpoint_dir}",
                        flush=True,
                    )

            if best_probe.score >= old_base_score and failed_retries + 1 >= args.max_retries:
                reason = f"forced_after_{args.max_retries}_retries"
                if retry_best is None:
                    raise RuntimeError("internal error: retry_best is missing at forced acceptance")
                history_retry_best = retry_best
                best = retry_best.result
                theta = retry_best.theta
                if worker is not None and not args.dry_run:
                    best = worker_accept_cached_trial(args, worker, best, log_dir)
                elif not args.dry_run and not args.keep_trials:
                    accepted_checkpoint = move_checkpoint_to_current(retry_best.result.checkpoint_dir, current_dir, runner_dir)
                else:
                    accepted_checkpoint = retry_best.result.checkpoint_dir
                retry_best = None
                base_metric = best.metric
                base_score = best.score
                failed_retries = 0
                step_scale = args.min_step_scale
                accepted_sbs += args.sb_per_trial
            elif best_probe.score >= old_base_score:
                reason = "retry"
                failed_retries += 1

            if worker is not None and not args.dry_run and reason == "retry":
                keep_tag = retry_best.result.tag if retry_best is not None else ""
                drop_keys = [trial.tag for trial in candidates if trial.tag != keep_tag]
                if previous_retry_best_tag and previous_retry_best_tag != keep_tag:
                    drop_keys.append(previous_retry_best_tag)
                worker_drop_cached_trials(
                    worker,
                    sorted(set(drop_keys)),
                    log_dir,
                    stream=not args.no_stream_child_output,
                )

            public_checkpoint = ""
            if reason != "retry" and not args.dry_run:
                if should_save_accepted_checkpoint(args, accepted_sbs):
                    if worker is not None:
                        save_dir = accepted_root / accepted_checkpoint_name(accepted_sbs, args.epoch_sbs)
                        if save_dir.exists():
                            raise FileExistsError(f"accepted checkpoint already exists: {save_dir}")
                        accepted_checkpoint = worker_save_checkpoint(
                            args,
                            worker,
                            accepted_checkpoint,
                            theta,
                            save_dir,
                            accepted_sbs,
                            log_dir,
                        )
                        public_checkpoint = str(accepted_checkpoint)
                    else:
                        public_checkpoint = str(copy_public_checkpoint(accepted_checkpoint, accepted_root, accepted_sbs, args.epoch_sbs))
                        event_line(args, "SAVE", f"accepted_sbs={accepted_sbs} checkpoint={public_checkpoint}", "cyan", bold=True)
            current_retry_best_tag = retry_best.result.tag if retry_best is not None else ""
            forced_retry_best_tag = history_retry_best.result.tag if reason.startswith("forced_after_") and history_retry_best is not None else ""
            if reason == "retry":
                selected_trial_tag = current_retry_best_tag
                selected_result_label = "retry_best"
            elif reason.startswith("forced_after_"):
                selected_trial_tag = forced_retry_best_tag
                selected_result_label = "forced_accepted"
            else:
                selected_trial_tag = best_probe.tag
                selected_result_label = "accepted"
            existing_iteration_rows = retry_window_rows(trial_summary_path, iteration)
            iteration_rows: list[dict[str, Any]] = [dict(row) for row in existing_iteration_rows]
            for trial in candidates:
                iteration_rows.append(
                    make_trial_summary_row(
                        iteration=iteration,
                        retry=retry,
                        result_label="pending",
                        reason=reason,
                        accepted_sbs=accepted_sbs,
                        trial=trial,
                        trial_theta=theta_for_result(trial, probe_a_theta, probe_b_theta),
                        theta_before=theta_before,
                        keys=keys,
                        saved_checkpoint=Path(public_checkpoint).name if public_checkpoint else "",
                        update_mode=args.update_mode,
                        step_scale_used=step_scale_used,
                        step_scale_next=step_scale,
                        spsa_move_ratio=args.spsa_move_ratio,
                    )
                )
            public_checkpoint_name = Path(public_checkpoint).name if public_checkpoint else ""
            if reason == "retry":
                if worker is not None and not args.dry_run:
                    selected_checkpoint_text = f"worker-cache:{selected_trial_tag}"
                elif retry_best is not None:
                    selected_checkpoint_text = str(retry_best.result.checkpoint_dir)
                else:
                    selected_checkpoint_text = ""
            elif worker is not None and not args.dry_run and not public_checkpoint:
                selected_checkpoint_text = f"worker-resident:{selected_trial_tag}"
            else:
                selected_checkpoint_text = str(accepted_checkpoint)
            for row in iteration_rows:
                trial_tag = str(row.get("trial_tag", ""))
                row["reason"] = reason
                row["accepted_sbs"] = accepted_sbs
                if trial_tag == selected_trial_tag:
                    row["result"] = selected_result_label
                    if selected_checkpoint_text:
                        row["checkpoint"] = selected_checkpoint_text
                    row["saved_checkpoint"] = public_checkpoint_name
                else:
                    row["result"] = "discarded"
                    row["saved_checkpoint"] = ""
            rewrite_trial_summary_iteration(trial_summary_path, iteration, iteration_rows)

            if reason != "retry" and not args.dry_run:
                append_accepted_summary(
                    accepted_summary_path,
                    {
                        "iteration": iteration,
                        "accepted_sbs": accepted_sbs,
                        "reason": reason,
                        "quantized_value_loss": fmt_metric(best.metric.qloss),
                        "quantized_value_accuracy": fmt_metric(best.metric.qacc),
                        "test_value_loss": fmt_metric(best.metric.test_loss),
                        "test_value_accuracy": fmt_metric(best.metric.test_acc),
                        "saved_checkpoint": Path(public_checkpoint).name if public_checkpoint else "",
                        "update_mode": args.update_mode,
                        "step_scale_used": f"{step_scale_used:.9f}",
                        "step_scale_next": f"{step_scale:.9f}",
                        "spsa_move_ratio": f"{args.spsa_move_ratio:.9f}",
                        "theta_change": theta_change_text(theta_before, theta, keys),
                        "theta_before_json": json.dumps(theta_before, ensure_ascii=False, sort_keys=True),
                        "theta_delta_json": json.dumps(theta_delta(theta_before, theta), ensure_ascii=False, sort_keys=True),
                        "theta_json": json.dumps(theta, ensure_ascii=False, sort_keys=True),
                    },
                )
            if not args.keep_trials and not args.dry_run:
                prune_trial_outputs(candidates, trial_output_folder)

            append_history(
                history_path,
                {
                    "iteration": iteration,
                    "retry": retry,
                    "reason": reason,
                    "base_score_before": f"{old_base_score:.9f}",
                    "probe_a_score": f"{probe_a.score:.9f}",
                    "probe_b_score": f"{probe_b.score:.9f}",
                    "probe_score_diff": f"{probe_score_diff:.9f}",
                    "best_probe_score": f"{best_probe.score:.9f}",
                    "new_base_score": f"{base_score:.9f}",
                    "new_base_checkpoint": str(accepted_checkpoint),
                    "accepted_sbs": accepted_sbs,
                    "public_checkpoint": public_checkpoint,
                    "retry_best_score": f"{history_retry_best.result.score:.9f}" if history_retry_best is not None else "",
                    "retry_best_checkpoint": str(history_retry_best.result.checkpoint_dir) if history_retry_best is not None else "",
                    "update_mode": args.update_mode,
                    "step_scale_used": f"{step_scale_used:.9f}",
                    "step_scale_next": f"{step_scale:.9f}",
                    "spsa_move_ratio": f"{args.spsa_move_ratio:.9f}",
                    "theta_change": theta_change_text(theta_before, theta, keys),
                    "theta_before_json": json.dumps(theta_before, ensure_ascii=False, sort_keys=True),
                    "theta_delta_json": json.dumps(theta_delta(theta_before, theta), ensure_ascii=False, sort_keys=True),
                    "theta_json": json.dumps(theta, ensure_ascii=False, sort_keys=True),
                    "theta_candidate_json": json.dumps(theta_candidate, ensure_ascii=False, sort_keys=True),
                    "spsa_gradient_log_json": json.dumps(spsa_gradient_log, ensure_ascii=False, sort_keys=True),
                    "spsa_log_update_json": json.dumps(spsa_log_update, ensure_ascii=False, sort_keys=True),
                    "delta_json": json.dumps(delta, ensure_ascii=False, sort_keys=True),
                },
            )
            next_iteration = iteration if reason == "retry" else iteration + 1
            phase = "between_retries" if reason == "retry" else "between_iterations"
            write_runner_state(
                state_path,
                phase=phase,
                iteration=iteration,
                next_iteration=next_iteration,
                base_checkpoint=accepted_checkpoint,
                base_score=base_score,
                base_metric=base_metric,
                theta=theta,
                step_scale=step_scale,
                failed_retries=failed_retries,
                accepted_sbs=accepted_sbs,
                retry_best=retry_best,
                update_mode=args.update_mode,
            )
            theta_change = theta_change_text(theta_before, theta, keys) or "-"
            if reason == "retry":
                event_line(
                    args,
                    "RETRY",
                    (
                        f"iteration={iteration} retry={retry} accepted_sbs={accepted_sbs} "
                        f"{score_compare_text(args, best_probe.score, old_base_score, value_name='best_trial')} "
                        f"decision=retry because best_trial did not beat start "
                        f"same_step_scale={step_scale:.9f}"
                    ),
                    "yellow",
                    bold=True,
                )
            else:
                decision_score = best.score
                forced_accept = decision_score >= old_base_score
                decision = "accept because final beat start" if not forced_accept else f"force_accept after {args.max_retries} retries"
                event_line(
                    args,
                    "FORCE" if forced_accept else "ACCEPT",
                    (
                        f"iteration={iteration} retry={retry} accepted_sbs={accepted_sbs} "
                        f"reason={reason} {score_compare_text(args, decision_score, old_base_score, value_name='final')} "
                        f"decision={decision} "
                        f"theta_change={theta_change}"
                    ),
                    "yellow" if forced_accept else "green",
                    bold=True,
                )
                if public_checkpoint:
                    event_line(
                        args,
                        "SAFE TO STOP",
                        f"saved_checkpoint={public_checkpoint}",
                        "green",
                        bold=True,
                    )
                elif worker is not None and not args.dry_run:
                    save_at = next_save_accept_count(args, accepted_sbs)
                    suffix = f" next_save_accept={save_at}" if save_at is not None else " public_save_disabled"
                    event_line(
                        args,
                        "WAIT FOR SAVE",
                        "accepted state is GPU-resident; stopping now loses progress since last save." + suffix,
                        "yellow",
                        bold=True,
                    )
                else:
                    event_line(
                        args,
                        "SAFE TO STOP",
                        f"runner_state_written checkpoint={accepted_checkpoint}",
                        "green",
                        bold=True,
                    )
            iteration = next_iteration

        write_runner_state(
            state_path,
            phase="complete",
            iteration=args.iterations,
            next_iteration=args.iterations + 1,
            base_checkpoint=accepted_checkpoint,
            base_score=base_score,
            base_metric=base_metric,
            theta=theta,
            step_scale=step_scale,
            failed_retries=failed_retries,
            accepted_sbs=accepted_sbs,
            retry_best=retry_best,
            update_mode=args.update_mode,
            complete=True,
        )
        print(f"[complete] runner_dir={runner_dir}", flush=True)
        print(f"[complete] best_checkpoint={accepted_checkpoint}", flush=True)
        print(f"[complete] best_{objective_label(args.metric)}={base_score:.9f}", flush=True)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
