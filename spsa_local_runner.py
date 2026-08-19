#!/usr/bin/env python3
"""Local SPSA-style runner for BulletOu fine-tuning experiments.

This script does not change BulletOu's training algorithm.  It repeatedly
branches from a checkpoint, runs two short trials with slightly different
factorizer hyperparameters, and adopts the better checkpoint according to
quantized validation loss.

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
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short plus/minus BulletOu trials and adopt better factorizer hyperparameters."
    )
    parser.add_argument("--exe", required=True, help="Path to bulletou.exe")
    parser.add_argument("--base-checkpoint", required=True, type=Path, help="Checkpoint directory containing state.bin")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--test-teacher", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--bucket-counts", type=Path, default=None)
    parser.add_argument("--output-folder", required=True, type=Path, help="Root folder for runner outputs")
    parser.add_argument("--runner-dir", type=Path, default=None, help="Exact runner directory. Default: output-folder/spsa-...")
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--factorizer", default="pair")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sb-per-trial", type=int, default=8)
    parser.add_argument("--positions-per-superbatch", type=int, default=40_000_000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-score", type=float, default=None, help="Base score override. Lower is better.")
    parser.add_argument("--metric", choices=["quantized_value_loss", "test_value_loss"], default="quantized_value_loss")
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
            "axis_count, pair_count. Default: non-zero parameters except shared_alpha."
        ),
    )
    parser.add_argument("--fixed", action="append", default=[], help="Parameter to keep fixed. Repeatable.")
    parser.add_argument("--step-scale", type=float, default=1.25, help="Initial multiplicative perturbation scale")
    parser.add_argument("--step-shrink", type=float, default=0.70, help="Shrink factor when both trials are worse")
    parser.add_argument("--step-grow", type=float, default=1.03, help="Grow factor after an improving acceptance")
    parser.add_argument("--min-step-scale", type=float, default=1.02)
    parser.add_argument("--max-step-scale", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--no-stream-child-output",
        action="store_true",
        help="Do not mirror bulletou.exe stdout to the console. Logs are still saved.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to bulletou.exe")
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")
    if args.sb_per_trial <= 0:
        parser.error("--sb-per-trial must be > 0")
    if args.positions_per_superbatch <= 0:
        parser.error("--positions-per-superbatch must be > 0")
    for name in ["step_scale", "step_shrink", "step_grow", "min_step_scale", "max_step_scale"]:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and > 0")
    if args.step_scale <= 1.0:
        parser.error("--step-scale should be > 1")
    if args.min_step_scale <= 1.0:
        parser.error("--min-step-scale should be > 1")
    if args.max_step_scale < args.min_step_scale:
        parser.error("--max-step-scale must be >= --min-step-scale")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def perturb_value(value: float, sign: int, step_scale: float, lo: float, hi: float) -> float:
    if lo == hi:
        return lo
    offset = max(1.0e-6, min(0.01, (hi - lo) * 1.0e-4))
    y = math.log(max(value, lo) + offset)
    y += sign * math.log(step_scale)
    return min(max(math.exp(y) - offset, lo), hi)


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
        str(args.sb_per_trial),
        "--validation-rate",
        str(args.sb_per_trial),
        "--quantized-validation-rate",
        str(args.sb_per_trial),
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


def run_trial(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    tag: str,
    side: str,
    theta: dict[str, float],
    trial_output_folder: Path,
    log_dir: Path,
) -> TrialResult:
    cmd = base_command(args, checkpoint_dir, tag, theta, trial_output_folder)
    log_path = log_dir / f"{tag}.stdout.log"
    print(f"[run] {side} tag={tag}", flush=True)
    print("      " + subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        return TrialResult(side, tag, log_dir, checkpoint_dir, Metric(None, None, None, None), float("inf"))
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
            log.write(line)
            log.flush()
            if not args.no_stream_child_output:
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = proc.wait()
    elapsed = time.time() - start
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
        f"[done] {side} tag={tag} score={score:.9f} "
        f"qloss={fmt_metric(metric.qloss)} qacc={fmt_metric(metric.qacc)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return TrialResult(side, tag, output_dir, next_checkpoint, metric, score)


def fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.9f}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main() -> int:
    args = parse_args()
    try:
        bounds = load_bounds(args.bounds_json)
        theta = clamp_theta(load_theta(args.theta_json, args.theta), bounds)
        keys = tuned_keys(args, theta)
        rng = random.Random(args.seed)

        base_checkpoint = args.base_checkpoint.resolve()
        if not (base_checkpoint / "state.bin").exists():
            raise FileNotFoundError(f"{base_checkpoint / 'state.bin'} does not exist")
        if not (base_checkpoint / "dataloader_pos.txt").exists():
            raise FileNotFoundError(f"{base_checkpoint / 'dataloader_pos.txt'} does not exist")

        runner_dir = args.runner_dir or (args.output_folder / f"spsa-{args.tag_prefix}-{now_stamp()}")
        trial_output_folder = runner_dir / "trials"
        log_dir = runner_dir / "logs"
        runner_dir.mkdir(parents=True, exist_ok=True)
        trial_output_folder.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        if args.base_score is not None:
            base_metric = Metric(None, None, None, None)
            base_score = args.base_score
        else:
            base_metric = checkpoint_metric(base_checkpoint, args.metric)
            base_score = base_metric.score(args.metric)

        state_path = runner_dir / "state.json"
        history_path = runner_dir / "history.csv"
        write_json(
            runner_dir / "config.json",
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "extra_args"},
                "extra_args": args.extra_args,
                "initial_theta": theta,
                "bounds": bounds,
                "tuned_keys": keys,
                "initial_base_checkpoint": str(base_checkpoint),
                "initial_base_score": base_score,
            },
        )

        step_scale = args.step_scale
        failed_retries = 0
        accepted_checkpoint = base_checkpoint

        for iteration in range(1, args.iterations + 1):
            old_base_score = base_score
            delta = {key: rng.choice([-1, 1]) for key in keys}
            plus_theta = make_variant(theta, bounds, delta, +1, step_scale)
            minus_theta = make_variant(theta, bounds, delta, -1, step_scale)
            retry = failed_retries + 1
            tag_base = f"{args.tag_prefix}-i{iteration:03d}-r{retry:02d}"

            print(
                f"[iter] {iteration}/{args.iterations} base_score={base_score:.9f} "
                f"step_scale={step_scale:.4f} failed_retries={failed_retries}",
                flush=True,
            )
            write_json(
                state_path,
                {
                    "iteration": iteration,
                    "base_checkpoint": str(accepted_checkpoint),
                    "base_score": base_score,
                    "base_metric": base_metric.__dict__,
                    "theta": theta,
                    "step_scale": step_scale,
                    "failed_retries": failed_retries,
                    "delta": delta,
                },
            )

            plus = run_trial(args, accepted_checkpoint, f"{tag_base}-plus", "plus", plus_theta, trial_output_folder, log_dir)
            minus = run_trial(args, accepted_checkpoint, f"{tag_base}-minus", "minus", minus_theta, trial_output_folder, log_dir)
            candidates = [plus, minus]
            best = min(candidates, key=lambda item: item.score)

            if best.score < base_score:
                reason = "improved"
                theta = plus_theta if best.side == "plus" else minus_theta
                accepted_checkpoint = best.checkpoint_dir
                base_metric = best.metric
                base_score = best.score
                failed_retries = 0
                step_scale = min(args.max_step_scale, max(args.min_step_scale, step_scale * args.step_grow))
            elif failed_retries + 1 >= args.max_retries:
                reason = f"forced_after_{args.max_retries}_retries"
                theta = plus_theta if best.side == "plus" else minus_theta
                accepted_checkpoint = best.checkpoint_dir
                base_metric = best.metric
                base_score = best.score
                failed_retries = 0
                step_scale = args.min_step_scale
            else:
                reason = "retry_with_smaller_step"
                failed_retries += 1
                step_scale = max(args.min_step_scale, step_scale * args.step_shrink)

            append_history(
                history_path,
                {
                    "iteration": iteration,
                    "retry": retry,
                    "reason": reason,
                    "accepted": best.side if reason != "retry_with_smaller_step" else "base",
                    "base_score_before": f"{old_base_score:.9f}",
                    "plus_score": f"{plus.score:.9f}",
                    "minus_score": f"{minus.score:.9f}",
                    "new_base_score": f"{base_score:.9f}",
                    "new_base_checkpoint": str(accepted_checkpoint),
                    "step_scale": f"{step_scale:.9f}",
                    "theta_json": json.dumps(theta, ensure_ascii=False, sort_keys=True),
                    "delta_json": json.dumps(delta, ensure_ascii=False, sort_keys=True),
                },
            )
            print(f"[accept] reason={reason} checkpoint={accepted_checkpoint} score={base_score:.9f}", flush=True)

        write_json(
            state_path,
            {
                "iteration": args.iterations,
                "base_checkpoint": str(accepted_checkpoint),
                "base_score": base_score,
                "base_metric": base_metric.__dict__,
                "theta": theta,
                "step_scale": step_scale,
                "failed_retries": failed_retries,
                "complete": True,
            },
        )
        print(f"[complete] runner_dir={runner_dir}", flush=True)
        print(f"[complete] best_checkpoint={accepted_checkpoint}", flush=True)
        print(f"[complete] best_score={base_score:.9f}", flush=True)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
