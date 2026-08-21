#!/usr/bin/env python3
"""Beam-style ES runner for BulletOu local fine-tuning.

This runner is intentionally simple:

* `parameters.json` owns the current hyperparameters and ES settings.
* Each generation creates a population of randomized candidates.
* Candidates are trained for the configured beam stages.
* At each stage, candidates are ranked by the configured metric and pruned.
* The final survivor's NN weights and hyperparameters become the next current
  state.

There is no gradient estimate and no partial parameter update.  The selected
candidate itself survives.

Typical use:

    python es_local_runner.py ^
      --exe .\\target\\release\\examples\\bulletou.exe ^
      --parameters-file .\\parameters.json ^
      --base-checkpoint C:\\...\\0256 ^
      --teacher D:\\sojoteam_datasets ^
      --test-teacher C:\\shogi\\teacher\\test\\test.hcpe ^
      --arch SFNN_halfka2_1024_8_64_hand1024_k3k3_progress4 ^
      --bucket-counts D:\\sojo_counts\\count-all.bin ^
      --output-folder D:\\BulletOu-snapshots\\20260820 ^
      --temp-folder C:\\BulletOu-es-temp ^
      --tag-prefix pair2-qloss ^
      -- --lr 0.000030 --lr-min 0.000010 --wrm-in-offset 0 --wrm-target-offset 0

Arguments after `--` are passed through to `bulletou.exe`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PARAMETER_VERSION = 1
STATE_VERSION = 1


ALPHA_PARAMETERS = {
    "shared": "shared",
    "king_axis": "king_axis",
    "hand_axis": "hand_axis",
    "progress_axis": "progress_axis",
    "pair": "pair",
}


CONFIDENCE_FLAGS = {
    "residual_count": "--sfnn-residual-count-confidence",
    "axis_count": "--sfnn-axis-count-confidence",
    "king_axis_count": "--sfnn-king-axis-count-confidence",
    "hand_axis_count": "--sfnn-hand-axis-count-confidence",
    "progress_axis_count": "--sfnn-progress-axis-count-confidence",
    "pair_count": "--sfnn-pair-count-confidence",
    "king_hand_pair_count": "--sfnn-king-hand-pair-count-confidence",
    "king_progress_pair_count": "--sfnn-king-progress-pair-count-confidence",
    "hand_progress_pair_count": "--sfnn-hand-progress-pair-count-confidence",
}


KNOWN_PARAMETERS = set(ALPHA_PARAMETERS) | set(CONFIDENCE_FLAGS)


DEFAULT_PARAMETER_SPECS: dict[str, dict[str, float | bool]] = {
    "shared": {"current": 1.0, "tune": False, "step": 0.0, "min": 0.0, "max": 10.0},
    "king_axis": {"current": 1.0, "tune": False, "step": 0.03, "min": 0.0, "max": 10.0},
    "hand_axis": {"current": 1.0, "tune": False, "step": 0.03, "min": 0.0, "max": 10.0},
    "progress_axis": {"current": 1.0, "tune": False, "step": 0.03, "min": 0.0, "max": 10.0},
    "pair": {"current": 1.0, "tune": False, "step": 0.03, "min": 0.0, "max": 10.0},
    "residual_count": {"current": 0.0, "tune": False, "step": 0.25, "min": 0.0, "max": 20.0},
    "axis_count": {"current": 0.0, "tune": False, "step": 0.25, "min": 0.0, "max": 100.0},
    "king_axis_count": {"current": 0.0, "tune": False, "step": 0.50, "min": 0.0, "max": 100.0},
    "hand_axis_count": {"current": 0.0, "tune": False, "step": 0.50, "min": 0.0, "max": 100.0},
    "progress_axis_count": {"current": 0.0, "tune": False, "step": 0.50, "min": 0.0, "max": 100.0},
    "pair_count": {"current": 0.0, "tune": False, "step": 1.00, "min": 0.0, "max": 200.0},
    "king_hand_pair_count": {"current": 0.0, "tune": False, "step": 1.00, "min": 0.0, "max": 200.0},
    "king_progress_pair_count": {"current": 0.0, "tune": False, "step": 1.00, "min": 0.0, "max": 200.0},
    "hand_progress_pair_count": {"current": 0.0, "tune": False, "step": 1.00, "min": 0.0, "max": 200.0},
}


RUNNER_CONTROLLED_FLAGS = {
    "--backend",
    "--teacher",
    "--test-teacher",
    "--arch",
    "--initial-state",
    "--initial-dataloader-pos",
    "--output",
    "--output-folder",
    "--tag",
    "--resume",
    "--no-resume",
    "--positions-per-superbatch",
    "--superbatches",
    "--max-epochs",
    "--save-rate",
    "--validation-rate",
    "--quantized-validation-rate",
    "--sfnn-factorizer",
    "--sfnn-factorizer-alpha",
    "--sfnn-bucket-counts",
    *CONFIDENCE_FLAGS.values(),
}


SUMMARY_FIELDS = [
    "generation",
    "stage_sbs",
    "candidate",
    "status",
    "rank",
    "score",
    "quantized_value_loss",
    "quantized_value_accuracy",
    "test_value_loss",
    "test_value_accuracy",
    "checkpoint",
    "output_dir",
    "parameters_json",
]


ACCEPTED_FIELDS = [
    "generation",
    "accepted_sbs",
    "score",
    "quantized_value_loss",
    "quantized_value_accuracy",
    "test_value_loss",
    "test_value_accuracy",
    "stage_sbs",
    "saved_checkpoint",
    "current_checkpoint",
    "parameters_json",
]


ANSI = {
    "reset": "\x1b[0m",
    "bold": "\x1b[1m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
    "magenta": "\x1b[35m",
}


@dataclass
class Metric:
    qloss: float | None = None
    qacc: float | None = None
    test_loss: float | None = None
    test_acc: float | None = None

    def score(self, metric_name: str) -> float:
        value: float | None
        if metric_name == "quantized_value_loss":
            value = self.qloss
        elif metric_name == "quantized_value_accuracy":
            value = self.qacc
        elif metric_name == "test_value_loss":
            value = self.test_loss
        elif metric_name == "test_value_accuracy":
            value = self.test_acc
        else:
            raise ValueError(f"unsupported metric {metric_name!r}")
        if value is None:
            raise ValueError(f"metric {metric_name!r} was not written by the candidate run")
        return value


@dataclass
class ParameterSpec:
    current: float
    tune: bool
    step: float
    minimum: float
    maximum: float

    def clamp(self, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("parameter value must be finite")
        return min(max(value, self.minimum), self.maximum)


@dataclass
class BeamStage:
    after_sbs: int
    keep: int


@dataclass
class EsSettings:
    generations: int
    population: int
    beam: list[BeamStage]
    metric: str
    lower_is_better: bool
    seed: int
    save_rate: int
    candidate_validation_rate: int
    candidate_quantized_validation_rate: int


@dataclass
class Candidate:
    index: int
    params: dict[str, float]
    checkpoint: Path
    output_dir: Path | None = None
    metric: Metric | None = None
    score: float | None = None
    stage_sbs: int = 0
    transient_dirs: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a beam-style ES search for BulletOu factorizer/count hyperparameters."
    )
    parser.add_argument("--exe", required=True, type=Path, help="Path to bulletou.exe")
    parser.add_argument("--parameters-file", type=Path, default=Path("parameters.json"))
    parser.add_argument("--base-checkpoint", type=Path, default=None, help="Initial checkpoint directory containing state.bin")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--test-teacher", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--bucket-counts", type=Path, default=None)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--temp-folder", type=Path, default=None, help="Temporary candidate checkpoint root. Default: runner temp/ under output-folder.")
    parser.add_argument("--tag-prefix", required=True)
    parser.add_argument("--factorizer", default="pair")
    parser.add_argument("--positions-per-superbatch", type=int, default=40_000_000)
    parser.add_argument("--generations", type=int, default=None, help="Override es.generations from parameters.json")
    parser.add_argument("--save-rate", type=int, default=None, help="Override es.save_rate. N=1 saves every accepted generation.")
    parser.add_argument(
        "--metric",
        choices=["quantized_value_loss", "quantized_value_accuracy", "test_value_loss", "test_value_accuracy"],
        default=None,
        help="Override es.metric",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from output-folder/es-<tag-prefix>/runner-state.json")
    parser.add_argument("--keep-temp", action="store_true", help="Keep candidate temp directories for debugging")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-stream-child-output",
        action="store_true",
        help="Do not mirror bulletou.exe stdout to console; logs are still written under runner logs/.",
    )
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to bulletou.exe")
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    if args.positions_per_superbatch <= 0:
        parser.error("--positions-per-superbatch must be > 0")
    if args.generations is not None and args.generations <= 0:
        parser.error("--generations must be > 0")
    if args.save_rate is not None and args.save_rate < 0:
        parser.error("--save-rate must be >= 0")
    if not args.resume and args.base_checkpoint is None:
        parser.error("--base-checkpoint is required unless --resume is specified")
    for token in args.extra_args:
        if token in RUNNER_CONTROLLED_FLAGS:
            parser.error(f"{token} is controlled by es_local_runner.py; remove it from arguments after --")
    return args


def color_enabled(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def paint(enabled: bool, text: str, color: str) -> str:
    if not enabled:
        return text
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def event(enabled: bool, label: str, message: str, color: str = "cyan") -> None:
    print(f"{paint(enabled, label, color)} {message}", flush=True)


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def parse_parameter_spec(name: str, value: Any) -> ParameterSpec:
    defaults = DEFAULT_PARAMETER_SPECS[name]
    if isinstance(value, (int, float)):
        raw: dict[str, Any] = {"current": float(value)}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError(f"parameters.{name} must be a number or object")

    current = float(raw.get("current", defaults["current"]))
    tune = bool(raw.get("tune", defaults["tune"]))
    step = float(raw.get("step", defaults["step"]))
    minimum = float(raw.get("min", defaults["min"]))
    maximum = float(raw.get("max", defaults["max"]))

    if minimum > maximum:
        raise ValueError(f"parameters.{name}: min must be <= max")
    if step < 0.0 or not math.isfinite(step):
        raise ValueError(f"parameters.{name}: step must be finite and >= 0")
    spec = ParameterSpec(current=current, tune=tune, step=step, minimum=minimum, maximum=maximum)
    spec.current = spec.clamp(spec.current)
    if spec.tune and spec.step == 0.0:
        raise ValueError(f"parameters.{name}: tune=true requires step > 0")
    return spec


def load_parameters(path: Path) -> tuple[dict[str, Any], dict[str, ParameterSpec], EsSettings]:
    root = load_json_object(path)
    version = int(root.get("version", PARAMETER_VERSION))
    if version != PARAMETER_VERSION:
        raise ValueError(f"{path}: unsupported version {version}; expected {PARAMETER_VERSION}")

    params_obj = root.get("parameters")
    if not isinstance(params_obj, dict):
        raise ValueError(f"{path}: `parameters` object is required")

    unknown = sorted(set(params_obj) - KNOWN_PARAMETERS)
    if unknown:
        raise ValueError(f"{path}: unknown parameter(s): {', '.join(unknown)}")

    specs: dict[str, ParameterSpec] = {}
    for name in sorted(KNOWN_PARAMETERS):
        specs[name] = parse_parameter_spec(name, params_obj.get(name, DEFAULT_PARAMETER_SPECS[name]))

    es_obj = root.get("es")
    if not isinstance(es_obj, dict):
        raise ValueError(f"{path}: `es` object is required")

    generations = int(es_obj.get("generations", 1))
    population = int(es_obj.get("population", 4))
    metric = str(es_obj.get("metric", "quantized_value_loss"))
    if metric not in {
        "quantized_value_loss",
        "quantized_value_accuracy",
        "test_value_loss",
        "test_value_accuracy",
    }:
        raise ValueError(f"{path}: unsupported es.metric {metric!r}")
    lower_is_better = bool(es_obj.get("lower_is_better", "loss" in metric))
    seed = int(es_obj.get("seed", 1))
    save_rate = int(es_obj.get("save_rate", 1))
    candidate_validation_rate = int(es_obj.get("candidate_validation_rate", 1))
    candidate_quantized_validation_rate = int(es_obj.get("candidate_quantized_validation_rate", 1))

    if generations <= 0:
        raise ValueError("es.generations must be > 0")
    if population <= 0:
        raise ValueError("es.population must be > 0")
    if save_rate < 0:
        raise ValueError("es.save_rate must be >= 0")
    if candidate_validation_rate <= 0:
        raise ValueError("es.candidate_validation_rate must be > 0")
    if candidate_quantized_validation_rate <= 0:
        raise ValueError("es.candidate_quantized_validation_rate must be > 0")

    beam_raw = es_obj.get("beam")
    if beam_raw is None:
        candidate_sbs = int(es_obj.get("candidate_sbs", 8))
        beam = [BeamStage(after_sbs=candidate_sbs, keep=1)]
    elif isinstance(beam_raw, list):
        beam = []
        for i, item in enumerate(beam_raw):
            if not isinstance(item, dict):
                raise ValueError(f"es.beam[{i}] must be an object")
            beam.append(BeamStage(after_sbs=int(item["after_sbs"]), keep=int(item["keep"])))
    else:
        raise ValueError("es.beam must be a list")

    if not beam:
        raise ValueError("es.beam must not be empty")
    prev_after = 0
    prev_keep = population
    for stage in beam:
        if stage.after_sbs <= prev_after:
            raise ValueError("es.beam after_sbs must be strictly increasing")
        if stage.keep <= 0 or stage.keep > prev_keep:
            raise ValueError("es.beam keep must be > 0 and <= previous live candidate count")
        prev_after = stage.after_sbs
        prev_keep = stage.keep
    if beam[-1].keep != 1:
        raise ValueError("final es.beam stage must keep exactly 1 candidate")

    settings = EsSettings(
        generations=generations,
        population=population,
        beam=beam,
        metric=metric,
        lower_is_better=lower_is_better,
        seed=seed,
        save_rate=save_rate,
        candidate_validation_rate=candidate_validation_rate,
        candidate_quantized_validation_rate=candidate_quantized_validation_rate,
    )
    return root, specs, settings


def write_current_parameters(path: Path, root: dict[str, Any], specs: dict[str, ParameterSpec]) -> None:
    params_obj: dict[str, Any] = {}
    for name in sorted(KNOWN_PARAMETERS):
        original = root.get("parameters", {}).get(name, {})
        if isinstance(original, dict):
            obj = dict(original)
        else:
            obj = {}
        spec = specs[name]
        obj["current"] = spec.current
        obj["tune"] = spec.tune
        obj["step"] = spec.step
        obj["min"] = spec.minimum
        obj["max"] = spec.maximum
        params_obj[name] = obj
    root["parameters"] = params_obj
    root["version"] = PARAMETER_VERSION
    atomic_write_json(path, root)


def current_values(specs: dict[str, ParameterSpec]) -> dict[str, float]:
    return {name: specs[name].current for name in sorted(KNOWN_PARAMETERS)}


def set_current_values(specs: dict[str, ParameterSpec], values: dict[str, float]) -> None:
    for name, value in values.items():
        specs[name].current = specs[name].clamp(float(value))


def perturb_parameters(specs: dict[str, ParameterSpec], rng: random.Random) -> dict[str, float]:
    out = current_values(specs)
    for name, spec in specs.items():
        if spec.tune:
            delta = rng.uniform(-spec.step, spec.step)
            out[name] = spec.clamp(spec.current + delta)
    return out


def alpha_arg(params: dict[str, float]) -> str:
    parts = []
    for name, cli_name in ALPHA_PARAMETERS.items():
        if name in params:
            parts.append(f"{cli_name}={params[name]:.9g}")
    return ",".join(parts)


def parameter_args(params: dict[str, float], bucket_counts: Path | None) -> list[str]:
    out = ["--sfnn-factorizer-alpha", alpha_arg(params)]
    if bucket_counts is not None:
        out.extend(["--sfnn-bucket-counts", str(bucket_counts)])
    for name, flag in CONFIDENCE_FLAGS.items():
        value = params.get(name)
        if value is not None and abs(value) > 0.0:
            out.extend([flag, f"{value:.9g}"])
    return out


def format_float(value: float | None, digits: int = 9) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}g}"


def parse_float_cell(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    return float(value)


def metric_from_summary_row(row: dict[str, str]) -> Metric:
    return Metric(
        qloss=parse_float_cell(row.get("quantized_value_loss")),
        qacc=parse_float_cell(row.get("quantized_value_accuracy")),
        test_loss=parse_float_cell(row.get("test_value_loss")),
        test_acc=parse_float_cell(row.get("test_value_accuracy")),
    )


def latest_summary_row(output_dir: Path) -> dict[str, str]:
    path = output_dir / "summary-learn.log"
    if not path.exists():
        raise RuntimeError(f"{path} was not written")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if any((value or "").strip() for value in row.values())]
    if not rows:
        raise RuntimeError(f"{path} has no data rows")
    return rows[-1]


def latest_checkpoint_dir(output_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "state.bin").exists():
            candidates.append((int(child.name), child))
    if not candidates:
        raise RuntimeError(f"no checkpoint directory containing state.bin found under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def ensure_csv(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = f.readline().strip()
        expected = ",".join(fields)
        if header != expected:
            raise RuntimeError(f"{path} has incompatible header\n  existing: {header}\n  expected: {expected}")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    ensure_csv(path, fields)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writerow({field: row.get(field, "") for field in fields})


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def copy_dir_replace(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"source directory does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".new")
    old = dst.with_name(dst.name + ".old")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        os.replace(dst, old)
    os.replace(tmp, dst)
    if old.exists():
        shutil.rmtree(old)


def copy_dir_new(src: Path, dst: Path) -> None:
    if dst.exists():
        raise RuntimeError(f"destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def remove_dir_quiet(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def run_command(cmd: list[str], log_path: Path, stream: bool) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace", newline="") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if stream:
                print(line, end="")
        code = proc.wait()
    return code, time.perf_counter() - start


def build_train_command(
    args: argparse.Namespace,
    params: dict[str, float],
    checkpoint: Path,
    output_dir: Path,
    stage_delta_sbs: int,
    settings: EsSettings,
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
        str(args.arch),
        "--initial-state",
        str(checkpoint / "state.bin"),
        "--initial-dataloader-pos",
        str(checkpoint / "dataloader_pos.txt"),
        "--output",
        str(output_dir),
        "--sfnn-factorizer",
        str(args.factorizer),
        "--positions-per-superbatch",
        str(args.positions_per_superbatch),
        "--superbatches",
        str(stage_delta_sbs),
        "--max-epochs",
        "1",
        "--save-rate",
        str(stage_delta_sbs),
        "--validation-rate",
        str(max(1, min(settings.candidate_validation_rate, stage_delta_sbs))),
        "--quantized-validation-rate",
        str(max(1, min(settings.candidate_quantized_validation_rate, stage_delta_sbs))),
    ]
    if args.bucket_counts is not None:
        cmd.extend(["--sfnn-bucket-counts", str(args.bucket_counts)])
    cmd.extend(parameter_args(params, None))
    cmd.extend(args.extra_args)
    return cmd


def train_candidate_stage(
    args: argparse.Namespace,
    settings: EsSettings,
    generation: int,
    candidate: Candidate,
    stage: BeamStage,
    prev_after_sbs: int,
    temp_root: Path,
    log_dir: Path,
    color: bool,
) -> Candidate:
    delta = stage.after_sbs - prev_after_sbs
    out_dir = temp_root / f"gen{generation:04d}" / f"stage{stage.after_sbs:04d}" / f"cand{candidate.index:03d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gen{generation:04d}-stage{stage.after_sbs:04d}-cand{candidate.index:03d}.stdout.log"
    event(
        color,
        f"[CAND {candidate.index:03d} START]",
        f"generation={generation} stage={stage.after_sbs}sb delta={delta}sb",
        "cyan",
    )
    cmd = build_train_command(args, candidate.params, candidate.checkpoint, out_dir, delta, settings)
    if args.dry_run:
        print("  " + subprocess.list2cmdline(cmd), flush=True)
        metric = Metric(qloss=math.inf, qacc=None, test_loss=None, test_acc=None)
        score = metric.score(settings.metric)
        checkpoint = candidate.checkpoint
        elapsed = 0.0
    else:
        code, elapsed = run_command(cmd, log_path, stream=not args.no_stream_child_output)
        if code != 0:
            raise RuntimeError(f"candidate {candidate.index} failed at stage {stage.after_sbs}sb; see {log_path}")
        row = latest_summary_row(out_dir)
        metric = metric_from_summary_row(row)
        score = metric.score(settings.metric)
        checkpoint = latest_checkpoint_dir(out_dir)

    event(
        color,
        f"[CAND {candidate.index:03d} END]",
        (
            f"generation={generation} stage={stage.after_sbs}sb "
            f"score={format_float(score)} qloss={format_float(metric.qloss)} "
            f"qacc={format_float(metric.qacc)} test_loss={format_float(metric.test_loss)} "
            f"elapsed={elapsed:.1f}s"
        ),
        "green",
    )

    if candidate.output_dir is not None and not args.keep_temp:
        remove_dir_quiet(candidate.output_dir)

    candidate.checkpoint = checkpoint
    candidate.output_dir = out_dir
    candidate.metric = metric
    candidate.score = score
    candidate.stage_sbs = stage.after_sbs
    candidate.transient_dirs.append(out_dir)
    return candidate


def rank_candidates(candidates: list[Candidate], lower_is_better: bool) -> list[Candidate]:
    def key(candidate: Candidate) -> float:
        if candidate.score is None:
            raise RuntimeError(f"candidate {candidate.index} has no score")
        return candidate.score

    return sorted(candidates, key=key, reverse=not lower_is_better)


def log_stage_rows(
    summary_path: Path,
    generation: int,
    stage: BeamStage,
    ranked: list[Candidate],
    keep: int,
) -> None:
    for rank, candidate in enumerate(ranked, start=1):
        metric = candidate.metric or Metric()
        append_csv(
            summary_path,
            SUMMARY_FIELDS,
            {
                "generation": generation,
                "stage_sbs": stage.after_sbs,
                "candidate": candidate.index,
                "status": "kept" if rank <= keep else "pruned",
                "rank": rank,
                "score": format_float(candidate.score),
                "quantized_value_loss": format_float(metric.qloss),
                "quantized_value_accuracy": format_float(metric.qacc),
                "test_value_loss": format_float(metric.test_loss),
                "test_value_accuracy": format_float(metric.test_acc),
                "checkpoint": str(candidate.checkpoint),
                "output_dir": str(candidate.output_dir or ""),
                "parameters_json": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
            },
        )


def validate_checkpoint_dir(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"checkpoint directory does not exist: {path}")
    if not (path / "state.bin").exists():
        raise RuntimeError(f"checkpoint directory has no state.bin: {path}")
    if not (path / "dataloader_pos.txt").exists():
        raise RuntimeError(f"checkpoint directory has no dataloader_pos.txt: {path}")


def load_state(path: Path) -> dict[str, Any]:
    state = load_json_object(path)
    version = int(state.get("version", STATE_VERSION))
    if version != STATE_VERSION:
        raise ValueError(f"{path}: unsupported state version {version}")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["version"] = STATE_VERSION
    atomic_write_json(path, state)


def metric_direction_text(settings: EsSettings) -> str:
    return "lower is better" if settings.lower_is_better else "higher is better"


def public_checkpoint_name(accepted_sbs: int) -> str:
    return f"sb{accepted_sbs:08d}"


def main() -> int:
    args = parse_args()
    color = color_enabled(args.color)
    root, specs, settings = load_parameters(args.parameters_file)
    if args.generations is not None:
        settings.generations = args.generations
    if args.save_rate is not None:
        settings.save_rate = args.save_rate
    if args.metric is not None:
        settings.metric = args.metric
        settings.lower_is_better = "loss" in args.metric

    if args.bucket_counts is None:
        nonzero_counts = [
            name for name in CONFIDENCE_FLAGS
            if abs(specs[name].current) > 0.0
        ]
        if nonzero_counts:
            raise RuntimeError(
                "--bucket-counts is required because count-confidence parameters are non-zero: "
                + ", ".join(nonzero_counts)
            )

    runner_root = args.output_folder / f"es-{args.tag_prefix}"
    temp_root = args.temp_folder / f"es-{args.tag_prefix}" if args.temp_folder else runner_root / "temp"
    log_dir = runner_root / "logs"
    accepted_root = runner_root / "accepted-checkpoints"
    current_dir = runner_root / "current"
    state_path = runner_root / "runner-state.json"
    summary_path = runner_root / "summary-learn.log"
    accepted_summary_path = runner_root / "accepted-summary-learn.log"
    history_path = runner_root / "parameters-history.jsonl"

    runner_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    accepted_root.mkdir(parents=True, exist_ok=True)
    ensure_csv(summary_path, SUMMARY_FIELDS)
    ensure_csv(accepted_summary_path, ACCEPTED_FIELDS)

    if args.resume:
        if not state_path.exists():
            raise RuntimeError(f"--resume was specified but {state_path} does not exist")
        state = load_state(state_path)
        current_checkpoint = Path(str(state["current_checkpoint"]))
        generation_start = int(state.get("generation", 0)) + 1
        accepted_sbs = int(state.get("accepted_sbs", 0))
        event(color, "[RESUME]", f"generation={generation_start} checkpoint={current_checkpoint}", "yellow")
    else:
        if state_path.exists():
            raise RuntimeError(f"{state_path} already exists; use --resume or choose a new --tag-prefix")
        assert args.base_checkpoint is not None
        validate_checkpoint_dir(args.base_checkpoint)
        copy_dir_replace(args.base_checkpoint, current_dir)
        current_checkpoint = current_dir
        generation_start = 1
        accepted_sbs = 0
        state = {
            "generation": 0,
            "accepted_sbs": 0,
            "current_checkpoint": str(current_checkpoint),
            "parameters_file": str(args.parameters_file.resolve()),
        }
        save_state(state_path, state)
        write_current_parameters(args.parameters_file, root, specs)
        event(color, "[START]", f"checkpoint={current_checkpoint}", "green")

    validate_checkpoint_dir(current_checkpoint)

    beam_text = ", ".join(f"{stage.after_sbs}sb=>keep{stage.keep}" for stage in settings.beam)
    event(
        color,
        "[CONFIG]",
        (
            f"population={settings.population} generations={settings.generations} "
            f"metric={settings.metric} ({metric_direction_text(settings)}) beam=[{beam_text}] "
            f"save_rate={settings.save_rate}"
        ),
        "cyan",
    )
    event(color, "[PARAMETERS]", json.dumps(current_values(specs), ensure_ascii=False, sort_keys=True), "cyan")

    total_generations = settings.generations
    for generation in range(generation_start, generation_start + total_generations):
        rng = random.Random(settings.seed + generation * 1_000_003)
        current_params = current_values(specs)
        candidates = [
            Candidate(index=i + 1, params=perturb_parameters(specs, rng), checkpoint=current_checkpoint)
            for i in range(settings.population)
        ]

        event(
            color,
            "[GEN START]",
            f"generation={generation} population={settings.population} from={current_checkpoint}",
            "magenta",
        )

        live = candidates
        prev_after_sbs = 0
        for stage in settings.beam:
            trained: list[Candidate] = []
            for candidate in live:
                trained.append(
                    train_candidate_stage(
                        args=args,
                        settings=settings,
                        generation=generation,
                        candidate=candidate,
                        stage=stage,
                        prev_after_sbs=prev_after_sbs,
                        temp_root=temp_root,
                        log_dir=log_dir,
                        color=color,
                    )
                )
            ranked = rank_candidates(trained, settings.lower_is_better)
            log_stage_rows(summary_path, generation, stage, ranked, stage.keep)
            kept = ranked[: stage.keep]
            pruned = ranked[stage.keep :]
            best = kept[0]
            worst_kept = kept[-1]
            event(
                color,
                "[BEAM]",
                (
                    f"generation={generation} stage={stage.after_sbs}sb "
                    f"keep={len(kept)}/{len(ranked)} best_score={format_float(best.score)} "
                    f"worst_kept={format_float(worst_kept.score)}"
                ),
                "yellow",
            )
            if not args.keep_temp:
                for candidate in pruned:
                    if candidate.output_dir is not None:
                        remove_dir_quiet(candidate.output_dir)
            live = kept
            prev_after_sbs = stage.after_sbs

        survivor = live[0]
        if survivor.metric is None or survivor.score is None:
            raise RuntimeError("final survivor has no metric")

        if not args.dry_run:
            copy_dir_replace(survivor.checkpoint, current_dir)
        current_checkpoint = current_dir
        accepted_sbs += settings.beam[-1].after_sbs

        set_current_values(specs, survivor.params)
        write_current_parameters(args.parameters_file, root, specs)

        saved_checkpoint = ""
        if settings.save_rate > 0 and (generation % settings.save_rate == 0):
            public_dir = accepted_root / public_checkpoint_name(accepted_sbs)
            if not args.dry_run:
                copy_dir_new(current_dir, public_dir)
            saved_checkpoint = public_dir.name

        params_json = json.dumps(survivor.params, ensure_ascii=False, sort_keys=True)
        metric = survivor.metric
        append_csv(
            accepted_summary_path,
            ACCEPTED_FIELDS,
            {
                "generation": generation,
                "accepted_sbs": accepted_sbs,
                "score": format_float(survivor.score),
                "quantized_value_loss": format_float(metric.qloss),
                "quantized_value_accuracy": format_float(metric.qacc),
                "test_value_loss": format_float(metric.test_loss),
                "test_value_accuracy": format_float(metric.test_acc),
                "stage_sbs": settings.beam[-1].after_sbs,
                "saved_checkpoint": saved_checkpoint,
                "current_checkpoint": str(current_checkpoint),
                "parameters_json": params_json,
            },
        )
        append_jsonl(
            history_path,
            {
                "generation": generation,
                "accepted_sbs": accepted_sbs,
                "score": survivor.score,
                "metric": {
                    "quantized_value_loss": metric.qloss,
                    "quantized_value_accuracy": metric.qacc,
                    "test_value_loss": metric.test_loss,
                    "test_value_accuracy": metric.test_acc,
                },
                "parameters": survivor.params,
                "current_checkpoint": str(current_checkpoint),
                "saved_checkpoint": saved_checkpoint,
            },
        )

        state = {
            "generation": generation,
            "accepted_sbs": accepted_sbs,
            "current_checkpoint": str(current_checkpoint),
            "last_score": survivor.score,
            "last_metric": {
                "quantized_value_loss": metric.qloss,
                "quantized_value_accuracy": metric.qacc,
                "test_value_loss": metric.test_loss,
                "test_value_accuracy": metric.test_acc,
            },
            "parameters_file": str(args.parameters_file.resolve()),
        }
        save_state(state_path, state)

        event(
            color,
            "[ACCEPT]",
            (
                f"generation={generation} accepted_sbs={accepted_sbs} "
                f"score={format_float(survivor.score)} qloss={format_float(metric.qloss)} "
                f"qacc={format_float(metric.qacc)}"
            ),
            "green",
        )
        if saved_checkpoint:
            event(color, "[SAVE]", f"{accepted_root / saved_checkpoint}", "green")
            event(color, "[SAFE TO STOP]", f"saved={accepted_root / saved_checkpoint}", "green")
        else:
            event(color, "[CURRENT]", f"resume checkpoint updated: {current_checkpoint}", "yellow")

        if not args.keep_temp:
            gen_temp = temp_root / f"gen{generation:04d}"
            remove_dir_quiet(gen_temp)

        event(
            color,
            "[GEN END]",
            f"generation={generation} survivor=cand{survivor.index:03d} params={params_json}",
            "magenta",
        )

    event(color, "[DONE]", f"current_checkpoint={current_checkpoint}", "green")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
