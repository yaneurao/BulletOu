#!/usr/bin/env python3
"""Beam-style ES runner for BulletOu local fine-tuning.

This runner is intentionally simple:

* `es-settings.json` owns the current hyperparameters and ES settings.
* `bulletou-settings.json` owns the ordinary `bulletou.exe` training options.
* Each generation creates a population of randomized candidates.
* Candidates are trained for the configured beam stages.
* At each stage, candidates are ranked by the configured metric and pruned.
* The final survivor's NN weights and hyperparameters become the next current
  state.

There is no gradient estimate and no partial parameter update.  The selected
candidate itself survives.

Typical use:

    python es_local_runner.py ^
      --es-settings-file .\\es-settings.json

Use --resume to continue from the runner root described by es-settings.json.
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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ES_SETTINGS_VERSION = 1
STATE_VERSION = 1


BORDA_COUNT_METRIC = "borda_count"
BORDA_COMPONENTS: tuple[tuple[str, bool], ...] = (
    ("test_value_accuracy", False),
    ("test_value_loss", True),
    ("quantized_value_accuracy", False),
    ("quantized_value_loss", True),
)


ALPHA_PARAMETERS = {
    "shared": "shared",
    "king_axis": "king_axis",
    "hand_axis": "hand_axis",
    "progress_axis": "progress_axis",
    "king_hand_pair": "king_hand_pair",
    "king_progress_pair": "king_progress_pair",
    "hand_progress_pair": "hand_progress_pair",
}


CONFIDENCE_FLAGS = {
    "residual_count": "--sfnn-residual-count-gate-confidence",
    "king_axis_count": "--sfnn-king-axis-count-confidence",
    "hand_axis_count": "--sfnn-hand-axis-count-confidence",
    "progress_axis_count": "--sfnn-progress-axis-count-confidence",
    "king_hand_pair_count": "--sfnn-king-hand-pair-count-confidence",
    "king_progress_pair_count": "--sfnn-king-progress-pair-count-confidence",
    "hand_progress_pair_count": "--sfnn-hand-progress-pair-count-confidence",
}


KNOWN_PARAMETERS = set(ALPHA_PARAMETERS) | set(CONFIDENCE_FLAGS)


DEFAULT_PARAMETER_SPECS: dict[str, dict[str, float | bool]] = {
    "shared": {"current": 1.0, "tune": False, "step": 0.0, "min": 0.0, "max": 100.0},
    "king_axis": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "hand_axis": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "progress_axis": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "king_hand_pair": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "king_progress_pair": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "hand_progress_pair": {"current": 1.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "residual_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "king_axis_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "hand_axis_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "progress_axis_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "king_hand_pair_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "king_progress_pair_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
    "hand_progress_pair_count": {"current": 0.0, "tune": False, "step": 0.005, "min": 0.0, "max": 100.0},
}


RUNNER_CONTROLLED_BULLETOU_SETTINGS = {
    "initial_state",
    "initial-state",
    "initial_dataloader_pos",
    "initial-dataloader-pos",
    "output",
    "output_folder",
    "output-folder",
    "tag",
    "resume",
    "no_resume",
    "no-resume",
    "superbatches",
    "max_epochs",
    "max-epochs",
    "save_rate",
    "save-rate",
    "validation_rate",
    "validation-rate",
    "quantized_validation_rate",
    "quantized-validation-rate",
    "sfnn_factorizer_alpha",
    "sfnn-factorizer-alpha",
    "cuda_cpp_skip_final_output",
    "cuda-cpp-skip-final-output",
    *{flag.removeprefix("--").replace("-", "_") for flag in CONFIDENCE_FLAGS.values()},
    *{flag.removeprefix("--") for flag in CONFIDENCE_FLAGS.values()},
}


SUMMARY_FIELDS = [
    "generation",
    "stage_sbs",
    "candidate",
    "status",
    "rank",
    "test_value_accuracy",
    "test_value_loss",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "checkpoint",
    "output_dir",
    "parameters",
]


ACCEPTED_FIELDS = [
    "generation",
    "accepted_sbs",
    "test_value_accuracy",
    "test_value_loss",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "stage_sbs",
    "saved_checkpoint",
    "current_checkpoint",
    "parameters",
]


LOG_WRITES_ENABLED = True


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
        elif metric_name == BORDA_COUNT_METRIC:
            raise ValueError("borda_count is computed across candidates, not from one metric value")
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
    enabled: bool
    use_worker: bool
    generations: int
    population: int
    beam: list[BeamStage]
    metric: str
    lower_is_better: bool
    seed: int
    save_rate: int
    validation_rate: int
    quantized_validation_rate: int


@dataclass
class RunSettings:
    exe: Path
    bulletou_settings_file: Path
    base_checkpoint: Path | None
    output_folder: Path | None
    temp_folder: Path | None
    tag_prefix: str | None


@dataclass
class Candidate:
    index: int
    params: dict[str, float]
    checkpoint: Path
    cache_key: str | None = None
    output_dir: Path | None = None
    metric: Metric | None = None
    score: float | None = None
    stage_sbs: int = 0
    transient_dirs: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a beam-style ES search for BulletOu factorizer/count hyperparameters."
    )
    parser.add_argument("--es-settings-file", type=Path, default=Path("es-settings.json"))
    parser.add_argument("--resume", action="store_true", help="Resume from run.output_folder/es-<run.tag_prefix>/runner-state.json")
    parser.add_argument("--keep-temp", action="store_true", help="Keep candidate temp directories for debugging")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-stream-child-output",
        action="store_true",
        help="Do not mirror bulletou.exe stdout to console; logs are still written under runner logs/.",
    )
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--debug", action="store_true", help="Print Python traceback for runner errors")
    args = parser.parse_args()
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
        raise ValueError(f"parameters.{name}: tune=true requires multiplicative step > 0")
    if spec.tune and spec.current <= 0.0:
        raise ValueError(f"parameters.{name}: multiplicative ES requires current > 0")
    return spec


def es_settings_path(base_file: Path, raw: Any, name: str, required: bool = True) -> Path | None:
    if raw is None:
        if required:
            raise ValueError(f"{base_file}: run.{name} is required")
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{base_file}: run.{name} must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = base_file.parent / path
    return path


def validate_bulletou_settings_for_es(path: Path) -> None:
    root = load_json_object(path)
    bad = sorted(set(root) & RUNNER_CONTROLLED_BULLETOU_SETTINGS)
    if bad:
        raise ValueError(
            f"{path}: these BulletOu settings are controlled by es_local_runner.py and must not be written here: "
            + ", ".join(bad)
        )


def load_parameters(path: Path) -> tuple[dict[str, Any], dict[str, ParameterSpec], EsSettings, RunSettings]:
    root = load_json_object(path)
    version = int(root.get("version", ES_SETTINGS_VERSION))
    if version != ES_SETTINGS_VERSION:
        raise ValueError(f"{path}: unsupported version {version}; expected {ES_SETTINGS_VERSION}")

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
    enabled = es_obj.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"{path}: es.enabled must be true or false")
    use_worker = bool(es_obj.get("use_worker", True))

    generations = int(es_obj.get("generations", 1))
    population = int(es_obj.get("population", 4))
    metric = str(es_obj.get("metric", "quantized_value_loss"))
    if metric not in {
        "quantized_value_loss",
        "quantized_value_accuracy",
        "test_value_loss",
        "test_value_accuracy",
        BORDA_COUNT_METRIC,
    }:
        raise ValueError(f"{path}: unsupported es.metric {metric!r}")
    lower_is_better = True if metric == BORDA_COUNT_METRIC else bool(es_obj.get("lower_is_better", "loss" in metric))
    seed = int(es_obj.get("seed", 1))
    save_rate = int(es_obj.get("save_rate", 1))
    if "candidate_validation_rate" in es_obj or "candidate_quantized_validation_rate" in es_obj:
        raise ValueError(
            f"{path}: es.candidate_validation_rate / es.candidate_quantized_validation_rate were renamed to "
            "es.validation_rate / es.quantized_validation_rate"
        )
    validation_rate = int(es_obj.get("validation_rate", 1))
    quantized_validation_rate = int(es_obj.get("quantized_validation_rate", 1))

    if generations <= 0:
        raise ValueError("es.generations must be > 0")
    if population <= 0:
        raise ValueError("es.population must be > 0")
    if save_rate < 0:
        raise ValueError("es.save_rate must be >= 0")
    if validation_rate < -1:
        raise ValueError("es.validation_rate must be >= -1; -1 disables f32 validation")
    if quantized_validation_rate < -1:
        raise ValueError(
            "es.quantized_validation_rate must be >= -1; -1 disables quantized validation"
        )
    metric_needs_validation = metric in {"test_value_loss", "test_value_accuracy", BORDA_COUNT_METRIC}
    metric_needs_quantized_validation = metric in {
        "quantized_value_loss",
        "quantized_value_accuracy",
        BORDA_COUNT_METRIC,
    }
    if metric_needs_validation and validation_rate < 0:
        raise ValueError(
            f"es.metric {metric!r} requires f32 validation; "
            "use 0 for final-stage-only validation"
        )
    if metric_needs_quantized_validation and quantized_validation_rate < 0:
        raise ValueError(
            f"es.metric {metric!r} requires quantized validation; "
            "use 0 for final-stage-only quantized validation"
        )

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
        enabled=enabled,
        use_worker=use_worker,
        generations=generations,
        population=population,
        beam=beam,
        metric=metric,
        lower_is_better=lower_is_better,
        seed=seed,
        save_rate=save_rate,
        validation_rate=validation_rate,
        quantized_validation_rate=quantized_validation_rate,
    )
    run_obj = root.get("run")
    if not isinstance(run_obj, dict):
        raise ValueError(f"{path}: `run` object is required")
    unknown_run = sorted(set(run_obj) - {"exe", "bulletou_settings_file", "base_checkpoint", "output_folder", "temp_folder", "tag_prefix"})
    if unknown_run:
        raise ValueError(f"{path}: unknown run field(s): {', '.join(unknown_run)}")
    exe = es_settings_path(path, run_obj.get("exe"), "exe")
    bulletou_settings_file = es_settings_path(path, run_obj.get("bulletou_settings_file"), "bulletou_settings_file")
    base_checkpoint = es_settings_path(path, run_obj.get("base_checkpoint"), "base_checkpoint", required=False)
    output_folder = es_settings_path(path, run_obj.get("output_folder"), "output_folder", required=enabled)
    temp_folder = es_settings_path(path, run_obj.get("temp_folder"), "temp_folder", required=False)
    tag_prefix = run_obj.get("tag_prefix")
    if enabled and (not isinstance(tag_prefix, str) or not tag_prefix.strip()):
        raise ValueError(f"{path}: run.tag_prefix must be a non-empty string")
    if not isinstance(tag_prefix, str) or not tag_prefix.strip():
        tag_prefix = None
    assert exe is not None
    assert bulletou_settings_file is not None
    if enabled:
        assert output_folder is not None
        validate_bulletou_settings_for_es(bulletou_settings_file)
    run = RunSettings(
        exe=exe,
        bulletou_settings_file=bulletou_settings_file,
        base_checkpoint=base_checkpoint,
        output_folder=output_folder,
        temp_folder=temp_folder,
        tag_prefix=tag_prefix,
    )
    return root, specs, settings, run


def write_current_parameters(path: Path, root: dict[str, Any], specs: dict[str, ParameterSpec]) -> None:
    params_obj: dict[str, Any] = {}
    original_params = root.get("parameters", {})
    if not isinstance(original_params, dict):
        original_params = {}
    for name in sorted(KNOWN_PARAMETERS):
        spec = specs[name]
        defaults = DEFAULT_PARAMETER_SPECS[name]
        was_explicit = name in original_params
        is_default_inactive = (
            not spec.tune
            and spec.current == float(defaults["current"])
            and spec.step == float(defaults["step"])
            and spec.minimum == float(defaults["min"])
            and spec.maximum == float(defaults["max"])
        )
        if not was_explicit and is_default_inactive:
            continue

        original = original_params.get(name, {})
        if isinstance(original, dict):
            obj = dict(original)
        else:
            obj = {}
        obj["current"] = spec.current
        obj["tune"] = spec.tune
        obj["step"] = spec.step
        obj["min"] = spec.minimum
        obj["max"] = spec.maximum
        params_obj[name] = obj
    root["parameters"] = params_obj
    root["version"] = ES_SETTINGS_VERSION
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
            out[name] = spec.clamp(spec.current * math.exp(delta))
    return out


def alpha_arg(params: dict[str, float]) -> str:
    parts = []
    for name, cli_name in ALPHA_PARAMETERS.items():
        if name in params:
            parts.append(f"{cli_name}={params[name]:.9g}")
    return ",".join(parts)


def parameter_args(params: dict[str, float]) -> list[str]:
    out = ["--sfnn-factorizer-alpha", alpha_arg(params)]
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


def metric_status_text(metric: Metric, *, prefix: str = "") -> str:
    return (
        f"{prefix}acc={format_float(metric.test_acc)} "
        f"{prefix}loss={format_float(metric.test_loss)} "
        f"{prefix}qacc={format_float(metric.qacc)} "
        f"{prefix}qloss={format_float(metric.qloss)}"
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


def latest_learn_log_metric(checkpoint: Path) -> Metric:
    path = checkpoint / "learn.log"
    if not path.exists():
        return Metric()
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [row for row in rows if any((value or "").strip() for value in row.values())]
    if not rows:
        return Metric()
    return metric_from_summary_row(rows[-1])


def latest_checkpoint_dir(output_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / "state.bin").exists():
            candidates.append((int(child.name), child))
    if not candidates:
        raise RuntimeError(f"no checkpoint directory containing state.bin found under {output_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def maybe_latest_checkpoint_dir(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    try:
        return latest_checkpoint_dir(output_dir)
    except RuntimeError:
        return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if any((value or "").strip() for value in row.values())]


def count_summary_rows_with_status(path: Path, status: str) -> int:
    return sum(1 for row in read_csv_rows(path) if row.get("status") == status)


def normal_summary_checkpoint_path(output_dir: Path, checkpoint_cell: str | None) -> str:
    if checkpoint_cell is None:
        return ""
    checkpoint_cell = checkpoint_cell.strip()
    if not checkpoint_cell or checkpoint_cell == "-":
        return ""
    path = Path(checkpoint_cell)
    if not path.is_absolute():
        path = output_dir / checkpoint_cell
    return str(path)


def parse_int_cell(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    value = value.strip()
    if not value or value == "-":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def import_ordinary_run_summaries(
    *,
    summary_path: Path,
    accepted_summary_path: Path,
    ordinary_output: Path,
    settings: EsSettings,
    params: dict[str, float],
    color: bool,
) -> int:
    source_summary = ordinary_output / "summary-learn.log"
    rows = read_csv_rows(source_summary)
    if not rows:
        event(color, "[LOG IMPORT]", f"no summary rows found: {source_summary}", "yellow")
        return 0

    ensure_csv(summary_path, SUMMARY_FIELDS)
    ensure_csv(accepted_summary_path, ACCEPTED_FIELDS)
    imported_rows = count_summary_rows_with_status(summary_path, "ordinary")
    if imported_rows >= len(rows):
        event(color, "[LOG IMPORT]", f"already up to date: imported_rows={imported_rows}", "yellow")
        return 0

    params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
    added = 0
    epoch_sbs = max(1, settings.beam[-1].after_sbs)
    for row in rows[imported_rows:]:
        epoch = parse_int_cell(row.get("epoch"), default=1)
        sb = parse_int_cell(row.get("superbatch"), default=0)
        accepted_sbs = (max(1, epoch) - 1) * epoch_sbs + sb
        metric = metric_from_summary_row(row)
        checkpoint_path = normal_summary_checkpoint_path(ordinary_output, row.get("checkpoint"))
        saved_checkpoint = (
            public_checkpoint_name(accepted_sbs)
            if checkpoint_path and settings.save_rate > 0 and max(1, epoch) % settings.save_rate == 0
            else ""
        )
        append_csv(
            summary_path,
            SUMMARY_FIELDS,
            {
                "generation": epoch,
                "stage_sbs": sb,
                "candidate": 1,
                "status": "ordinary",
                "rank": 1,
                "test_value_accuracy": format_float(metric.test_acc),
                "test_value_loss": format_float(metric.test_loss),
                "quantized_value_accuracy": format_float(metric.qacc),
                "quantized_value_loss": format_float(metric.qloss),
                "checkpoint": checkpoint_path,
                "output_dir": str(ordinary_output),
                "parameters": params_json,
            },
        )
        append_csv(
            accepted_summary_path,
            ACCEPTED_FIELDS,
            {
                "generation": epoch,
                "accepted_sbs": accepted_sbs,
                "test_value_accuracy": format_float(metric.test_acc),
                "test_value_loss": format_float(metric.test_loss),
                "quantized_value_accuracy": format_float(metric.qacc),
                "quantized_value_loss": format_float(metric.qloss),
                "stage_sbs": sb,
                "saved_checkpoint": saved_checkpoint,
                "current_checkpoint": checkpoint_path,
                "parameters": params_json,
            },
        )
        added += 1

    event(color, "[LOG IMPORT]", f"imported_rows={added} source={source_summary}", "green")
    return added


def sync_ordinary_accepted_checkpoints(
    *,
    accepted_root: Path,
    current_dir: Path,
    ordinary_output: Path,
    settings: EsSettings,
    es_settings_file: Path,
    bulletou_settings_file: Path,
    color: bool,
) -> int:
    rows = read_csv_rows(ordinary_output / "summary-learn.log")
    if not rows:
        return 0

    accepted_root.mkdir(parents=True, exist_ok=True)
    epoch_sbs = max(1, settings.beam[-1].after_sbs)
    copied = 0
    latest_checkpoint: Path | None = None
    for row in rows:
        checkpoint_path_text = normal_summary_checkpoint_path(ordinary_output, row.get("checkpoint"))
        if not checkpoint_path_text:
            continue
        checkpoint = Path(checkpoint_path_text)
        if not (checkpoint / "state.bin").exists():
            continue
        latest_checkpoint = checkpoint
        epoch = parse_int_cell(row.get("epoch"), default=1)
        sb = parse_int_cell(row.get("superbatch"), default=0)
        accepted_sbs = (max(1, epoch) - 1) * epoch_sbs + sb
        if settings.save_rate <= 0 or max(1, epoch) % settings.save_rate != 0:
            continue
        public_dir = accepted_root / public_checkpoint_name(accepted_sbs)
        if public_dir.exists():
            if not (public_dir / "state.bin").exists():
                raise RuntimeError(f"accepted checkpoint exists but has no state.bin: {public_dir}")
            continue
        copy_dir_new(checkpoint, public_dir)
        copy_settings_files(public_dir, es_settings_file, bulletou_settings_file)
        copied += 1

    if latest_checkpoint is not None:
        copy_dir_replace(latest_checkpoint, current_dir)
        copy_settings_files(current_dir, es_settings_file, bulletou_settings_file)

    event(
        color,
        "[CHECKPOINT SYNC]",
        f"copied={copied} accepted_root={accepted_root} current={current_dir}",
        "green" if copied else "yellow",
    )
    return copied


def ensure_csv(path: Path, fields: list[str]) -> None:
    if not LOG_WRITES_ENABLED:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as f:
            header = f.readline().strip()
        expected = ",".join(fields)
        if header != expected:
            existing_fields = next(csv.reader([header]))
            if set(existing_fields) != set(fields):
                raise RuntimeError(f"{path} has incompatible header\n  existing: {header}\n  expected: {expected}")
            rewrite_csv_with_fields(path, fields)
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()


def rewrite_csv_with_fields(path: Path, fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp, path)


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    if not LOG_WRITES_ENABLED:
        return
    ensure_csv(path, fields)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writerow({field: row.get(field, "") for field in fields})


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    if not LOG_WRITES_ENABLED:
        return
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


def copy_settings_files(dst: Path, es_settings_file: Path, bulletou_settings_file: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(es_settings_file, dst / "es-settings.json")
    shutil.copy2(bulletou_settings_file, dst / "bulletou-settings.json")


def run_command(cmd: list[str], log_path: Path, stream: bool, stream_prefix: str = "") -> tuple[int, float]:
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
                if stream_prefix and line.strip():
                    print(f"{stream_prefix}{line}", end="")
                else:
                    print(line, end="")
        code = proc.wait()
    return code, time.perf_counter() - start


class WorkerClient:
    def __init__(self, exe: Path, log_path: Path, stream: bool, color: bool):
        self.exe = exe
        self.log_path = log_path
        self.stream = stream
        self.color = color
        self.request_id = 0
        self._prefix = paint(color, "[WORKER] ", "magenta")
        self._prefix_lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8", errors="replace", newline="")
        self.proc = subprocess.Popen(
            [str(exe), "worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        self._stderr_thread = threading.Thread(target=self._pump_stderr, daemon=True)
        self._stderr_thread.start()

    def _pump_stderr(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self._log.write(line)
            self._log.flush()
            if self.stream:
                with self._prefix_lock:
                    prefix = self._prefix
                if line.strip():
                    print(f"{prefix}{line}", end="")
                else:
                    print(line, end="")

    def set_prefix(self, prefix: str) -> None:
        with self._prefix_lock:
            self._prefix = prefix

    def request(self, cmd: str, payload: dict[str, Any] | None = None, prefix: str | None = None) -> dict[str, Any]:
        if self.proc.poll() is not None:
            raise RuntimeError(f"bulletou worker already exited with code {self.proc.returncode}; see {self.log_path}")
        self.request_id += 1
        request = dict(payload or {})
        request["id"] = self.request_id
        request["cmd"] = cmd
        if prefix is not None:
            self.set_prefix(prefix)
        line = json.dumps(request, ensure_ascii=False)
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        while True:
            response_line = self.proc.stdout.readline()
            if response_line == "":
                code = self.proc.poll()
                raise RuntimeError(f"bulletou worker closed stdout while waiting for `{cmd}` response; code={code}; see {self.log_path}")
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"bulletou worker returned non-JSON response: {response_line.strip()!r}: {exc}") from exc
            if response.get("id") != self.request_id:
                raise RuntimeError(f"bulletou worker response id mismatch: got {response.get('id')}, expected {self.request_id}")
            if not response.get("ok", False):
                raise RuntimeError(f"bulletou worker `{cmd}` failed: {response.get('error')}; see {self.log_path}")
            payload = response.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError(f"bulletou worker `{cmd}` returned non-object payload")
            return payload

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self.request("quit", prefix=paint(self.color, "[WORKER] ", "magenta"))
                except Exception:
                    self.proc.terminate()
        finally:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self._log.close()


def metric_from_worker_payload(payload: dict[str, Any]) -> Metric:
    def as_float(name: str) -> float | None:
        value = payload.get(name)
        if value is None:
            return None
        return float(value)

    return Metric(
        qloss=as_float("quantized_value_loss"),
        qacc=as_float("quantized_value_accuracy"),
        test_loss=as_float("test_value_loss"),
        test_acc=as_float("test_value_accuracy"),
    )


def metric_to_borda_json(metric: Metric) -> dict[str, float]:
    return {
        "test_value_accuracy": metric.score("test_value_accuracy"),
        "test_value_loss": metric.score("test_value_loss"),
        "quantized_value_accuracy": metric.score("quantized_value_accuracy"),
        "quantized_value_loss": metric.score("quantized_value_loss"),
    }


def metric_pareto_dominates(a: Metric, b: Metric) -> bool:
    a_qloss = a.score("quantized_value_loss")
    a_qacc = a.score("quantized_value_accuracy")
    a_test_loss = a.score("test_value_loss")
    a_test_acc = a.score("test_value_accuracy")
    b_qloss = b.score("quantized_value_loss")
    b_qacc = b.score("quantized_value_accuracy")
    b_test_loss = b.score("test_value_loss")
    b_test_acc = b.score("test_value_accuracy")
    return (
        a_qloss <= b_qloss
        and a_qacc >= b_qacc
        and a_test_loss <= b_test_loss
        and a_test_acc >= b_test_acc
        and (a_qloss < b_qloss or a_qacc > b_qacc or a_test_loss < b_test_loss or a_test_acc > b_test_acc)
    )


def build_train_args(
    run: RunSettings,
    params: dict[str, float],
    checkpoint: Path | None,
    output_dir: Path,
    stage_delta_sbs: int,
    settings: EsSettings,
    *,
    include_initial_state: bool,
) -> list[str]:
    validation_rate = (
        0
        if settings.validation_rate < 0
        else stage_delta_sbs
        if settings.validation_rate == 0
        else max(1, min(settings.validation_rate, stage_delta_sbs))
    )
    quantized_validation_rate = (
        0
        if settings.quantized_validation_rate < 0
        else stage_delta_sbs
        if settings.quantized_validation_rate == 0
        else max(1, min(settings.quantized_validation_rate, stage_delta_sbs))
    )
    cmd = [
        "--settings-file",
        str(run.bulletou_settings_file),
        "--output",
        str(output_dir),
        "--superbatches",
        str(stage_delta_sbs),
        "--max-epochs",
        "1",
        "--save-rate",
        str(stage_delta_sbs),
        "--validation-rate",
        str(validation_rate),
        "--quantized-validation-rate",
        str(quantized_validation_rate),
    ]
    if include_initial_state:
        if checkpoint is None:
            raise RuntimeError("include_initial_state requires a checkpoint")
        cmd[2:2] = [
            "--initial-state",
            str(checkpoint / "state.bin"),
            "--initial-dataloader-pos",
            str(checkpoint / "dataloader_pos.txt"),
        ]
    cmd.extend(parameter_args(params))
    return cmd


def build_train_command(
    run: RunSettings,
    params: dict[str, float],
    checkpoint: Path,
    output_dir: Path,
    stage_delta_sbs: int,
    settings: EsSettings,
) -> list[str]:
    return [
        str(run.exe),
        *build_train_args(
            run,
            params,
            checkpoint,
            output_dir,
            stage_delta_sbs,
            settings,
            include_initial_state=True,
        ),
    ]


def train_candidate_stage(
    args: argparse.Namespace,
    run: RunSettings,
    settings: EsSettings,
    summary_path: Path,
    base_metric: Metric,
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
    if not args.dry_run:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"gen{generation:04d}-stage{stage.after_sbs:04d}-cand{candidate.index:03d}.stdout.log"
    event(
        color,
        f"[CAND {candidate.index:03d} START]",
        (
            f"generation={generation} stage={stage.after_sbs}sb delta={delta}sb "
            f"{metric_status_text(base_metric, prefix='base_')}"
        ),
        "cyan",
    )
    append_csv(
        summary_path,
        SUMMARY_FIELDS,
        {
            "generation": generation,
            "stage_sbs": stage.after_sbs,
            "candidate": candidate.index,
            "status": "started",
            "rank": "",
            "quantized_value_loss": "",
            "quantized_value_accuracy": "",
            "test_value_loss": "",
            "test_value_accuracy": "",
            "checkpoint": str(candidate.checkpoint),
            "output_dir": str(out_dir),
            "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        },
    )
    cmd = build_train_command(run, candidate.params, candidate.checkpoint, out_dir, delta, settings)
    if args.dry_run:
        print("  " + subprocess.list2cmdline(cmd), flush=True)
        metric = Metric(qloss=math.inf, qacc=-math.inf, test_loss=math.inf, test_acc=-math.inf)
        score = None if settings.metric == BORDA_COUNT_METRIC else metric.score(settings.metric)
        checkpoint = candidate.checkpoint
        elapsed = 0.0
    else:
        child_prefix = paint(color, f"[G{generation:04d} S{stage.after_sbs:04d} C{candidate.index:03d}] ", "magenta")
        code, elapsed = run_command(
            cmd,
            log_path,
            stream=not args.no_stream_child_output,
            stream_prefix=child_prefix,
        )
        if code != 0:
            append_csv(
                summary_path,
                SUMMARY_FIELDS,
                {
                    "generation": generation,
                    "stage_sbs": stage.after_sbs,
                    "candidate": candidate.index,
                    "status": "failed",
                    "rank": "",
                    "quantized_value_loss": "",
                    "quantized_value_accuracy": "",
                    "test_value_loss": "",
                    "test_value_accuracy": "",
                    "checkpoint": str(candidate.checkpoint),
                    "output_dir": str(out_dir),
                    "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
                },
            )
            raise RuntimeError(f"candidate {candidate.index} failed at stage {stage.after_sbs}sb; see {log_path}")
        row = latest_summary_row(out_dir)
        metric = metric_from_summary_row(row)
        score = None if settings.metric == BORDA_COUNT_METRIC else metric.score(settings.metric)
        checkpoint = latest_checkpoint_dir(out_dir)

    score_text = "pending" if score is None else format_float(score)
    event(
        color,
        f"[CAND {candidate.index:03d} END]",
        (
            f"generation={generation} stage={stage.after_sbs}sb "
            f"{metric_status_text(metric)} {settings.metric}={score_text} "
            f"elapsed={elapsed:.1f}s"
        ),
        "green",
    )
    append_csv(
        summary_path,
        SUMMARY_FIELDS,
        {
            "generation": generation,
            "stage_sbs": stage.after_sbs,
            "candidate": candidate.index,
            "status": "finished",
            "rank": "",
            "quantized_value_loss": format_float(metric.qloss),
            "quantized_value_accuracy": format_float(metric.qacc),
            "test_value_loss": format_float(metric.test_loss),
            "test_value_accuracy": format_float(metric.test_acc),
            "checkpoint": str(checkpoint),
            "output_dir": str(out_dir),
            "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        },
    )

    if candidate.output_dir is not None and not args.keep_temp and not args.dry_run:
        remove_dir_quiet(candidate.output_dir)

    candidate.checkpoint = checkpoint
    candidate.output_dir = out_dir
    candidate.metric = metric
    candidate.score = score
    candidate.stage_sbs = stage.after_sbs
    candidate.transient_dirs.append(out_dir)
    return candidate


def train_candidate_stage_worker(
    args: argparse.Namespace,
    worker: WorkerClient | None,
    run: RunSettings,
    settings: EsSettings,
    summary_path: Path,
    base_metric: Metric,
    generation: int,
    candidate: Candidate,
    stage: BeamStage,
    prev_after_sbs: int,
    temp_root: Path,
    borda_dominators: list[Metric] | None,
    color: bool,
) -> Candidate:
    delta = stage.after_sbs - prev_after_sbs
    out_dir = temp_root / f"gen{generation:04d}" / f"stage{stage.after_sbs:04d}" / f"cand{candidate.index:03d}"
    cache_key = f"g{generation:04d}-s{stage.after_sbs:04d}-c{candidate.index:03d}"
    disk_cache = settings.metric == BORDA_COUNT_METRIC
    checkpoint_text = f"worker-disk-cache:{cache_key}" if disk_cache else f"worker-cache:{cache_key}"
    event(
        color,
        f"[CAND {candidate.index:03d} START]",
        (
            f"generation={generation} stage={stage.after_sbs}sb delta={delta}sb worker=on "
            f"{metric_status_text(base_metric, prefix='base_')}"
        ),
        "cyan",
    )
    append_csv(
        summary_path,
        SUMMARY_FIELDS,
        {
            "generation": generation,
            "stage_sbs": stage.after_sbs,
            "candidate": candidate.index,
            "status": "started",
            "rank": "",
            "quantized_value_loss": "",
            "quantized_value_accuracy": "",
            "test_value_loss": "",
            "test_value_accuracy": "",
            "checkpoint": str(candidate.checkpoint),
            "output_dir": checkpoint_text,
            "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        },
    )
    trial_args = build_train_args(
        run,
        candidate.params,
        None,
        out_dir,
        delta,
        settings,
        include_initial_state=False,
    )
    if args.dry_run:
        trial_request = {"args": trial_args, "keep": False, "cache_key": cache_key}
        if disk_cache:
            trial_request["cache_dir"] = str(out_dir)
        if borda_dominators:
            trial_request["borda_dominators"] = [metric_to_borda_json(metric) for metric in borda_dominators]
        print("  worker trial " + json.dumps(trial_request, ensure_ascii=False), flush=True)
        metric = Metric(qloss=math.inf, qacc=-math.inf, test_loss=math.inf, test_acc=-math.inf)
        score = None if settings.metric == BORDA_COUNT_METRIC else metric.score(settings.metric)
        elapsed = 0.0
        cached = True
        dominated_by_prior = False
    else:
        if worker is None:
            raise RuntimeError("worker client is not open")
        prefix = paint(color, f"[G{generation:04d} S{stage.after_sbs:04d} C{candidate.index:03d}] ", "magenta")
        started = time.perf_counter()
        trial_request = {"args": trial_args, "keep": False, "cache_key": cache_key}
        if disk_cache:
            trial_request["cache_dir"] = str(out_dir)
        if borda_dominators:
            trial_request["borda_dominators"] = [metric_to_borda_json(metric) for metric in borda_dominators]
        payload = worker.request(
            "trial",
            trial_request,
            prefix=prefix,
        )
        elapsed = time.perf_counter() - started
        metric = metric_from_worker_payload(payload)
        score = None if settings.metric == BORDA_COUNT_METRIC else metric.score(settings.metric)
        cached = bool(payload.get("cached", True))
        dominated_by_prior = bool(payload.get("dominated_by_prior", False))

    score_text = "pending" if score is None else format_float(score)
    cache_text = "skipped_dominated" if dominated_by_prior else ("disk" if disk_cache else "host")
    event(
        color,
        f"[CAND {candidate.index:03d} END]",
        (
            f"generation={generation} stage={stage.after_sbs}sb worker=on "
            f"{metric_status_text(metric)} {settings.metric}={score_text} "
            f"cache={cache_text if cached or dominated_by_prior else 'none'} elapsed={elapsed:.1f}s"
        ),
        "yellow" if dominated_by_prior else "green",
    )
    append_csv(
        summary_path,
        SUMMARY_FIELDS,
        {
            "generation": generation,
            "stage_sbs": stage.after_sbs,
            "candidate": candidate.index,
            "status": "finished",
            "rank": "",
            "quantized_value_loss": format_float(metric.qloss),
            "quantized_value_accuracy": format_float(metric.qacc),
            "test_value_loss": format_float(metric.test_loss),
            "test_value_accuracy": format_float(metric.test_acc),
            "checkpoint": checkpoint_text if cached else "worker-dominated:no-cache",
            "output_dir": checkpoint_text if cached else "worker-dominated:no-cache",
            "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        },
    )

    candidate.cache_key = cache_key if cached else None
    candidate.output_dir = None
    candidate.metric = metric
    candidate.score = score
    candidate.stage_sbs = stage.after_sbs
    return candidate


def assign_borda_count_scores(candidates: list[Candidate]) -> None:
    """Assign Borda rank-sum scores to candidates.

    Lower is better.  Each component contributes rank 1 for the best candidate,
    rank 2 for the next, and so on.  Ties receive the average rank for the tied
    range so stable ordering does not accidentally decide an equal metric.
    """
    scores = {id(candidate): 0.0 for candidate in candidates}
    for metric_name, lower_is_better in BORDA_COMPONENTS:
        ranked_values: list[tuple[Candidate, float]] = []
        for candidate in candidates:
            if candidate.metric is None:
                raise RuntimeError(f"candidate {candidate.index} has no metric")
            ranked_values.append((candidate, candidate.metric.score(metric_name)))

        ranked_values.sort(key=lambda item: item[1], reverse=not lower_is_better)
        i = 0
        while i < len(ranked_values):
            j = i + 1
            value = ranked_values[i][1]
            while j < len(ranked_values) and ranked_values[j][1] == value:
                j += 1
            # Ranks are 1-based.  For a tied block [i, j), average the ranks
            # (i + 1) through j.
            average_rank = (i + 1 + j) / 2.0
            for candidate, _ in ranked_values[i:j]:
                scores[id(candidate)] += average_rank
            i = j

    for candidate in candidates:
        candidate.score = scores[id(candidate)]


def rank_candidates(candidates: list[Candidate], settings: EsSettings) -> list[Candidate]:
    if settings.metric == BORDA_COUNT_METRIC:
        assign_borda_count_scores(candidates)
        return sorted(candidates, key=lambda candidate: candidate.score if candidate.score is not None else math.inf)

    def key(candidate: Candidate) -> float:
        if candidate.score is None:
            raise RuntimeError(f"candidate {candidate.index} has no score")
        return candidate.score

    return sorted(candidates, key=key, reverse=not settings.lower_is_better)


def log_stage_rows(
    summary_path: Path,
    generation: int,
    stage: BeamStage,
    ranked: list[Candidate],
    keep: int,
) -> None:
    for rank, candidate in enumerate(ranked, start=1):
        metric = candidate.metric or Metric()
        checkpoint_text = f"worker-cache:{candidate.cache_key}" if candidate.cache_key else str(candidate.checkpoint)
        output_text = f"worker-cache:{candidate.cache_key}" if candidate.cache_key else str(candidate.output_dir or "")
        append_csv(
            summary_path,
            SUMMARY_FIELDS,
            {
                "generation": generation,
                "stage_sbs": stage.after_sbs,
                "candidate": candidate.index,
                "status": "kept" if rank <= keep else "pruned",
                "rank": rank,
                "quantized_value_loss": format_float(metric.qloss),
                "quantized_value_accuracy": format_float(metric.qacc),
                "test_value_loss": format_float(metric.test_loss),
                "test_value_accuracy": format_float(metric.test_acc),
                "checkpoint": checkpoint_text,
                "output_dir": output_text,
                "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
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
    if settings.metric == BORDA_COUNT_METRIC:
        return "lower rank sum is better"
    return "lower is better" if settings.lower_is_better else "higher is better"


def is_better_score(score: float, best_score: float | None, lower_is_better: bool) -> bool:
    if best_score is None:
        return True
    return score < best_score if lower_is_better else score > best_score


def public_checkpoint_name(accepted_sbs: int) -> str:
    return f"sb{accepted_sbs:08d}"


def run_once_from_settings(
    args: argparse.Namespace,
    run: RunSettings,
    settings: EsSettings,
    specs: dict[str, ParameterSpec],
    color: bool,
) -> int:
    """Run one ordinary bulletou.exe training command using current parameter values."""
    epoch_sbs = settings.beam[-1].after_sbs
    params = current_values(specs)
    managed = run.output_folder is not None and run.tag_prefix is not None

    runner_root: Path | None = None
    ordinary_output: Path | None = None
    summary_path: Path | None = None
    accepted_summary_path: Path | None = None
    accepted_root: Path | None = None
    current_dir: Path | None = None

    if managed:
        assert run.output_folder is not None
        assert run.tag_prefix is not None
        runner_root = run.output_folder / f"es-{run.tag_prefix}"
        ordinary_output = runner_root / "bulletou-run"
        log_dir = runner_root / "logs"
        accepted_root = runner_root / "accepted-checkpoints"
        current_dir = runner_root / "current"
        summary_path = runner_root / "summary-learn.log"
        accepted_summary_path = runner_root / "accepted-summary-learn.log"
        if not args.dry_run:
            runner_root.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            accepted_root.mkdir(parents=True, exist_ok=True)
            ensure_csv(summary_path, SUMMARY_FIELDS)
            ensure_csv(accepted_summary_path, ACCEPTED_FIELDS)
            copy_settings_files(runner_root, args.es_settings_file, run.bulletou_settings_file)
        log_path = log_dir / "bulletou-settings-run.stdout.log"
    else:
        log_path = run.bulletou_settings_file.parent / "bulletou-settings-run.stdout.log"

    cmd = [str(run.exe), "--settings-file", str(run.bulletou_settings_file)]
    if managed:
        assert ordinary_output is not None
        cmd.extend(["--output", str(ordinary_output)])
    cmd.extend(["--superbatches", str(epoch_sbs)])
    cmd.extend(["--max-epochs", str(settings.generations)])
    cmd.extend(["--save-rate", str(epoch_sbs)])
    cmd.extend(["--validation-rate", str(settings.validation_rate)])
    cmd.extend(["--quantized-validation-rate", str(settings.quantized_validation_rate)])
    cmd.extend(parameter_args(params))

    initial_checkpoint: Path | None = None
    use_bulletou_resume = False
    if managed:
        assert ordinary_output is not None
        assert current_dir is not None
        if args.resume and maybe_latest_checkpoint_dir(ordinary_output) is not None:
            use_bulletou_resume = True
        elif args.resume and (current_dir / "state.bin").exists():
            initial_checkpoint = current_dir
        elif run.base_checkpoint is not None:
            initial_checkpoint = run.base_checkpoint
    elif args.resume:
        use_bulletou_resume = True

    if use_bulletou_resume:
        cmd.append("--resume")
    elif initial_checkpoint is not None:
        validate_checkpoint_dir(initial_checkpoint)
        cmd.extend(
            [
                "--initial-state",
                str(initial_checkpoint / "state.bin"),
                "--initial-dataloader-pos",
                str(initial_checkpoint / "dataloader_pos.txt"),
            ]
        )

    event(
        color,
        "[RUN]",
        (
            "es.enabled=false; launching one managed ordinary bulletou.exe run "
            f"(superbatches={epoch_sbs}, max_epochs={settings.generations})"
        ),
        "cyan",
    )
    event(color, "[SETTINGS]", f"bulletou={run.bulletou_settings_file}", "cyan")
    if managed:
        event(color, "[OUTPUT]", f"runner={runner_root} bulletou={ordinary_output}", "cyan")
        if use_bulletou_resume:
            event(color, "[RESUME]", f"bulletou output={ordinary_output}", "yellow")
        elif initial_checkpoint is not None:
            event(color, "[INITIAL]", f"checkpoint={initial_checkpoint}", "yellow")

    if args.dry_run:
        print("  " + subprocess.list2cmdline(cmd), flush=True)
        return 0

    code, elapsed = run_command(cmd, log_path, stream=not args.no_stream_child_output)

    if managed:
        assert ordinary_output is not None
        assert summary_path is not None
        assert accepted_summary_path is not None
        assert accepted_root is not None
        assert current_dir is not None
        import_ordinary_run_summaries(
            summary_path=summary_path,
            accepted_summary_path=accepted_summary_path,
            ordinary_output=ordinary_output,
            settings=settings,
            params=params,
            color=color,
        )
        sync_ordinary_accepted_checkpoints(
            accepted_root=accepted_root,
            current_dir=current_dir,
            ordinary_output=ordinary_output,
            settings=settings,
            es_settings_file=args.es_settings_file,
            bulletou_settings_file=run.bulletou_settings_file,
            color=color,
        )

    if code == 0:
        event(color, "[DONE]", f"elapsed={elapsed:.1f}s", "green")
    else:
        event(color, "[ERROR]", f"bulletou.exe exited with code {code}; see {log_path}", "red")
    return code



def main() -> int:
    global LOG_WRITES_ENABLED
    args = parse_args()
    LOG_WRITES_ENABLED = not args.dry_run
    color = color_enabled(args.color)
    root, specs, settings, run = load_parameters(args.es_settings_file)

    if not settings.enabled:
        return run_once_from_settings(args, run, settings, specs, color)

    assert run.output_folder is not None
    assert run.tag_prefix is not None
    runner_root = run.output_folder / f"es-{run.tag_prefix}"
    temp_root = run.temp_folder / f"es-{run.tag_prefix}" if run.temp_folder else runner_root / "temp"
    log_dir = runner_root / "logs"
    accepted_root = runner_root / "accepted-checkpoints"
    current_dir = runner_root / "current"
    state_path = runner_root / "runner-state.json"
    summary_path = runner_root / "summary-learn.log"
    accepted_summary_path = runner_root / "accepted-summary-learn.log"
    history_path = runner_root / "parameters-history.jsonl"

    if not args.dry_run:
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
            raise RuntimeError(f"{state_path} already exists; use --resume or choose a new run.tag_prefix")
        if run.base_checkpoint is None:
            raise RuntimeError(f"{args.es_settings_file}: run.base_checkpoint is required unless --resume is specified")
        validate_checkpoint_dir(run.base_checkpoint)
        # Do not eagerly copy the initial checkpoint into the runner directory.
        # A full SFNN state.bin can be several GiB, and doing this before the
        # first status line makes the runner look frozen.  The first generation
        # can read directly from run.base_checkpoint; after a survivor is chosen
        # we materialize runner_root/current as usual.
        current_checkpoint = run.base_checkpoint
        generation_start = 1
        accepted_sbs = 0
        state = {
            "generation": 0,
            "accepted_sbs": 0,
            "current_checkpoint": str(current_checkpoint),
            "es_settings_file": str(args.es_settings_file.resolve()),
            "bulletou_settings_file": str(run.bulletou_settings_file.resolve()),
        }
        if not args.dry_run:
            save_state(state_path, state)
            write_current_parameters(args.es_settings_file, root, specs)
            copy_settings_files(runner_root, args.es_settings_file, run.bulletou_settings_file)
        event(color, "[START]", f"base_checkpoint={current_checkpoint}", "green")

    validate_checkpoint_dir(current_checkpoint)
    worker_enabled = settings.use_worker and all(stage.keep == 1 for stage in settings.beam)
    if settings.use_worker and not worker_enabled:
        event(
            color,
            "[WORKER]",
            "disabled for this run because the current worker protocol cannot keep multiple beam branches in memory; using ordinary candidate processes",
            "yellow",
        )

    beam_text = ", ".join(f"{stage.after_sbs}sb=>keep{stage.keep}" for stage in settings.beam)
    event(
        color,
        "[CONFIG]",
        (
            f"population={settings.population} generations={settings.generations} "
            f"metric={settings.metric} ({metric_direction_text(settings)}) beam=[{beam_text}] "
            f"save_rate={settings.save_rate} worker={'on' if worker_enabled else 'off'} "
            f"bulletou_settings={run.bulletou_settings_file}"
        ),
        "cyan",
    )
    event(color, "[PARAMETERS]", json.dumps(current_values(specs), ensure_ascii=False, sort_keys=True), "cyan")

    worker: WorkerClient | None = None
    try:
        if worker_enabled:
            if args.dry_run:
                open_args = build_train_args(
                    run,
                    current_values(specs),
                    current_checkpoint,
                    runner_root / "worker-session",
                    settings.beam[-1].after_sbs,
                    settings,
                    include_initial_state=True,
                )
                print("  worker open " + json.dumps({"args": open_args}, ensure_ascii=False), flush=True)
            else:
                worker = WorkerClient(
                    run.exe,
                    log_dir / "worker.stderr.log",
                    stream=not args.no_stream_child_output,
                    color=color,
                )
                worker.request("hello", prefix=paint(color, "[WORKER] ", "magenta"))
                open_args = build_train_args(
                    run,
                    current_values(specs),
                    current_checkpoint,
                    runner_root / "worker-session",
                    settings.beam[-1].after_sbs,
                    settings,
                    include_initial_state=True,
                )
                event(color, "[WORKER OPEN]", f"checkpoint={current_checkpoint}", "yellow")
                payload = worker.request("open", {"args": open_args}, prefix=paint(color, "[WORKER OPEN] ", "magenta"))
                event(
                    color,
                    "[WORKER READY]",
                    (
                        f"arch={payload.get('arch')} batch_size={payload.get('batch_size')} "
                        f"completed_steps={payload.get('completed_steps')}"
                    ),
                    "green",
                )

        total_generations = settings.generations
        for generation in range(generation_start, generation_start + total_generations):
            rng = random.Random(settings.seed + generation * 1_000_003)
            base_metric = latest_learn_log_metric(current_checkpoint)
            candidates = [
                Candidate(index=i + 1, params=perturb_parameters(specs, rng), checkpoint=current_checkpoint)
                for i in range(settings.population)
            ]

            event(
                color,
                f"[GEN {generation} START]",
                (
                    f"generation={generation} population={settings.population} from={current_checkpoint} "
                    f"{metric_status_text(base_metric, prefix='base_')}"
                ),
                "magenta",
            )

            live = candidates
            prev_after_sbs = 0
            for stage in settings.beam:
                trained: list[Candidate] = []
                stage_best_cache_key: str | None = None
                stage_best_score: float | None = None
                for candidate in live:
                    if worker_enabled:
                        trained_candidate = train_candidate_stage_worker(
                            args=args,
                            worker=worker,
                            run=run,
                            settings=settings,
                            summary_path=summary_path,
                            base_metric=base_metric,
                            generation=generation,
                            candidate=candidate,
                            stage=stage,
                            prev_after_sbs=prev_after_sbs,
                            temp_root=temp_root,
                            borda_dominators=[
                                candidate.metric
                                for candidate in trained
                                if settings.metric == BORDA_COUNT_METRIC
                                and candidate.cache_key is not None
                                and candidate.metric is not None
                            ],
                            color=color,
                        )
                        if (
                            not args.dry_run
                            and settings.metric == BORDA_COUNT_METRIC
                            and trained_candidate.cache_key is not None
                            and trained_candidate.metric is not None
                        ):
                            dominated_prior_keys = [
                                prior.cache_key
                                for prior in trained
                                if prior.cache_key is not None
                                and prior.metric is not None
                                and metric_pareto_dominates(trained_candidate.metric, prior.metric)
                            ]
                            if dominated_prior_keys:
                                assert worker is not None
                                worker.request(
                                    "drop-cached-trials",
                                    {"cache_keys": dominated_prior_keys},
                                    prefix=paint(color, f"[G{generation:04d} DOMINATED DROP] ", "magenta"),
                                )
                                dominated_prior_set = set(dominated_prior_keys)
                                for prior in trained:
                                    if prior.cache_key in dominated_prior_set:
                                        prior.cache_key = None
                        trained.append(trained_candidate)
                        if (
                            not args.dry_run
                            and settings.metric != BORDA_COUNT_METRIC
                            and stage.keep == 1
                            and trained_candidate.cache_key is not None
                            and trained_candidate.score is not None
                        ):
                            assert worker is not None
                            if is_better_score(trained_candidate.score, stage_best_score, settings.lower_is_better):
                                if stage_best_cache_key is not None:
                                    worker.request(
                                        "drop-cached-trials",
                                        {"cache_keys": [stage_best_cache_key]},
                                        prefix=paint(color, f"[G{generation:04d} DROP] ", "magenta"),
                                    )
                                stage_best_cache_key = trained_candidate.cache_key
                                stage_best_score = trained_candidate.score
                            else:
                                worker.request(
                                    "drop-cached-trials",
                                    {"cache_keys": [trained_candidate.cache_key]},
                                    prefix=paint(color, f"[G{generation:04d} DROP] ", "magenta"),
                                )
                    else:
                        trained.append(
                            train_candidate_stage(
                                args=args,
                                run=run,
                                settings=settings,
                                summary_path=summary_path,
                                base_metric=base_metric,
                                generation=generation,
                                candidate=candidate,
                                stage=stage,
                                prev_after_sbs=prev_after_sbs,
                                temp_root=temp_root,
                                log_dir=log_dir,
                                color=color,
                            )
                        )
                ranked = rank_candidates(trained, settings)
                log_stage_rows(summary_path, generation, stage, ranked, stage.keep)
                kept = ranked[: stage.keep]
                pruned = ranked[stage.keep :]
                best = kept[0]
                worst_kept = kept[-1]
                event(
                    color,
                    f"[BEAM END]",
                    (
                        f"generation={generation} stage={stage.after_sbs}sb "
                        f"keep={len(kept)}/{len(ranked)} best_{settings.metric}={format_float(best.score)} "
                        f"worst_kept_{settings.metric}={format_float(worst_kept.score)} "
                        f"status=pruned_not_saved"
                    ),
                    "yellow",
                )
                if worker_enabled and not args.dry_run and settings.metric == BORDA_COUNT_METRIC:
                    cache_keys_to_drop = [candidate.cache_key for candidate in pruned if candidate.cache_key is not None]
                    if cache_keys_to_drop:
                        assert worker is not None
                        worker.request(
                            "drop-cached-trials",
                            {"cache_keys": cache_keys_to_drop},
                            prefix=paint(color, f"[G{generation:04d} DROP] ", "magenta"),
                        )
                if worker_enabled and not args.dry_run:
                    if best.cache_key is None:
                        raise RuntimeError("worker survivor has no cache key")
                    assert worker is not None
                    worker.request(
                        "accept-cached-trial",
                        {"cache_key": best.cache_key},
                        prefix=paint(color, f"[G{generation:04d} ACCEPT] ", "magenta"),
                    )
                    best.cache_key = None
                    best.checkpoint = current_checkpoint
                if not args.keep_temp and not args.dry_run:
                    for candidate in pruned:
                        if candidate.output_dir is not None:
                            remove_dir_quiet(candidate.output_dir)
                live = kept
                prev_after_sbs = stage.after_sbs

            survivor = live[0]
            if survivor.metric is None or survivor.score is None:
                raise RuntimeError("final survivor has no metric")

            accepted_sbs += settings.beam[-1].after_sbs
            if not args.dry_run:
                if worker_enabled:
                    assert worker is not None
                    save_args = build_train_args(
                        run,
                        survivor.params,
                        None,
                        runner_root / "worker-save-output",
                        1,
                        settings,
                        include_initial_state=False,
                    )
                    worker.request(
                        "save",
                        {
                            "args": save_args,
                            "dir": str(current_dir),
                            "epoch": generation,
                            "superbatch": settings.beam[-1].after_sbs,
                        },
                        prefix=paint(color, f"[G{generation:04d} SAVE] ", "magenta"),
                    )
                else:
                    copy_dir_replace(survivor.checkpoint, current_dir)
            current_checkpoint = current_dir

            set_current_values(specs, survivor.params)
            if not args.dry_run:
                write_current_parameters(args.es_settings_file, root, specs)
                copy_settings_files(current_dir, args.es_settings_file, run.bulletou_settings_file)

            saved_checkpoint = ""
            if settings.save_rate > 0 and (generation % settings.save_rate == 0):
                public_dir = accepted_root / public_checkpoint_name(accepted_sbs)
                if not args.dry_run:
                    copy_dir_new(current_dir, public_dir)
                    copy_settings_files(public_dir, args.es_settings_file, run.bulletou_settings_file)
                saved_checkpoint = public_dir.name

            params_json = json.dumps(survivor.params, ensure_ascii=False, sort_keys=True)
            metric = survivor.metric
            append_csv(
                accepted_summary_path,
                ACCEPTED_FIELDS,
                {
                    "generation": generation,
                    "accepted_sbs": accepted_sbs,
                    "quantized_value_loss": format_float(metric.qloss),
                    "quantized_value_accuracy": format_float(metric.qacc),
                    "test_value_loss": format_float(metric.test_loss),
                    "test_value_accuracy": format_float(metric.test_acc),
                    "stage_sbs": settings.beam[-1].after_sbs,
                    "saved_checkpoint": saved_checkpoint,
                    "current_checkpoint": str(current_checkpoint),
                    "parameters": params_json,
                },
            )
            append_jsonl(
                history_path,
                {
                    "generation": generation,
                    "accepted_sbs": accepted_sbs,
                    "metric_value": survivor.score,
                    "metric_name": settings.metric,
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
                "last_metric_value": survivor.score,
                "last_metric_name": settings.metric,
                "last_metric": {
                    "quantized_value_loss": metric.qloss,
                    "quantized_value_accuracy": metric.qacc,
                    "test_value_loss": metric.test_loss,
                    "test_value_accuracy": metric.test_acc,
                },
                "es_settings_file": str(args.es_settings_file.resolve()),
                "bulletou_settings_file": str(run.bulletou_settings_file.resolve()),
            }
            if not args.dry_run:
                save_state(state_path, state)

            event(
                color,
                "[ACCEPT]",
                (
                    f"generation={generation} accepted_sbs={accepted_sbs} "
                    f"{metric_status_text(metric)} {settings.metric}={format_float(survivor.score)}"
                ),
                "green",
            )
            if saved_checkpoint:
                event(color, "[SAVE]", f"{accepted_root / saved_checkpoint}", "green")
                event(color, "[SAFE TO STOP]", f"saved={accepted_root / saved_checkpoint}", "green")
            else:
                event(color, "[CURRENT]", f"resume checkpoint updated: {current_checkpoint}", "yellow")

            if not args.keep_temp and not args.dry_run:
                gen_temp = temp_root / f"gen{generation:04d}"
                remove_dir_quiet(gen_temp)

            event(
                color,
                "[GEN END]",
                f"generation={generation} survivor=cand{survivor.index:03d} params={params_json}",
                "magenta",
            )
    finally:
        if worker is not None:
            worker.close()

    event(color, "[DONE]", f"current_checkpoint={current_checkpoint}", "green")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError) as exc:
        if "--debug" in sys.argv:
            raise
        print(f"error: {exc}", file=sys.stderr)
        if "already exists; use --resume" in str(exc):
            print("hint: add --resume to continue the existing ES run, or change run.tag_prefix for a new run.", file=sys.stderr)
        print("hint: rerun with --debug for a Python traceback.", file=sys.stderr)
        raise SystemExit(1)
