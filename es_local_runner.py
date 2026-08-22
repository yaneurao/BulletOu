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
    "residual_count": "--sfnn-residual-count-confidence",
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
    "quantized_value_loss",
    "quantized_value_accuracy",
    "test_value_loss",
    "test_value_accuracy",
    "checkpoint",
    "output_dir",
    "parameters",
]


ACCEPTED_FIELDS = [
    "generation",
    "accepted_sbs",
    "quantized_value_loss",
    "quantized_value_accuracy",
    "test_value_loss",
    "test_value_accuracy",
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
    candidate_validation_rate: int
    candidate_quantized_validation_rate: int


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
        enabled=enabled,
        use_worker=use_worker,
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


def ensure_csv(path: Path, fields: list[str]) -> None:
    if not LOG_WRITES_ENABLED:
        return
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
        str(max(1, min(settings.candidate_validation_rate, stage_delta_sbs))),
        "--quantized-validation-rate",
        str(max(1, min(settings.candidate_quantized_validation_rate, stage_delta_sbs))),
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
            f"base_qloss={format_float(base_metric.qloss)} "
            f"base_qacc={format_float(base_metric.qacc)} "
            f"base_test_loss={format_float(base_metric.test_loss)} "
            f"base_test_acc={format_float(base_metric.test_acc)}"
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
        metric = Metric(qloss=math.inf, qacc=None, test_loss=None, test_acc=None)
        score = metric.score(settings.metric)
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
        score = metric.score(settings.metric)
        checkpoint = latest_checkpoint_dir(out_dir)

    event(
        color,
        f"[CAND {candidate.index:03d} END]",
        (
            f"generation={generation} stage={stage.after_sbs}sb "
            f"{settings.metric}={format_float(score)} qloss={format_float(metric.qloss)} "
            f"qacc={format_float(metric.qacc)} test_loss={format_float(metric.test_loss)} "
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
    color: bool,
) -> Candidate:
    delta = stage.after_sbs - prev_after_sbs
    out_dir = temp_root / f"gen{generation:04d}" / f"stage{stage.after_sbs:04d}" / f"cand{candidate.index:03d}"
    cache_key = f"g{generation:04d}-s{stage.after_sbs:04d}-c{candidate.index:03d}"
    event(
        color,
        f"[CAND {candidate.index:03d} START]",
        (
            f"generation={generation} stage={stage.after_sbs}sb delta={delta}sb worker=on "
            f"base_qloss={format_float(base_metric.qloss)} "
            f"base_qacc={format_float(base_metric.qacc)} "
            f"base_test_loss={format_float(base_metric.test_loss)} "
            f"base_test_acc={format_float(base_metric.test_acc)}"
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
            "output_dir": f"worker-cache:{cache_key}",
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
        print("  worker trial " + json.dumps({"args": trial_args, "keep": False, "cache_key": cache_key}, ensure_ascii=False), flush=True)
        metric = Metric(qloss=math.inf, qacc=None, test_loss=None, test_acc=None)
        score = metric.score(settings.metric)
        elapsed = 0.0
    else:
        if worker is None:
            raise RuntimeError("worker client is not open")
        prefix = paint(color, f"[G{generation:04d} S{stage.after_sbs:04d} C{candidate.index:03d}] ", "magenta")
        started = time.perf_counter()
        payload = worker.request(
            "trial",
            {"args": trial_args, "keep": False, "cache_key": cache_key},
            prefix=prefix,
        )
        elapsed = time.perf_counter() - started
        metric = metric_from_worker_payload(payload)
        score = metric.score(settings.metric)

    event(
        color,
        f"[CAND {candidate.index:03d} END]",
        (
            f"generation={generation} stage={stage.after_sbs}sb worker=on "
            f"{settings.metric}={format_float(score)} qloss={format_float(metric.qloss)} "
            f"qacc={format_float(metric.qacc)} test_loss={format_float(metric.test_loss)} "
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
            "checkpoint": f"worker-cache:{cache_key}",
            "output_dir": f"worker-cache:{cache_key}",
            "parameters": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        },
    )

    candidate.cache_key = cache_key
    candidate.output_dir = None
    candidate.metric = metric
    candidate.score = score
    candidate.stage_sbs = stage.after_sbs
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
    specs: dict[str, ParameterSpec],
    color: bool,
) -> int:
    """Run one ordinary bulletou.exe training command using current parameter values."""
    cmd = [str(run.exe), "--settings-file", str(run.bulletou_settings_file)]
    cmd.extend(parameter_args(current_values(specs)))
    if args.resume:
        cmd.append("--resume")
    event(
        color,
        "[RUN]",
        "es.enabled=false; launching one bulletou.exe run with current parameter values",
        "cyan",
    )
    event(color, "[SETTINGS]", f"bulletou={run.bulletou_settings_file}", "cyan")
    if args.dry_run:
        print("  " + subprocess.list2cmdline(cmd), flush=True)
        return 0
    code, elapsed = run_command(cmd, Path("bulletou-settings-run.stdout.log"), stream=not args.no_stream_child_output)
    if code == 0:
        event(color, "[DONE]", f"elapsed={elapsed:.1f}s", "green")
    else:
        event(color, "[ERROR]", f"bulletou.exe exited with code {code}; see bulletou-settings-run.stdout.log", "red")
    return code


def main() -> int:
    global LOG_WRITES_ENABLED
    args = parse_args()
    LOG_WRITES_ENABLED = not args.dry_run
    color = color_enabled(args.color)
    root, specs, settings, run = load_parameters(args.es_settings_file)

    if not settings.enabled:
        return run_once_from_settings(args, run, specs, color)

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
                    f"base_qloss={format_float(base_metric.qloss)} "
                    f"base_qacc={format_float(base_metric.qacc)} "
                    f"base_test_loss={format_float(base_metric.test_loss)} "
                    f"base_test_acc={format_float(base_metric.test_acc)}"
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
                            color=color,
                        )
                        trained.append(trained_candidate)
                        if (
                            not args.dry_run
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
                ranked = rank_candidates(trained, settings.lower_is_better)
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
                    f"{settings.metric}={format_float(survivor.score)} qloss={format_float(metric.qloss)} "
                    f"qacc={format_float(metric.qacc)}"
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
