#!/usr/bin/env python3
"""Dependency-free fixed-trial parameter tuner for BulletOu.

This is a small local sampler for expensive BulletOu trials:

* each trial starts from scratch or a fixed checkpoint,
* each trial runs for a fixed number of superbatches,
* lr / lr-min and factorizer/count parameters can be sampled together,
* only the current best checkpoint is kept by default.

The sampler starts with uniform random trials, then samples around the best
completed trials in the transformed search space.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bulletou_tuner as tuner


SETTINGS_VERSION = 1

SPECIAL_TRAIN_PARAMETERS = {"lr", "lr_min", "lr_min_ratio"}
KNOWN_SEARCH_PARAMETERS = set(tuner.KNOWN_PARAMETERS) | SPECIAL_TRAIN_PARAMETERS

SUMMARY_FIELDS = [
    "trial",
    "status",
    "score",
    "test_value_accuracy",
    "test_value_loss",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "checkpoint",
    "output_dir",
    "parameters",
]


@dataclass
class SearchParameter:
    name: str
    tune: bool
    current: float | None
    minimum: float | None
    maximum: float | None
    log: bool

    def validate(self) -> None:
        if self.tune:
            if self.minimum is None or self.maximum is None:
                raise ValueError(f"parameters.{self.name}: tune=true requires min/max")
            if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
                raise ValueError(f"parameters.{self.name}: min/max must be finite")
            if self.minimum > self.maximum:
                raise ValueError(f"parameters.{self.name}: min must be <= max")
            if self.log and self.minimum <= 0.0:
                raise ValueError(f"parameters.{self.name}: log sampling requires min > 0")
        else:
            if self.current is None:
                raise ValueError(f"parameters.{self.name}: tune=false requires current")
            if not math.isfinite(self.current):
                raise ValueError(f"parameters.{self.name}: current must be finite")

    def sample_random(self, rng: random.Random) -> float:
        assert self.minimum is not None and self.maximum is not None
        if self.log:
            return math.exp(rng.uniform(math.log(self.minimum), math.log(self.maximum)))
        return rng.uniform(self.minimum, self.maximum)

    def sample_near(self, rng: random.Random, center: float, sigma_ratio: float) -> float:
        assert self.minimum is not None and self.maximum is not None
        if self.log:
            lo = math.log(self.minimum)
            hi = math.log(self.maximum)
            c = math.log(max(center, self.minimum))
            x = rng.gauss(c, max(1e-12, (hi - lo) * sigma_ratio))
            return math.exp(min(max(x, lo), hi))
        x = rng.gauss(center, max(1e-12, (self.maximum - self.minimum) * sigma_ratio))
        return min(max(x, self.minimum), self.maximum)


@dataclass
class StudySettings:
    trials: int
    trial_sbs: int
    metric: str
    lower_is_better: bool
    seed: int
    startup_trials: int
    elite_fraction: float
    elite_sigma: float
    validation_rate: int
    quantized_validation_rate: int
    keep_all_trials: bool


@dataclass
class RunSettings:
    exe: Path
    bulletou_settings_file: Path
    base_checkpoint: Path | None
    output_folder: Path
    temp_folder: Path | None
    tag_prefix: str


@dataclass
class TrialResult:
    trial: int
    params: dict[str, float]
    metric: tuner.Metric
    score: float
    checkpoint: Path
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-length BulletOu parameter tuning trials.")
    parser.add_argument("--settings-file", type=Path, default=Path("tuning-settings.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--no-stream-child-output", action="store_true")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    return parser.parse_args()


def use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


def resolve_path(base: Path, raw: Any, name: str, *, required: bool = True) -> Path | None:
    if raw is None:
        if required:
            raise ValueError(f"{base}: run.{name} is required")
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{base}: run.{name} must be a non-empty string")
    path = Path(raw)
    if not path.is_absolute():
        path = base.parent / path
    return path


def load_settings(path: Path) -> tuple[dict[str, Any], StudySettings, RunSettings, dict[str, SearchParameter]]:
    root = tuner.load_json_object(path)
    version = int(root.get("version", SETTINGS_VERSION))
    if version != SETTINGS_VERSION:
        raise ValueError(f"{path}: unsupported version {version}; expected {SETTINGS_VERSION}")

    if "study" in root:
        raise ValueError(f"{path}: `study` was replaced by `tuning`; update this settings file")
    tuning_obj = root.get("tuning")
    if not isinstance(tuning_obj, dict):
        raise ValueError(f"{path}: `tuning` object is required")
    removed_tuning_keys = sorted(
        set(tuning_obj) & {"method", "beam", "candidate_sbs", "parameter_step_scale", "trials"}
    )
    if removed_tuning_keys:
        raise ValueError(
            f"{path}: obsolete tuning key(s): {', '.join(removed_tuning_keys)}. "
            "Use tuning.trial_sbs and per-parameter ranges instead."
        )
    allowed_tuning_keys = {
        "population",
        "trial_sbs",
        "metric",
        "lower_is_better",
        "seed",
        "startup_trials",
        "elite_fraction",
        "elite_sigma",
        "validation_rate",
        "quantized_validation_rate",
        "keep_all_trials",
    }
    unknown_tuning_keys = sorted(set(tuning_obj) - allowed_tuning_keys)
    if unknown_tuning_keys:
        raise ValueError(f"{path}: unknown tuning field(s): {', '.join(unknown_tuning_keys)}")
    metric = str(tuning_obj.get("metric", "quantized_value_loss"))
    if metric not in {
        "quantized_value_loss",
        "quantized_value_accuracy",
        "test_value_loss",
        "test_value_accuracy",
    }:
        raise ValueError(f"{path}: unsupported tuning.metric {metric!r}")
    lower_is_better = bool(tuning_obj.get("lower_is_better", "loss" in metric))
    settings = StudySettings(
        trials=int(tuning_obj.get("population", 1)),
        trial_sbs=int(tuning_obj.get("trial_sbs", 16)),
        metric=metric,
        lower_is_better=lower_is_better,
        seed=int(tuning_obj.get("seed", 1)),
        startup_trials=int(tuning_obj.get("startup_trials", 16)),
        elite_fraction=float(tuning_obj.get("elite_fraction", 0.25)),
        elite_sigma=float(tuning_obj.get("elite_sigma", 0.15)),
        validation_rate=int(tuning_obj.get("validation_rate", 0)),
        quantized_validation_rate=int(tuning_obj.get("quantized_validation_rate", 0)),
        keep_all_trials=bool(tuning_obj.get("keep_all_trials", False)),
    )
    if settings.trials <= 0:
        raise ValueError("tuning.population must be > 0")
    if settings.trial_sbs <= 0:
        raise ValueError("tuning.trial_sbs must be > 0")
    if settings.startup_trials < 0:
        raise ValueError("tuning.startup_trials must be >= 0")
    if not (0.0 < settings.elite_fraction <= 1.0):
        raise ValueError("tuning.elite_fraction must be in (0, 1]")
    if settings.elite_sigma <= 0.0 or not math.isfinite(settings.elite_sigma):
        raise ValueError("tuning.elite_sigma must be finite and > 0")
    if settings.validation_rate < -1 or settings.quantized_validation_rate < -1:
        raise ValueError("tuning.validation_rate / quantized_validation_rate must be >= -1")
    if metric.startswith("test_value_") and settings.validation_rate < 0:
        raise ValueError(f"tuning.metric {metric!r} requires tuning.validation_rate >= 0")
    if metric.startswith("quantized_value_") and settings.quantized_validation_rate < 0:
        raise ValueError(f"tuning.metric {metric!r} requires tuning.quantized_validation_rate >= 0")

    run_obj = root.get("run")
    if not isinstance(run_obj, dict):
        raise ValueError(f"{path}: `run` object is required")
    unknown_run = sorted(
        set(run_obj)
        - {"exe", "bulletou_settings_file", "base_checkpoint", "output_folder", "temp_folder", "tag_prefix"}
    )
    if unknown_run:
        raise ValueError(f"{path}: unknown run field(s): {', '.join(unknown_run)}")
    exe = resolve_path(path, run_obj.get("exe"), "exe")
    bulletou_settings_file = resolve_path(path, run_obj.get("bulletou_settings_file"), "bulletou_settings_file")
    base_checkpoint = resolve_path(path, run_obj.get("base_checkpoint"), "base_checkpoint", required=False)
    output_folder = resolve_path(path, run_obj.get("output_folder"), "output_folder")
    temp_folder = resolve_path(path, run_obj.get("temp_folder"), "temp_folder", required=False)
    tag_prefix = run_obj.get("tag_prefix")
    if not isinstance(tag_prefix, str) or not tag_prefix.strip():
        raise ValueError(f"{path}: run.tag_prefix must be a non-empty string")
    assert exe is not None
    assert bulletou_settings_file is not None
    assert output_folder is not None
    tuner.validate_bulletou_settings_for_tuning(bulletou_settings_file)
    run = RunSettings(
        exe=exe,
        bulletou_settings_file=bulletou_settings_file,
        base_checkpoint=base_checkpoint,
        output_folder=output_folder,
        temp_folder=temp_folder,
        tag_prefix=tag_prefix.strip(),
    )

    params_obj = root.get("parameters")
    if not isinstance(params_obj, dict):
        raise ValueError(f"{path}: `parameters` object is required")
    unknown = sorted(set(params_obj) - KNOWN_SEARCH_PARAMETERS)
    if unknown:
        raise ValueError(f"{path}: unknown parameter(s): {', '.join(unknown)}")
    if "lr_min" in params_obj and "lr_min_ratio" in params_obj:
        raise ValueError(f"{path}: use either lr_min or lr_min_ratio, not both")

    params: dict[str, SearchParameter] = {}
    for name, raw in params_obj.items():
        if isinstance(raw, (int, float)):
            spec = SearchParameter(name=name, tune=False, current=float(raw), minimum=None, maximum=None, log=False)
        elif isinstance(raw, dict):
            unknown_fields = sorted(set(raw) - {"tune", "current", "min", "max", "log"})
            if unknown_fields:
                raise ValueError(f"parameters.{name}: unknown field(s): {', '.join(unknown_fields)}")
            tune = bool(raw.get("tune", "min" in raw or "max" in raw))
            current = float(raw["current"]) if "current" in raw else None
            minimum = float(raw["min"]) if "min" in raw else None
            maximum = float(raw["max"]) if "max" in raw else None
            if "log" in raw:
                log = bool(raw["log"])
            else:
                # Factorizer/count parameters sometimes intentionally include 0.0
                # in the search range to allow disabling a component.  Log-scale
                # sampling cannot represent 0, so default to linear sampling
                # whenever the lower bound is non-positive.
                log = name in SPECIAL_TRAIN_PARAMETERS or name in tuner.KNOWN_PARAMETERS
                if minimum is not None and minimum <= 0.0:
                    log = False
            spec = SearchParameter(
                name=name,
                tune=tune,
                current=current,
                minimum=minimum,
                maximum=maximum,
                log=log,
            )
        else:
            raise ValueError(f"parameters.{name} must be a number or object")
        spec.validate()
        params[name] = spec

    return root, settings, run, params


def current_fixed_values(params: dict[str, SearchParameter]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, spec in params.items():
        if not spec.tune:
            assert spec.current is not None
            out[name] = spec.current
    return out


def sample_params(
    specs: dict[str, SearchParameter],
    rng: random.Random,
    completed: list[TrialResult],
    settings: StudySettings,
) -> dict[str, float]:
    out = current_fixed_values(specs)
    tuned = [spec for spec in specs.values() if spec.tune]
    # Sample lr_min after lr so lr_min can be constrained to the sampled lr.
    tuned.sort(key=lambda spec: 1 if spec.name == "lr_min" else 0)
    use_elite = len(completed) >= settings.startup_trials and completed
    elite: list[TrialResult] = []
    if use_elite:
        ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
        keep = max(1, math.ceil(len(ordered) * settings.elite_fraction))
        elite = ordered[:keep]

    for spec in tuned:
        sample_spec = spec
        if spec.name == "lr_min" and "lr" in out and spec.maximum is not None:
            assert spec.minimum is not None
            maximum = min(spec.maximum, out["lr"])
            if maximum < spec.minimum:
                maximum = spec.minimum
            sample_spec = SearchParameter(
                name=spec.name,
                tune=spec.tune,
                current=spec.current,
                minimum=spec.minimum,
                maximum=maximum,
                log=spec.log,
            )
        if elite:
            center_trial = rng.choice(elite)
            center = center_trial.params[spec.name]
            out[spec.name] = sample_spec.sample_near(rng, center, settings.elite_sigma)
        else:
            out[spec.name] = sample_spec.sample_random(rng)

    if "lr_min_ratio" in out:
        lr = out.get("lr")
        if lr is None:
            lr_spec = specs.get("lr")
            if lr_spec is None or lr_spec.current is None:
                raise ValueError("lr_min_ratio requires parameter `lr`")
            lr = lr_spec.current
            out["lr"] = lr
        out["lr_min"] = lr * out.pop("lr_min_ratio")
    if "lr" in out and "lr_min" in out and out["lr_min"] > out["lr"]:
        raise ValueError(f"sampled lr_min ({out['lr_min']}) > lr ({out['lr']}); use lr_min_ratio")
    return out


def train_args(run: RunSettings, params: dict[str, float], output_dir: Path, settings: StudySettings) -> list[str]:
    validation_rate = (
        -1
        if settings.validation_rate < 0
        else settings.trial_sbs
        if settings.validation_rate == 0
        else max(1, min(settings.validation_rate, settings.trial_sbs))
    )
    quantized_validation_rate = (
        -1
        if settings.quantized_validation_rate < 0
        else settings.trial_sbs
        if settings.quantized_validation_rate == 0
        else max(1, min(settings.quantized_validation_rate, settings.trial_sbs))
    )
    cmd = [
        str(run.exe),
        "--settings-file",
        str(run.bulletou_settings_file),
        "--output",
        str(output_dir),
        "--superbatches",
        str(settings.trial_sbs),
        "--max-epochs",
        "1",
        "--save-rate",
        str(settings.trial_sbs),
        "--validation-rate",
        str(validation_rate),
        "--quantized-validation-rate",
        str(quantized_validation_rate),
    ]
    if run.base_checkpoint is not None:
        cmd.extend(
            [
                "--initial-state",
                str(run.base_checkpoint / "state.bin"),
                "--initial-dataloader-pos",
                str(run.base_checkpoint / "dataloader_pos.txt"),
            ]
        )
    alpha_values = {k: v for k, v in params.items() if k in tuner.ALPHA_PARAMETERS}
    if alpha_values:
        cmd.extend(["--sfnn-factorizer-alpha", tuner.alpha_arg(alpha_values)])
    for name, flag in tuner.CONFIDENCE_FLAGS.items():
        if name in params and abs(params[name]) > 0.0:
            cmd.extend([flag, f"{params[name]:.12g}"])
    if "lr" in params:
        cmd.extend(["--lr", f"{params['lr']:.12g}"])
    if "lr_min" in params:
        cmd.extend(["--lr-min", f"{params['lr_min']:.12g}"])
    return cmd


def metric_score(metric: tuner.Metric, name: str) -> float:
    return metric.score(name)


def load_completed(summary_path: Path) -> list[TrialResult]:
    if not summary_path.exists():
        return []
    out: list[TrialResult] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "finished":
                continue
            try:
                metric = tuner.metric_from_summary_row(row)
                params = json.loads(row["parameters"])
                out.append(
                    TrialResult(
                        trial=int(row["trial"]),
                        params=params,
                        metric=metric,
                        score=float(row["score"]),
                        checkpoint=Path(row["checkpoint"]),
                        output_dir=Path(row["output_dir"]),
                    )
                )
            except Exception:
                continue
    return out


def write_json(path: Path, obj: dict[str, Any]) -> None:
    tuner.atomic_write_json(path, obj)


def recommended_params(
    specs: dict[str, SearchParameter], completed: list[TrialResult], settings: StudySettings
) -> dict[str, float]:
    if not completed:
        return current_fixed_values(specs)
    ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
    keep = max(1, math.ceil(len(ordered) * settings.elite_fraction))
    elite = ordered[:keep]
    weights = [float(keep - i) for i in range(keep)]
    weight_sum = sum(weights)
    out = current_fixed_values(specs)
    for name, spec in specs.items():
        if not spec.tune:
            continue
        values = [trial.params[name] for trial in elite if name in trial.params]
        if not values:
            continue
        used_weights = weights[: len(values)]
        used_weight_sum = sum(used_weights)
        if spec.log:
            out[name] = math.exp(
                sum(w * math.log(max(v, 1e-300)) for w, v in zip(used_weights, values)) / used_weight_sum
            )
        else:
            out[name] = sum(w * v for w, v in zip(used_weights, values)) / used_weight_sum
    if "lr_min_ratio" in out:
        lr = out.get("lr")
        if lr is not None:
            out["lr_min"] = lr * out["lr_min_ratio"]
    return out


def bulletou_override_preview(params: dict[str, float]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    alpha_values = {k: v for k, v in params.items() if k in tuner.ALPHA_PARAMETERS}
    if alpha_values:
        preview["sfnn_factorizer_alpha"] = tuner.alpha_arg(alpha_values)
    for name, flag in tuner.CONFIDENCE_FLAGS.items():
        if name in params and abs(params[name]) > 0.0:
            preview[flag.removeprefix("--").replace("-", "_")] = params[name]
    if "lr" in params:
        preview["lr"] = params["lr"]
    if "lr_min" in params:
        preview["lr_min"] = params["lr_min"]
    return preview


def write_recommendation(
    path: Path,
    specs: dict[str, SearchParameter],
    completed: list[TrialResult],
    settings: StudySettings,
) -> dict[str, Any]:
    if not completed:
        obj = {
            "trial_count": 0,
            "metric": settings.metric,
            "lower_is_better": settings.lower_is_better,
            "best_observed": None,
            "recommended": {"parameters": current_fixed_values(specs), "bulletou_overrides": {}},
        }
        write_json(path, obj)
        return obj
    ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
    best = ordered[0]
    params = recommended_params(specs, completed, settings)
    obj = {
        "trial_count": len(completed),
        "metric": settings.metric,
        "lower_is_better": settings.lower_is_better,
        "recommendation_method": (
            "rank-weighted elite mean; log-scale parameters use weighted geometric mean"
        ),
        "elite_fraction": settings.elite_fraction,
        "elite_count": max(1, math.ceil(len(completed) * settings.elite_fraction)),
        "best_observed": {
            "trial": best.trial,
            "score": best.score,
            "checkpoint": str(best.checkpoint),
            "parameters": best.params,
            "metrics": {
                "test_value_accuracy": best.metric.test_acc,
                "test_value_loss": best.metric.test_loss,
                "quantized_value_accuracy": best.metric.qacc,
                "quantized_value_loss": best.metric.qloss,
            },
        },
        "recommended": {
            "parameters": params,
            "bulletou_overrides": bulletou_override_preview(params),
        },
    }
    write_json(path, obj)
    return obj


def main() -> int:
    args = parse_args()
    color = use_color(args.color)
    try:
        root, settings, run, specs = load_settings(args.settings_file)
        runner_root = run.output_folder / f"tuning-{run.tag_prefix}"
        trials_root = (run.temp_folder / f"tuning-{run.tag_prefix}" if run.temp_folder else runner_root / "trials")
        log_dir = runner_root / "logs"
        summary_path = runner_root / "summary-learn.log"
        state_path = runner_root / "runner-state.json"
        recommendation_path = runner_root / "recommended-parameters.json"
        best_dir = runner_root / "best-checkpoint"

        if runner_root.exists() and not args.resume and not args.dry_run:
            raise RuntimeError(f"{runner_root} already exists; use --resume or choose a new run.tag_prefix")
        if not args.dry_run:
            runner_root.mkdir(parents=True, exist_ok=True)
            trials_root.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            tuner.ensure_csv(summary_path, SUMMARY_FIELDS)
            shutil.copy2(args.settings_file, runner_root / args.settings_file.name)
            shutil.copy2(run.bulletou_settings_file, runner_root / run.bulletou_settings_file.name)

        completed = load_completed(summary_path) if args.resume else []
        next_trial = 1 + max((r.trial for r in completed), default=0)
        rng = random.Random(settings.seed + next_trial * 1_000_003)
        best = None
        if completed:
            best = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)[0]
        tuner.event(
            color,
            "[CONFIG]",
            (
                f"population={settings.trials} trial_sbs={settings.trial_sbs} metric={settings.metric} "
                f"direction={'minimize' if settings.lower_is_better else 'maximize'} "
                f"startup_trials={settings.startup_trials} elite_fraction={settings.elite_fraction:g} "
                f"elite_sigma={settings.elite_sigma:g}"
            ),
            "cyan",
        )
        if completed:
            tuner.event(color, "[RESUME]", f"completed={len(completed)} next_trial={next_trial}", "yellow")
        if best:
            tuner.event(
                color,
                "[BEST]",
                f"trial={best.trial} score={best.score:.9g} {tuner.metric_status_text(best.metric)}",
                "green",
            )
        if not args.dry_run:
            rec = write_recommendation(recommendation_path, specs, completed, settings)
            if rec.get("best_observed"):
                tuner.event(color, "[RECOMMEND]", f"parameters={recommendation_path}", "green")

        for trial in range(next_trial, settings.trials + 1):
            params = sample_params(specs, rng, completed, settings)
            out_dir = trials_root / f"trial{trial:04d}"
            if out_dir.exists() and not args.dry_run:
                shutil.rmtree(out_dir)
            log_path = log_dir / f"trial{trial:04d}.stdout.log"
            tuner.event(
                color,
                f"[TRIAL {trial:04d} START]",
                " ".join(f"{k}={v:.9g}" for k, v in sorted(params.items())),
                "magenta",
            )
            cmd = train_args(run, params, out_dir, settings)
            if args.dry_run:
                print("  " + subprocess.list2cmdline(cmd), flush=True)
                continue
            code, elapsed = tuner.run_command(
                cmd,
                log_path,
                stream=not args.no_stream_child_output,
                stream_prefix=tuner.paint(color, f"[T{trial:04d}] ", "magenta"),
            )
            if code != 0:
                tuner.append_csv(
                    summary_path,
                    SUMMARY_FIELDS,
                    {
                        "trial": trial,
                        "status": "failed",
                        "score": "",
                        "test_value_accuracy": "",
                        "test_value_loss": "",
                        "quantized_value_accuracy": "",
                        "quantized_value_loss": "",
                        "checkpoint": "",
                        "output_dir": str(out_dir),
                        "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    },
                )
                raise RuntimeError(f"trial {trial} failed; see {log_path}")
            row = tuner.latest_summary_row(out_dir)
            metric = tuner.metric_from_summary_row(row)
            score = metric_score(metric, settings.metric)
            checkpoint = tuner.latest_checkpoint_dir(out_dir)
            result = TrialResult(trial=trial, params=params, metric=metric, score=score, checkpoint=checkpoint, output_dir=out_dir)
            completed.append(result)
            is_best = best is None or ((score < best.score) if settings.lower_is_better else (score > best.score))
            tuner.append_csv(
                summary_path,
                SUMMARY_FIELDS,
                {
                    "trial": trial,
                    "status": "finished",
                    "score": tuner.format_float(score),
                    "test_value_accuracy": tuner.format_float(metric.test_acc),
                    "test_value_loss": tuner.format_float(metric.test_loss),
                    "quantized_value_accuracy": tuner.format_float(metric.qacc),
                    "quantized_value_loss": tuner.format_float(metric.qloss),
                    "checkpoint": str(best_dir if is_best else checkpoint),
                    "output_dir": str(out_dir),
                    "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
                },
            )
            tuner.event(
                color,
                f"[TRIAL {trial:04d} END]",
                f"score={score:.9g} {tuner.metric_status_text(metric)} elapsed={elapsed:.1f}s best={'yes' if is_best else 'no'}",
                "green" if is_best else "cyan",
            )
            if is_best:
                old_best = best_dir.with_name("best-checkpoint.old")
                if old_best.exists():
                    shutil.rmtree(old_best)
                if best_dir.exists():
                    best_dir.rename(old_best)
                shutil.move(str(out_dir), str(best_dir))
                if old_best.exists():
                    shutil.rmtree(old_best)
                result.checkpoint = best_dir
                result.output_dir = best_dir
                best = result
            elif not settings.keep_all_trials and not args.keep_temp:
                tuner.remove_dir_quiet(out_dir)
            state = {
                "version": SETTINGS_VERSION,
                "next_trial": trial + 1,
                "best_trial": best.trial if best else None,
                "best_score": best.score if best else None,
                "best_checkpoint": str(best_dir) if best else None,
                "recommended_parameters": str(recommendation_path),
                "settings_file": str(args.settings_file.resolve()),
            }
            write_json(state_path, state)
            rec = write_recommendation(recommendation_path, specs, completed, settings)
            rec_params = rec["recommended"]["parameters"]
            tuner.event(
                color,
                "[RECOMMEND]",
                " ".join(f"{k}={v:.9g}" for k, v in sorted(rec_params.items())),
                "green",
            )
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: rerun with --debug is not available for this lightweight runner; inspect logs/ for child stdout.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
