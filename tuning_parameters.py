#!/usr/bin/env python3
"""Dependency-free fixed-trial parameter tuner for BulletOu.

This is a small local sampler for expensive BulletOu trials:

* each generation starts from scratch or the current checkpoint,
* each trial in a generation starts from that generation's checkpoint,
* each trial runs for a fixed number of superbatches,
* lr / lr-min and factorizer/count parameters can be sampled together,
* one generation-end commit run creates the next generation's checkpoint,
* only the current checkpoint and the observed best checkpoint are kept by default.

The first generation starts with uniform random trials. Later generations use
the previous generation: TPE when enough trials exist, otherwise Gaussian
sampling around the previous generation's recommended parameters.
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
    "test_value_accuracy",
    "test_value_loss",
    "quantized_value_accuracy",
    "quantized_value_loss",
    "parameters",
    "selection_metric",
    "checkpoint",
]

TPE_CANDIDATES = 64


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
    generations: int
    population_schedule: list[int]
    trial_sbs_schedule: list[int]
    metric: str
    lower_is_better: bool
    sampler: str
    seed: int
    tpe_startup_trials: int
    tpe_good_fraction: float
    tpe_bandwidth: float
    validation_rate: int
    quantized_validation_rate: int
    keep_all_trials: bool
    use_worker: bool
    commit_source: str

    def population_for_generation(self, generation: int) -> int:
        return self.population_schedule[min(generation - 1, len(self.population_schedule) - 1)]

    def trial_sbs_for_generation(self, generation: int) -> int:
        return self.trial_sbs_schedule[min(generation - 1, len(self.trial_sbs_schedule) - 1)]

    @property
    def trials(self) -> int:
        return sum(self.population_for_generation(generation) for generation in range(1, self.generations + 1))


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


def parse_positive_int_schedule(raw: Any, name: str, default: int) -> list[int]:
    if raw is None:
        values = [default]
    elif isinstance(raw, list):
        if not raw:
            raise ValueError(f"{name} must not be an empty array")
        values = [int(value) for value in raw]
    else:
        values = [int(raw)]
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain only positive integers")
    return values


def generation_for_trial(settings: StudySettings, trial: int) -> tuple[int, int]:
    if trial <= 0:
        raise ValueError("trial must be >= 1")
    first_trial = 1
    for generation in range(1, settings.generations + 1):
        population = settings.population_for_generation(generation)
        last_trial = first_trial + population - 1
        if trial <= last_trial:
            return generation, trial - first_trial + 1
        first_trial = last_trial + 1
    return settings.generations, settings.population_for_generation(settings.generations)


def first_trial_of_generation(settings: StudySettings, generation: int) -> int:
    first = 1
    for prior in range(1, generation):
        first += settings.population_for_generation(prior)
    return first


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
    renamed_tuning_keys = {
        "startup_trials": "tpe_startup_trials",
        "elite_fraction": "tpe_good_fraction",
        "elite_sigma": "tpe_bandwidth",
    }
    present_renamed_keys = [key for key in renamed_tuning_keys if key in tuning_obj]
    if present_renamed_keys:
        replacements = ", ".join(f"{key}->{renamed_tuning_keys[key]}" for key in present_renamed_keys)
        raise ValueError(f"{path}: renamed tuning key(s): {replacements}")
    allowed_tuning_keys = {
        "generations",
        "population",
        "trial_sbs",
        "metric",
        "lower_is_better",
        "sampler",
        "seed",
        "tpe_startup_trials",
        "tpe_good_fraction",
        "tpe_bandwidth",
        "validation_rate",
        "quantized_validation_rate",
        "keep_all_trials",
        "use_worker",
        "commit_source",
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
    sampler = str(tuning_obj.get("sampler", "tpe"))
    if sampler not in {"tpe", "random"}:
        raise ValueError(f"{path}: unsupported tuning.sampler {sampler!r}; expected 'tpe' or 'random'")
    commit_source = str(tuning_obj.get("commit_source", "best"))
    if commit_source not in {"best", "recommended"}:
        raise ValueError(f"{path}: unsupported tuning.commit_source {commit_source!r}; expected 'best' or 'recommended'")
    population_schedule = parse_positive_int_schedule(tuning_obj.get("population"), "tuning.population", 1)
    trial_sbs_schedule = parse_positive_int_schedule(tuning_obj.get("trial_sbs"), "tuning.trial_sbs", 16)
    default_generations = max(len(population_schedule), len(trial_sbs_schedule), 1)
    generations = int(tuning_obj.get("generations", default_generations))
    if generations <= 0:
        raise ValueError("tuning.generations must be > 0")
    settings = StudySettings(
        generations=generations,
        population_schedule=population_schedule,
        trial_sbs_schedule=trial_sbs_schedule,
        metric=metric,
        lower_is_better=lower_is_better,
        sampler=sampler,
        seed=int(tuning_obj.get("seed", 1)),
        tpe_startup_trials=int(tuning_obj.get("tpe_startup_trials", 16)),
        tpe_good_fraction=float(tuning_obj.get("tpe_good_fraction", 0.25)),
        tpe_bandwidth=float(tuning_obj.get("tpe_bandwidth", 0.15)),
        validation_rate=int(tuning_obj.get("validation_rate", 0)),
        quantized_validation_rate=int(tuning_obj.get("quantized_validation_rate", 0)),
        keep_all_trials=bool(tuning_obj.get("keep_all_trials", False)),
        use_worker=bool(tuning_obj.get("use_worker", True)),
        commit_source=commit_source,
    )
    if settings.tpe_startup_trials < 0:
        raise ValueError("tuning.tpe_startup_trials must be >= 0")
    if not (0.0 < settings.tpe_good_fraction <= 1.0):
        raise ValueError("tuning.tpe_good_fraction must be in (0, 1]")
    if settings.tpe_bandwidth <= 0.0 or not math.isfinite(settings.tpe_bandwidth):
        raise ValueError("tuning.tpe_bandwidth must be finite and > 0")
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


def transformed_bounds(spec: SearchParameter) -> tuple[float, float]:
    assert spec.minimum is not None and spec.maximum is not None
    if spec.log:
        return math.log(spec.minimum), math.log(spec.maximum)
    return spec.minimum, spec.maximum


def transform_value(spec: SearchParameter, value: float) -> float:
    if spec.log:
        assert spec.minimum is not None
        return math.log(max(value, spec.minimum))
    return value


def untransform_value(spec: SearchParameter, value: float) -> float:
    lo, hi = transformed_bounds(spec)
    value = min(max(value, lo), hi)
    if spec.log:
        return math.exp(value)
    return value


def gaussian_log_pdf(x: float, mean: float, sigma: float) -> float:
    sigma = max(sigma, 1e-12)
    z = (x - mean) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)


def logsumexp(values: list[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if not math.isfinite(m):
        return m
    return m + math.log(sum(math.exp(value - m) for value in values))


def kde_bandwidth(values: list[float], lo: float, hi: float, sigma_ratio: float) -> float:
    width = max(hi - lo, 1e-12)
    if len(values) <= 1:
        return max(width * sigma_ratio, 1e-12)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    std = math.sqrt(max(0.0, variance))
    return max(width * sigma_ratio, std * 0.5, 1e-12)


def kde_log_density(x: float, values: list[float], lo: float, hi: float, sigma_ratio: float) -> float:
    if not values:
        return -math.inf
    sigma = kde_bandwidth(values, lo, hi, sigma_ratio)
    return logsumexp([gaussian_log_pdf(x, value, sigma) for value in values]) - math.log(len(values))


def split_good_bad(completed: list[TrialResult], settings: StudySettings) -> tuple[list[TrialResult], list[TrialResult]]:
    ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
    good_count = max(1, math.ceil(len(ordered) * settings.tpe_good_fraction))
    good_count = min(good_count, max(1, len(ordered) - 1))
    return ordered[:good_count], ordered[good_count:]


def sample_params_random(specs: dict[str, SearchParameter], rng: random.Random) -> dict[str, float]:
    out = current_fixed_values(specs)
    tuned = [spec for spec in specs.values() if spec.tune]
    # Sample lr_min after lr so lr_min can be constrained to the sampled lr.
    tuned.sort(key=lambda spec: 1 if spec.name == "lr_min" else 0)
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


def sample_params_near(
    specs: dict[str, SearchParameter],
    rng: random.Random,
    center: dict[str, float],
    sigma_ratio: float,
) -> dict[str, float]:
    out = current_fixed_values(specs)
    tuned = [spec for spec in specs.values() if spec.tune]
    # Sample lr_min after lr so lr_min can be constrained to the sampled lr.
    tuned.sort(key=lambda spec: 1 if spec.name == "lr_min" else 0)
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
        center_value = center.get(spec.name)
        if center_value is None or not math.isfinite(float(center_value)):
            out[spec.name] = sample_spec.sample_random(rng)
        else:
            out[spec.name] = sample_spec.sample_near(rng, float(center_value), sigma_ratio)

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
        out["lr_min"] = out["lr"]
    return out


def sample_one_parameter_tpe(
    spec: SearchParameter,
    rng: random.Random,
    good: list[TrialResult],
    *,
    override_maximum: float | None = None,
    sigma_ratio: float,
) -> float:
    lo, hi = transformed_bounds(spec)
    if override_maximum is not None:
        if spec.log:
            assert spec.minimum is not None
            hi = min(hi, math.log(max(override_maximum, spec.minimum)))
        else:
            hi = min(hi, override_maximum)
    if hi < lo:
        hi = lo
    values = [transform_value(spec, trial.params[spec.name]) for trial in good if spec.name in trial.params]
    if not values:
        return spec.sample_random(rng)
    center = rng.choice(values)
    sigma = kde_bandwidth(values, lo, hi, sigma_ratio)
    return untransform_value(spec, rng.gauss(center, sigma))


def tpe_candidate_score(
    candidate: dict[str, float],
    specs: dict[str, SearchParameter],
    good: list[TrialResult],
    bad: list[TrialResult],
    sigma_ratio: float,
) -> float:
    score = 0.0
    for name, value in candidate.items():
        spec = specs[name]
        if not spec.tune or not bad:
            continue
        lo, hi = transformed_bounds(spec)
        x = transform_value(spec, value)
        good_values = [transform_value(spec, trial.params[name]) for trial in good if name in trial.params]
        bad_values = [transform_value(spec, trial.params[name]) for trial in bad if name in trial.params]
        if not good_values or not bad_values:
            continue
        score += kde_log_density(x, good_values, lo, hi, sigma_ratio) - kde_log_density(
            x, bad_values, lo, hi, sigma_ratio
        )
    return score


def sample_params_tpe(
    specs: dict[str, SearchParameter],
    rng: random.Random,
    completed: list[TrialResult],
    settings: StudySettings,
    center: dict[str, float] | None,
) -> dict[str, float]:
    if len(completed) < settings.tpe_startup_trials:
        if center is not None:
            return sample_params_near(specs, rng, center, settings.tpe_bandwidth)
        return sample_params_random(specs, rng)
    good, bad = split_good_bad(completed, settings)
    if not good or not bad:
        if center is not None:
            return sample_params_near(specs, rng, center, settings.tpe_bandwidth)
        return sample_params_random(specs, rng)

    best_params: dict[str, float] | None = None
    best_score = -math.inf
    for _ in range(TPE_CANDIDATES):
        out = current_fixed_values(specs)
        tuned = [spec for spec in specs.values() if spec.tune]
        tuned.sort(key=lambda spec: 1 if spec.name == "lr_min" else 0)
        for spec in tuned:
            override_maximum = out["lr"] if spec.name == "lr_min" and "lr" in out else None
            out[spec.name] = sample_one_parameter_tpe(
                spec,
                rng,
                good,
                override_maximum=override_maximum,
                sigma_ratio=settings.tpe_bandwidth,
            )
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
            out["lr_min"] = out["lr"]
        score = tpe_candidate_score(out, specs, good, bad, settings.tpe_bandwidth)
        if best_params is None or score > best_score:
            best_params = out
            best_score = score
    assert best_params is not None
    return best_params


def sample_params(
    specs: dict[str, SearchParameter],
    rng: random.Random,
    completed: list[TrialResult],
    settings: StudySettings,
    center: dict[str, float] | None = None,
) -> dict[str, float]:
    if settings.sampler == "random":
        return sample_params_random(specs, rng)
    return sample_params_tpe(specs, rng, completed, settings, center)


def train_args(
    run: RunSettings,
    params: dict[str, float],
    base_checkpoint: Path | None,
    output_dir: Path,
    settings: StudySettings,
    trial_sbs: int,
    *,
    include_initial_state: bool = True,
    save_checkpoint: bool = True,
) -> list[str]:
    validation_rate = (
        -1
        if settings.validation_rate < 0
        else trial_sbs
        if settings.validation_rate == 0
        else max(1, min(settings.validation_rate, trial_sbs))
    )
    quantized_validation_rate = (
        -1
        if settings.quantized_validation_rate < 0
        else trial_sbs
        if settings.quantized_validation_rate == 0
        else max(1, min(settings.quantized_validation_rate, trial_sbs))
    )
    cmd = [
        str(run.exe),
        "--settings-file",
        str(run.bulletou_settings_file),
        "--output",
        str(output_dir),
        "--superbatches",
        str(trial_sbs),
        "--max-epochs",
        "1",
        "--save-rate",
        str(trial_sbs if save_checkpoint else 999999999),
        "--validation-rate",
        str(validation_rate),
        "--quantized-validation-rate",
        str(quantized_validation_rate),
    ]
    if not save_checkpoint:
        cmd.append("--no-save-epoch-end")
    if include_initial_state and base_checkpoint is not None:
        cmd.extend(
            [
                "--initial-state",
                str(base_checkpoint / "state.bin"),
                "--initial-dataloader-pos",
                str(base_checkpoint / "dataloader_pos.txt"),
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


def worker_args_from_train_args(cmd: list[str]) -> list[str]:
    if not cmd:
        raise ValueError("empty bulletou command")
    return cmd[1:]


def metric_score(metric: tuner.Metric, name: str) -> float:
    return metric.score(name)


def move_dir_replace(src: Path, dst: Path) -> None:
    if not src.exists():
        raise RuntimeError(f"source directory does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    old = dst.with_name(dst.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if dst.exists():
        os.replace(dst, old)
    try:
        try:
            os.replace(src, dst)
        except OSError:
            # os.replace cannot move directories across drives.  This is mainly
            # for non-worker mode when temp_folder and output_folder live on
            # different volumes.
            shutil.move(str(src), str(dst))
    except Exception:
        if old.exists() and not dst.exists():
            os.replace(old, dst)
        raise
    if old.exists():
        shutil.rmtree(old)


def is_better_score(score: float, best_score: float | None, lower_is_better: bool) -> bool:
    if best_score is None:
        return True
    return score < best_score if lower_is_better else score > best_score


def result_generation(settings: StudySettings, result: TrialResult) -> int:
    return generation_for_trial(settings, result.trial)[0]


def generation_results(settings: StudySettings, completed: list[TrialResult], generation: int) -> list[TrialResult]:
    return [result for result in completed if result_generation(settings, result) == generation]


def latest_generation_with_results(settings: StudySettings, completed: list[TrialResult]) -> int | None:
    if not completed:
        return None
    return max(result_generation(settings, result) for result in completed)


def best_result(results: list[TrialResult], settings: StudySettings) -> TrialResult | None:
    if not results:
        return None
    return sorted(results, key=lambda r: r.score, reverse=not settings.lower_is_better)[0]


def upgrade_summary_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)
    if old_fields == SUMMARY_FIELDS:
        return
    if "score" not in old_fields and "selection_metric" not in old_fields:
        expected = ",".join(SUMMARY_FIELDS)
        existing = ",".join(old_fields)
        raise RuntimeError(f"{path} has incompatible header\n  existing: {existing}\n  expected: {expected}")
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row["selection_metric"] = row.get("selection_metric") or row.get("score") or ""
            checkpoint = row.get("checkpoint") or ""
            if checkpoint.startswith(("worker-cache:", "worker-disk-cache:", "worker-dominated:")):
                row["checkpoint"] = ""
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})
    tmp.replace(path)


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
                selection_metric = row.get("selection_metric") or row.get("score")
                if selection_metric is None:
                    continue
                out.append(
                    TrialResult(
                        trial=int(row["trial"]),
                        params=params,
                        metric=metric,
                        score=float(selection_metric),
                        checkpoint=Path(row["checkpoint"]),
                        output_dir=Path(row.get("output_dir") or row.get("checkpoint") or ""),
                    )
                )
            except Exception:
                continue
    return out


def write_json(path: Path, obj: dict[str, Any]) -> None:
    tuner.atomic_write_json(path, obj)


def configured_teacher_memory_cache_sbs(path: Path) -> int:
    try:
        obj = tuner.load_json_object(path)
    except Exception:
        return 0
    raw = obj.get("teacher_memory_cache_sbs")
    if raw is None:
        return 0
    try:
        return int(raw)
    except Exception:
        return 0


def open_worker_session(
    *,
    worker: tuner.WorkerClient,
    run: RunSettings,
    specs: dict[str, SearchParameter],
    current_checkpoint: Path | None,
    runner_root: Path,
    settings: StudySettings,
    trial_sbs: int,
    color: bool,
) -> None:
    open_dir = runner_root / "worker-session"
    open_params = current_fixed_values(specs)
    open_args = worker_args_from_train_args(
        train_args(run, open_params, current_checkpoint, open_dir, settings, trial_sbs)
    )
    tuner.event(color, "[WORKER OPEN]", f"checkpoint={current_checkpoint or 'scratch'}", "yellow")
    payload = worker.request(
        "open",
        {"args": open_args},
        prefix=tuner.paint(color, "[WORKER OPEN] ", "magenta"),
    )
    tuner.event(
        color,
        "[WORKER READY]",
        (
            f"arch={payload.get('arch')} batch_size={payload.get('batch_size')} "
            f"completed_steps={payload.get('completed_steps')}"
        ),
        "green",
    )


def recommended_params(
    specs: dict[str, SearchParameter], completed: list[TrialResult], settings: StudySettings
) -> dict[str, float]:
    if not completed:
        return current_fixed_values(specs)
    ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
    keep = max(1, math.ceil(len(ordered) * settings.tpe_good_fraction))
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
    latest_generation = latest_generation_with_results(settings, completed)
    recommendation_trials = (
        generation_results(settings, completed, latest_generation) if latest_generation is not None else []
    )
    if not completed:
        obj = {
            "trial_count": 0,
            "metric": settings.metric,
            "lower_is_better": settings.lower_is_better,
            "best_observed": None,
            "recommendation_scope": "none",
            "recommended": {"parameters": current_fixed_values(specs), "bulletou_overrides": {}},
        }
        write_json(path, obj)
        return obj
    ordered = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)
    best = ordered[0]
    params = recommended_params(specs, recommendation_trials, settings)
    obj = {
        "trial_count": len(completed),
        "recommended_trial_count": len(recommendation_trials),
        "metric": settings.metric,
        "lower_is_better": settings.lower_is_better,
        "recommendation_method": (
            "latest-generation rank-weighted TPE-good mean; log-scale parameters use weighted geometric mean"
        ),
        "recommendation_scope": (
            f"generation {latest_generation}" if latest_generation is not None else "none"
        ),
        "tpe_good_fraction": settings.tpe_good_fraction,
        "tpe_good_count": max(1, math.ceil(len(recommendation_trials) * settings.tpe_good_fraction))
        if recommendation_trials
        else 0,
        "best_observed": {
            "trial": best.trial,
            "generation": result_generation(settings, best),
            "selection_metric": best.score,
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
        current_dir = runner_root / "current-checkpoint"
        pending_commit_dir = runner_root / "pending-commit-checkpoint"
        best_dir = runner_root / "best-checkpoint"

        if runner_root.exists() and not args.resume and not args.dry_run:
            raise RuntimeError(f"{runner_root} already exists; use --resume or choose a new run.tag_prefix")
        if not args.dry_run:
            runner_root.mkdir(parents=True, exist_ok=True)
            trials_root.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            upgrade_summary_csv(summary_path)
            tuner.ensure_csv(summary_path, SUMMARY_FIELDS)
            shutil.copy2(args.settings_file, runner_root / args.settings_file.name)
            shutil.copy2(run.bulletou_settings_file, runner_root / run.bulletou_settings_file.name)

        completed = load_completed(summary_path) if args.resume else []
        state: dict[str, Any] = {}
        if args.resume and state_path.exists():
            state = tuner.load_json_object(state_path)
            current_checkpoint_raw = state.get("current_checkpoint")
            current_checkpoint = Path(current_checkpoint_raw) if current_checkpoint_raw else (
                current_dir if current_dir.exists() else run.base_checkpoint
            )
            next_trial = int(state.get("next_trial", 1 + max((r.trial for r in completed), default=0)))
            if next_trial < 1:
                next_trial = 1
            completed = [result for result in completed if result.trial < next_trial]
        elif args.resume:
            current_checkpoint = current_dir if current_dir.exists() else (best_dir if best_dir.exists() else run.base_checkpoint)
            next_trial = 1 + max((r.trial for r in completed), default=0)
        else:
            current_checkpoint = run.base_checkpoint
            next_trial = 1
            state = {
                "version": SETTINGS_VERSION,
                "next_trial": next_trial,
                "current_checkpoint": str(current_checkpoint) if current_checkpoint else None,
                "best_trial": None,
                "best_selection_metric": None,
                "best_checkpoint": None,
                "best_commit_generation": None,
                "best_commit_selection_metric": None,
                "pending_commit_generation": None,
                "pending_commit_source": None,
                "pending_commit_params": None,
                "recommended_parameters": str(recommendation_path),
                "settings_file": str(args.settings_file.resolve()),
            }
            if not args.dry_run:
                write_json(state_path, state)
        if current_checkpoint is not None:
            tuner.validate_checkpoint_dir(current_checkpoint)
        rng = random.Random(settings.seed + next_trial * 1_000_003)
        best = None
        if completed:
            best = sorted(completed, key=lambda r: r.score, reverse=not settings.lower_is_better)[0]
        best_commit_score = None
        if state.get("best_commit_selection_metric") is not None:
            best_commit_score = float(state["best_commit_selection_metric"])
        elif state.get("best_selection_metric") is not None and best_dir.exists():
            # Older state files used best_selection_metric for the saved best checkpoint.
            best_commit_score = float(state["best_selection_metric"])
        tuner.event(
            color,
            "[CONFIG]",
            (
                f"generations={settings.generations} total_trials={settings.trials} "
                f"population={settings.population_schedule} trial_sbs={settings.trial_sbs_schedule} "
                f"sampler={settings.sampler} metric={settings.metric} "
                f"direction={'minimize' if settings.lower_is_better else 'maximize'} "
                f"commit_source={settings.commit_source} "
                f"tpe_startup_trials={settings.tpe_startup_trials} "
                f"tpe_good_fraction={settings.tpe_good_fraction:g} "
                f"tpe_bandwidth={settings.tpe_bandwidth:g} worker={'on' if settings.use_worker else 'off'}"
            ),
            "cyan",
        )
        cache_sbs = configured_teacher_memory_cache_sbs(run.bulletou_settings_file)
        if cache_sbs > 0 and settings.use_worker:
            tuner.event(
                color,
                "[CACHE]",
                f"teacher_memory_cache_sbs={cache_sbs} will be used by the long-lived worker process",
                "green",
            )
        elif cache_sbs > 0:
            tuner.event(
                color,
                "[WARN]",
                (
                    f"teacher_memory_cache_sbs={cache_sbs} is worker-mode only; "
                    "tuning.use_worker=false starts a fresh bulletou.exe process for each trial, "
                    "so the RAM teacher cache is not used here."
                ),
                "yellow",
            )
        if completed:
            tuner.event(
                color,
                "[RESUME]",
                f"completed={len(completed)} next_trial={next_trial} current_checkpoint={current_checkpoint or 'scratch'}",
                "yellow",
            )
        if best:
            tuner.event(
                color,
                "[BEST]",
                f"trial={best.trial} selection_metric={best.score:.9g} {tuner.metric_status_text(best.metric)}",
                "green",
            )
        if not args.dry_run:
            rec = write_recommendation(recommendation_path, specs, completed, settings)
            if rec.get("best_observed"):
                tuner.event(color, "[RECOMMEND]", f"parameters={recommendation_path}", "green")

        worker: tuner.WorkerClient | None = None
        try:
            if settings.use_worker:
                open_trial_sbs = settings.trial_sbs_for_generation(generation_for_trial(settings, next_trial)[0])
                if args.dry_run:
                    open_dir = runner_root / "worker-session"
                    open_params = current_fixed_values(specs)
                    open_args = worker_args_from_train_args(
                        train_args(run, open_params, current_checkpoint, open_dir, settings, open_trial_sbs)
                    )
                    print("  worker open " + json.dumps({"args": open_args}, ensure_ascii=False), flush=True)
                else:
                    worker = tuner.WorkerClient(
                        run.exe,
                        log_dir / "worker.stderr.log",
                        stream=not args.no_stream_child_output,
                        color=color,
                    )
                    worker.request("hello", prefix=tuner.paint(color, "[WORKER] ", "magenta"))
                    open_worker_session(
                        worker=worker,
                        run=run,
                        specs=specs,
                        current_checkpoint=current_checkpoint,
                        runner_root=runner_root,
                        settings=settings,
                        trial_sbs=open_trial_sbs,
                        color=color,
                    )

            pending_commit_generation = state.get("pending_commit_generation")
            if pending_commit_generation is not None:
                start_generation = int(pending_commit_generation)
            else:
                start_generation = (
                    generation_for_trial(settings, next_trial)[0]
                    if next_trial <= settings.trials
                    else settings.generations + 1
                )
            for generation in range(start_generation, settings.generations + 1):
                generation_first_trial = first_trial_of_generation(settings, generation)
                population = settings.population_for_generation(generation)
                generation_last_trial = generation_first_trial + population - 1
                trial_sbs = settings.trial_sbs_for_generation(generation)
                previous_generation_trials = (
                    generation_results(settings, completed, generation - 1) if generation > 1 else []
                )
                previous_recommended_params = (
                    recommended_params(specs, previous_generation_trials, settings)
                    if previous_generation_trials
                    else None
                )
                if settings.sampler == "random" or not previous_generation_trials:
                    sampler_center = "random"
                elif (
                    previous_recommended_params is not None
                    and (
                        len(previous_generation_trials) < settings.tpe_startup_trials
                        or len(previous_generation_trials) < 2
                    )
                ):
                    sampler_center = "previous-recommended"
                else:
                    sampler_center = "tpe"
                rng = random.Random(settings.seed + generation * 1_000_003)
                tuner.event(
                    color,
                    f"[GEN {generation} START]",
                    (
                        f"population={population} trial_sbs={trial_sbs} "
                        f"base={current_checkpoint or 'scratch'} "
                        f"sampler_trials={len(previous_generation_trials)} "
                        f"sampler_center={sampler_center}"
                    ),
                    "magenta",
                )

                generation_best = best_result(generation_results(settings, completed, generation), settings)
                if generation_best is not None:
                    tuner.event(
                        color,
                        f"[GEN {generation} RESUME]",
                        (
                            f"completed_trials={len(generation_results(settings, completed, generation))} "
                            f"best_trial={generation_best.trial} "
                            f"best_selection_metric={generation_best.score:.9g}"
                        ),
                        "yellow",
                    )
                trial_start = max(next_trial, generation_first_trial)
                for _ in range(generation_first_trial, trial_start):
                    sample_params(specs, rng, previous_generation_trials, settings, previous_recommended_params)
                for trial in range(trial_start, generation_last_trial + 1):
                    generation_trial = trial - generation_first_trial + 1
                    params = sample_params(specs, rng, previous_generation_trials, settings, previous_recommended_params)
                    out_dir = trials_root / f"trial{trial:04d}"
                    if out_dir.exists() and not args.dry_run:
                        shutil.rmtree(out_dir)
                    log_path = log_dir / f"trial{trial:04d}.stdout.log"
                    display_prefix = f"[GEN {generation}][TRIAL {generation_trial}]"
                    tuner.event(
                        color,
                        f"{display_prefix} START",
                        " ".join(f"{k}={v:.9g}" for k, v in sorted(params.items())),
                        "magenta",
                    )
                    cmd = train_args(
                        run,
                        params,
                        current_checkpoint,
                        out_dir,
                        settings,
                        trial_sbs,
                        include_initial_state=not settings.use_worker,
                        save_checkpoint=(not settings.use_worker) and (settings.keep_all_trials or args.keep_temp),
                    )
                    if args.dry_run:
                        if settings.use_worker:
                            print(
                                "  worker trial "
                                + json.dumps(
                                    {
                                        "args": worker_args_from_train_args(cmd),
                                        "keep": False,
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                        else:
                            print("  " + subprocess.list2cmdline(cmd), flush=True)
                        continue

                    if settings.use_worker:
                        if worker is None:
                            raise RuntimeError("worker client is not open")
                        started = time.perf_counter()
                        payload = worker.request(
                            "trial",
                            {
                                "args": worker_args_from_train_args(cmd),
                                "keep": False,
                            },
                            prefix=tuner.paint(color, f"{display_prefix} ", "magenta"),
                        )
                        elapsed = time.perf_counter() - started
                        metric = tuner.metric_from_worker_payload(payload)
                        score = metric_score(metric, settings.metric)
                        checkpoint = Path("")
                    else:
                        code, elapsed = tuner.run_command(
                            cmd,
                            log_path,
                            stream=not args.no_stream_child_output,
                            stream_prefix=tuner.paint(color, f"{display_prefix} ", "magenta"),
                        )
                        if code != 0:
                            tuner.append_csv(
                                summary_path,
                                SUMMARY_FIELDS,
                                {
                                    "trial": trial,
                                    "status": "failed",
                                    "test_value_accuracy": "",
                                    "test_value_loss": "",
                                    "quantized_value_accuracy": "",
                                    "quantized_value_loss": "",
                                    "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
                                    "selection_metric": "",
                                    "checkpoint": "",
                                },
                            )
                            raise RuntimeError(f"trial {trial} failed; see {log_path}")
                        row = tuner.latest_summary_row(out_dir)
                        metric = tuner.metric_from_summary_row(row)
                        score = metric_score(metric, settings.metric)
                        checkpoint = (
                            tuner.latest_checkpoint_dir(out_dir)
                            if settings.keep_all_trials or args.keep_temp
                            else Path("")
                        )

                    result = TrialResult(
                        trial=trial,
                        params=params,
                        metric=metric,
                        score=score,
                        checkpoint=checkpoint,
                        output_dir=out_dir,
                    )
                    completed.append(result)

                    is_generation_best = is_better_score(
                        score,
                        generation_best.score if generation_best is not None else None,
                        settings.lower_is_better,
                    )
                    if is_generation_best:
                        generation_best = result
                    if (
                        not settings.use_worker
                        and not settings.keep_all_trials
                        and not args.keep_temp
                        and out_dir.exists()
                    ):
                        tuner.remove_dir_quiet(out_dir)
                    if is_better_score(score, best.score if best is not None else None, settings.lower_is_better):
                        best = result

                    keep_non_best_checkpoint = (
                        (not settings.use_worker) and (settings.keep_all_trials or args.keep_temp)
                    )
                    summary_checkpoint = str(result.checkpoint) if keep_non_best_checkpoint else ""
                    tuner.append_csv(
                        summary_path,
                        SUMMARY_FIELDS,
                        {
                            "trial": trial,
                            "status": "finished",
                            "test_value_accuracy": tuner.format_float(metric.test_acc),
                            "test_value_loss": tuner.format_float(metric.test_loss),
                            "quantized_value_accuracy": tuner.format_float(metric.qacc),
                            "quantized_value_loss": tuner.format_float(metric.qloss),
                            "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
                            "selection_metric": tuner.format_float(score),
                            "checkpoint": summary_checkpoint,
                        },
                    )
                    tuner.event(
                        color,
                        f"{display_prefix} END",
                        (
                            f"selection_metric={score:.9g} {tuner.metric_status_text(metric)} "
                            f"elapsed={elapsed:.1f}s generation_best={'yes' if is_generation_best else 'no'}"
                        ),
                        "green" if is_generation_best else "cyan",
                    )
                    state = {
                        "version": SETTINGS_VERSION,
                        "next_trial": trial + 1,
                        "current_checkpoint": str(current_checkpoint) if current_checkpoint else None,
                        "generation_best_trial": generation_best.trial if generation_best else None,
                        "generation_best_selection_metric": generation_best.score if generation_best else None,
                        "pending_commit_generation": generation if trial == generation_last_trial else None,
                        "pending_commit_source": settings.commit_source if trial == generation_last_trial else None,
                        "pending_commit_params": None,
                        "best_trial": best.trial if best else None,
                        "best_selection_metric": best.score if best else None,
                        "best_commit_selection_metric": best_commit_score,
                        "best_checkpoint": str(best_dir) if best_commit_score is not None else None,
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

                if args.dry_run:
                    continue
                if generation_best is None:
                    raise RuntimeError(f"generation {generation} produced no completed trials")

                generation_completed = generation_results(settings, completed, generation)
                rec = write_recommendation(recommendation_path, specs, completed, settings)
                rec_params = rec["recommended"]["parameters"]
                pending_matches = state.get("pending_commit_generation") is not None and int(
                    state.get("pending_commit_generation")
                ) == generation
                pending_params = state.get("pending_commit_params") if pending_matches else None
                if isinstance(pending_params, dict):
                    commit_params = {str(k): float(v) for k, v in pending_params.items()}
                    commit_source = str(state.get("pending_commit_source") or settings.commit_source)
                elif settings.commit_source == "recommended":
                    commit_params = rec_params
                    commit_source = "recommended"
                else:
                    commit_params = generation_best.params
                    commit_source = "best"
                commit_source_trial = generation_best.trial if commit_source == "best" else None

                tuner.event(
                    color,
                    f"[GEN {generation} SELECT]",
                    (
                        f"best_trial={generation_best.trial} best_selection_metric={generation_best.score:.9g} "
                        f"commit_source={commit_source} completed_trials={len(generation_completed)}"
                    ),
                    "green",
                )

                state = {
                    "version": SETTINGS_VERSION,
                    "next_trial": generation_last_trial + 1,
                    "current_checkpoint": str(current_checkpoint) if current_checkpoint else None,
                    "pending_commit_generation": generation,
                    "pending_commit_source": commit_source,
                    "pending_commit_params": commit_params,
                    "generation_best_trial": generation_best.trial,
                    "generation_best_selection_metric": generation_best.score,
                    "best_trial": best.trial if best else None,
                    "best_selection_metric": best.score if best else None,
                    "best_commit_selection_metric": best_commit_score,
                    "best_checkpoint": str(best_dir) if best_commit_score is not None else None,
                    "recommended_parameters": str(recommendation_path),
                    "settings_file": str(args.settings_file.resolve()),
                }
                write_json(state_path, state)

                commit_metric: tuner.Metric | None = None
                commit_score: float | None = None
                pending_has_checkpoint = (pending_commit_dir / "state.bin").exists()
                if pending_matches and pending_has_checkpoint:
                    commit_metric = tuner.latest_learn_log_metric(pending_commit_dir)
                    commit_score = metric_score(commit_metric, settings.metric)
                    tuner.event(
                        color,
                        f"[GEN {generation} COMMIT RESUME]",
                        f"using existing {pending_commit_dir} selection_metric={commit_score:.9g}",
                        "yellow",
                    )
                else:
                    tuner.remove_dir_quiet(pending_commit_dir)
                    commit_out_dir = trials_root / f"gen{generation:04d}-commit"
                    if commit_out_dir.exists():
                        tuner.remove_dir_quiet(commit_out_dir)
                    tuner.event(
                        color,
                        f"[GEN {generation} COMMIT START]",
                        (
                            f"source={commit_source} "
                            f"from={current_checkpoint or 'scratch'} "
                            f"params="
                            + " ".join(f"{k}={v:.9g}" for k, v in sorted(commit_params.items()))
                        ),
                        "magenta",
                    )
                    commit_cmd = train_args(
                        run,
                        commit_params,
                        current_checkpoint,
                        commit_out_dir,
                        settings,
                        trial_sbs,
                        include_initial_state=not settings.use_worker,
                        save_checkpoint=True,
                    )
                    if settings.use_worker:
                        if worker is None:
                            raise RuntimeError("worker client is not open")
                        started = time.perf_counter()
                        payload = worker.request(
                            "trial",
                            {
                                "args": worker_args_from_train_args(commit_cmd),
                                "keep": True,
                            },
                            prefix=tuner.paint(color, f"[GEN {generation} COMMIT] ", "magenta"),
                        )
                        elapsed = time.perf_counter() - started
                        commit_metric = tuner.metric_from_worker_payload(payload)
                        commit_score = metric_score(commit_metric, settings.metric)
                        save_args = worker_args_from_train_args(
                            train_args(
                                run,
                                commit_params,
                                None,
                                runner_root / "worker-save-output",
                                settings,
                                trial_sbs,
                                include_initial_state=False,
                                save_checkpoint=True,
                            )
                        )
                        worker.request(
                            "save",
                            {
                                "args": save_args,
                                "dir": str(pending_commit_dir),
                                "epoch": generation,
                                "superbatch": trial_sbs,
                            },
                            prefix=tuner.paint(color, f"[GEN {generation} SAVE] ", "green"),
                        )
                    else:
                        code, elapsed = tuner.run_command(
                            commit_cmd,
                            log_dir / f"generation{generation:04d}-commit.stdout.log",
                            stream=not args.no_stream_child_output,
                            stream_prefix=tuner.paint(color, f"[GEN {generation} COMMIT] ", "magenta"),
                        )
                        if code != 0:
                            raise RuntimeError(
                                f"generation {generation} commit run failed; "
                                f"see {log_dir / f'generation{generation:04d}-commit.stdout.log'}"
                            )
                        row = tuner.latest_summary_row(commit_out_dir)
                        commit_metric = tuner.metric_from_summary_row(row)
                        commit_score = metric_score(commit_metric, settings.metric)
                        move_dir_replace(tuner.latest_checkpoint_dir(commit_out_dir), pending_commit_dir)
                        if not args.keep_temp:
                            tuner.remove_dir_quiet(commit_out_dir)
                    tuner.event(
                        color,
                        f"[GEN {generation} COMMIT END]",
                        (
                            f"source={commit_source} selection_metric={commit_score:.9g} "
                            f"{tuner.metric_status_text(commit_metric)} elapsed={elapsed:.1f}s"
                        ),
                        "green",
                    )

                assert commit_metric is not None
                assert commit_score is not None
                move_dir_replace(pending_commit_dir, current_dir)
                current_checkpoint = current_dir

                tuner.append_csv(
                    summary_path,
                    SUMMARY_FIELDS,
                    {
                        "trial": f"gen{generation}-commit",
                        "status": f"commit_{commit_source}",
                        "test_value_accuracy": tuner.format_float(commit_metric.test_acc),
                        "test_value_loss": tuner.format_float(commit_metric.test_loss),
                        "quantized_value_accuracy": tuner.format_float(commit_metric.qacc),
                        "quantized_value_loss": tuner.format_float(commit_metric.qloss),
                        "parameters": json.dumps(commit_params, ensure_ascii=False, sort_keys=True),
                        "selection_metric": tuner.format_float(commit_score),
                        "checkpoint": str(current_dir),
                    },
                )

                if is_better_score(commit_score, best_commit_score, settings.lower_is_better):
                    tuner.copy_dir_replace(current_dir, best_dir)
                    best_commit_score = commit_score

                next_trial = generation_last_trial + 1
                state = {
                    "version": SETTINGS_VERSION,
                    "next_trial": next_trial,
                    "current_checkpoint": str(current_checkpoint),
                    "current_generation": generation,
                    "current_generation_trial": commit_source_trial,
                    "current_generation_selection_metric": commit_score,
                    "current_generation_commit_source": commit_source,
                    "generation_best_trial": None,
                    "generation_best_selection_metric": None,
                    "pending_commit_generation": None,
                    "pending_commit_source": None,
                    "pending_commit_params": None,
                    "best_trial": best.trial if best else None,
                    "best_selection_metric": best.score if best else None,
                    "best_commit_selection_metric": best_commit_score,
                    "best_checkpoint": str(best_dir) if best_commit_score is not None else None,
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
                if settings.use_worker and worker is not None and generation < settings.generations:
                    next_trial_sbs = settings.trial_sbs_for_generation(generation + 1)
                    open_worker_session(
                        worker=worker,
                        run=run,
                        specs=specs,
                        current_checkpoint=current_checkpoint,
                        runner_root=runner_root,
                        settings=settings,
                        trial_sbs=next_trial_sbs,
                        color=color,
                    )
        finally:
            if worker is not None:
                worker.close()
        return 0
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        print("hint: rerun with --debug is not available for this lightweight runner; inspect logs/ for child stdout.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
