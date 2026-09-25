#!/usr/bin/env python3
"""Sequential, independent BulletOu parameter-grid runs and CSV aggregation.

See docs/ja/advanced/grid-search.md (English: docs/en/advanced/grid-search.md).
Only the Python standard library is required. This runner does not tune a
survivor, reset optimizers, prune trials, or delete checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from bulletou_csv import SUMMARY_CSV_NAME


METRICS = (
    "test_value_accuracy", "test_value_loss",
    "quantized_value_accuracy", "quantized_value_loss",
)
EXTREMA = ("max_acc", "min_loss", "max_qacc", "min_qloss")
PLURAL_OPTIONS = {
    "lrs": "lr", "lr_mins": "lr_min",
    "wrm_target_scalings": "wrm_target_scaling",
    "wrm_target_epsilons": "wrm_target_epsilon",
    "wrm_in_scalings": "wrm_in_scaling",
    "wrm_nnue2scores": "wrm_nnue2score",
    "batch_sizes": "batch_size", "batches_per_updates": "batches_per_update",
    "factorizers": "sfnn_factorizer", "loss_pow_exps": "loss_pow_exp",
}
OUTPUT_KEYS = {"output", "output_folder", "tag", "resume", "no_resume"}
FORBIDDEN_GRID = OUTPUT_KEYS | {
    "settings_file", "initial_state", "initial_dataloader_pos", "max_epochs",
    "cuda_cpp_train_steps",
}
COMMON_COLUMNS = (
    "nnue_bn_ft", "nnue_bn_l1", "nnue_bn_l2", "nnue_bn_gamma", "nnue_bn_beta", "nnue_bn_momentum", "nnue_bn_epsilon",
    "sfnn_bn_ft", "sfnn_bn_l1", "sfnn_bn_l2", "sfnn_bn_qat", "sfnn_bn_gamma", "sfnn_bn_beta", "sfnn_bn_momentum", "sfnn_bn_epsilon",
    "arch", "lr", "lr_min", "lr_schedule", "warmup_sb", "batch_size", "batches_per_update",
    "positions_per_superbatch", "superbatches", "sfnn_factorizer",
    "sfnn_ft_saturation_penalty", "sfnn_ft_saturation_rate", "sfnn_ft_saturation_patience",
    "sfnn_factorizer_alpha", "sfnn_norm_loss_strength", "sfnn_l2_l3_center", "sfnn_l1_center", "sfnn_l1_effective_weight_clip", "sfnn_init_l2_l3_glorot", "loss_bce_with_logits", "bce_error_weight_k", "wrm_nnue2score", "wrm_in_scaling",
    "wrm_target_scaling", "wrm_target_epsilon", "wrm_in_offset", "wrm_target_offset", "loss_pow_exp",
)
MANIFEST = "grid-manifest.json"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: JSON object required")
    return obj


def atomic_json(path: Path, obj: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    temp.replace(path)


def key_name(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key):
        raise ValueError(f"invalid BulletOu option name: {key!r}")
    return key.replace("-", "_")


def scalar(value):
    if isinstance(value, (list, dict)):
        raise ValueError("grid/settings values must be null, bool, number or string")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite grid/settings value")
    return value


def parse_value(text: str):
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    return scalar(value)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    p.add_argument("--settings-file", type=Path, help="Common bulletou-settings.json; not tuning-settings.json")
    p.add_argument("--output-folder", type=Path, required=True, help="Dedicated grid root containing trials/ and grid_summary.csv")
    p.add_argument("--exe", type=Path, default=Path(__file__).resolve().parent / "target/release/examples/bulletou.exe")
    p.add_argument("--checkpoint", type=Path, help="Common initial checkpoint DIRECTORY (state.bin + dataloader_pos.txt)")
    p.add_argument("--epochs", type=int, nargs="+", help="Epochs to report, e.g. 1 2 5; train each condition once through max=5")
    for option in PLURAL_OPTIONS:
        p.add_argument("--" + option.replace("_", "-"), nargs="+", metavar="VALUE",
                       help=f"Grid values for BulletOu {PLURAL_OPTIONS[option]} (Cartesian product)")
    p.add_argument("--grid", action="append", nargs="+", default=[], metavar="KEY_OR_VALUE",
                   help="Repeatable generic axis: --grid sfnn_factorizer_alpha shared=0.5 shared=1.0")
    p.add_argument("--summary-csv", type=Path)
    p.add_argument("--summary-only", action="store_true", help="Rebuild CSV from the manifest and existing logs; no trainer needed")
    p.add_argument("--dry-run", action="store_true", help="Validate and display the plan without writing files or training")
    p.add_argument("--resume", action="store_true", help="Resume selected existing grid conditions; increase --epochs to extend their total epoch budget")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--verbose", action="store_true", help="Pass --verbose to BulletOu to display qstats/qstats-unit; CSV measurements are unchanged")
    a = p.parse_args(argv)
    if a.epochs is not None and (not a.epochs or min(a.epochs) < 1):
        p.error("--epochs must contain positive integers")
    if not a.summary_only and a.settings_file is None:
        p.error("--settings-file is required unless --summary-only is used")
    if a.summary_only and (a.dry_run or a.resume or a.checkpoint or a.grid
                           or any(getattr(a, key) for key in PLURAL_OPTIONS)):
        p.error("--summary-only cannot be combined with execution/grid options")
    return a


def collect_axes(args) -> dict:
    axes = {}
    entries = [(key, getattr(args, option)) for option, key in PLURAL_OPTIONS.items()
               if getattr(args, option) is not None]
    for entry in args.grid:
        if len(entry) < 2:
            if entry:
                option = entry[0]
                raise ValueError(
                    f"--grid {option}: missing value(s) after '{option}'. "
                    f"Use --grid {option} VALUE [VALUE ...]. "
                    f"For a boolean option, use --grid {option} true "
                    f"(enable only) or --grid {option} false true (A/B comparison)."
                )
            raise ValueError("--grid: missing option name and values; use --grid OPTION VALUE [VALUE ...]")
        entries.append((key_name(entry[0]), entry[1:]))
    for key, raw_values in entries:
        if key in FORBIDDEN_GRID:
            raise ValueError(f"--grid {key}: lifecycle/initial-state option cannot be varied")
        if key in axes:
            raise ValueError(f"grid axis specified twice: {key}")
        values = [parse_value(value) for value in raw_values]
        # Reject duplicates rather than silently run a condition twice.
        encoded = [json.dumps(v, sort_keys=True) for v in values]
        if len(set(encoded)) != len(encoded):
            raise ValueError(f"duplicate value in grid axis {key}")
        axes[key] = values
    if not axes:
        raise ValueError("specify at least one grid axis (e.g. --lrs 0.0001 0.0002)")
    return axes


def positive_int(settings: dict, key: str) -> int:
    value = settings.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} must be an explicit positive integer in the settings")
    return value


EPOCH_SETTING_KEYS = {
    "sfnn_ft_saturation_penalty", "sfnn_ft_saturation_rate", "sfnn_ft_saturation_patience",
    "lr", "lr_min", "batches_per_update", "sfnn_qat_l1", "sfnn_freeze_l1", "sfnn_l2_l3_center", "sfnn_l1_center", "sfnn_l1_effective_weight_clip",
    "sfnn_l1_lr_mult", "sfnn_norm_loss_strength", "sfnn_saturation_penalty",
    "sfnn_saturation_threshold", "optimizer_weight_clip", "optimizer_weight_decay",
    "bce_error_weight_k",
}


def resolve_epoch_settings(settings: dict, epoch: int) -> dict:
    epoch = max(1, epoch)  # Warmup epoch 0 inherits epoch 1 controls.
    resolved = dict(settings)
    for key, value in settings.items():
        if not isinstance(value, dict):
            continue
        if key not in EPOCH_SETTING_KEYS:
            raise ValueError(f"epoch schedule is not supported for {key}")
        if "epoch1" not in value:
            raise ValueError(f"{key}: epoch schedule requires epoch1")
        for name, item in value.items():
            if not re.fullmatch(r"epoch[1-9][0-9]*", name):
                raise ValueError(f"{key}: invalid epoch key {name!r}")
            if key in ("sfnn_qat_l1", "sfnn_freeze_l1", "sfnn_l1_center", "sfnn_l2_l3_center", "sfnn_l1_effective_weight_clip"):
                if type(item) is not bool:
                    raise ValueError(f"{key}.{name} must be true/false")
            elif type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(f"{key}.{name} must be a finite number")
        name = max((n for n in value if int(n[5:]) <= epoch), key=lambda n: int(n[5:]))
        resolved[key] = value[name]
    return resolved


def check_settings(settings: dict) -> None:
    if any(isinstance(v, dict) for v in settings.values()):
        resolve_epoch_settings(settings, 1)  # Validate keys/types before parsing boundaries.
        boundaries = {1} | {int(n[5:]) for v in settings.values() if isinstance(v, dict) for n in v}
        for epoch in sorted(boundaries):
            check_settings(resolve_epoch_settings(settings, epoch))
        return
    for key, value in settings.items():
        scalar(value)
        if key == "settings_file":
            raise ValueError("nested settings_file is not supported")
    if settings.get("backend", "cuda-cpp") != "cuda-cpp":
        raise ValueError("grid_search.py currently supports the cuda-cpp production schedule")
    if settings.get("cuda_cpp_train_steps") is not None:
        raise ValueError("use superbatches/max_epochs, not cuda_cpp_train_steps")
    for key in ("teacher", "test_teacher", "arch"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f"{key} must be specified in common settings")
    for key in ("max_epochs", "superbatches"):
        positive_int(settings, key)
    warmup = settings.get("warmup_sb", 0)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup_sb must be a nonnegative integer")
    if warmup and settings.get("lr_schedule", "step") == "plateau":
        raise ValueError("warmup_sb supports step/geometric/cos, not plateau")
    test_positions = settings.get("test_positions")
    if test_positions not in (None, "all") and (type(test_positions) is not int or test_positions < 1):
        raise ValueError("test_positions must be a positive integer or 'all'; omission/null also uses all validation positions")
    for key in ("batch_size", "batches_per_update", "positions_per_superbatch"):
        if key in settings:
            positive_int(settings, key)
    for key in ("lr", "lr_min", "wrm_target_scaling", "wrm_in_scaling", "wrm_nnue2score"):
        if key in settings and (type(settings[key]) not in (float, int) or settings[key] <= 0):
            raise ValueError(f"{key} must be positive")
    k = settings.get("bce_error_weight_k", 0)
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0:
        raise ValueError("bce_error_weight_k must be finite and >= 0")
    if k != 0 and not settings.get("loss_bce_with_logits"):
        raise ValueError("bce_error_weight_k requires loss_bce_with_logits: true")
    if settings.get("loss_bce_with_logits"):
        if settings.get("loss_sigmoid_mse") or settings.get("win_rate_model"):
            raise ValueError("loss_bce_with_logits conflicts with loss_sigmoid_mse / win_rate_model")
        if settings.get("wrm_in_offset", 270) != 0:
            raise ValueError("loss_bce_with_logits requires wrm_in_offset: 0")
        if settings.get("loss_pow_exp", 2) != 2:
            raise ValueError("loss_pow_exp does not apply to BCE; leave it at its default 2")
    if "wrm_target_epsilon" in settings:
        epsilon = settings["wrm_target_epsilon"]
        if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or not 0 <= epsilon < 0.5:
            raise ValueError("wrm_target_epsilon must be finite and 0 <= epsilon < 0.5")
        if epsilon and settings.get("loss_sigmoid_mse"):
            raise ValueError("wrm_target_epsilon requires WRM loss")
    if "lr" in settings and "lr_min" in settings and settings["lr_min"] > settings["lr"]:
        raise ValueError("lr_min > lr in a grid combination; choose compatible lists (no combinations are silently skipped)")
    for key in ("validation_rate", "quantized_validation_rate"):
        if key in settings and (type(settings[key]) is not int or settings[key] < -1):
            raise ValueError(f"{key} must be -1, 0 or a positive integer")


def make_plan(args) -> dict:
    template = {}
    for raw_key, value in read_json(args.settings_file).items():
        key = key_name(raw_key)
        if key in template:
            raise ValueError(f"duplicate normalized setting: {key}")
        template[key] = value if isinstance(value, dict) else scalar(value)
    for key in OUTPUT_KEYS:
        template.pop(key, None)
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve()
        for name in ("state.bin", "dataloader_pos.txt"):
            if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
                raise ValueError(f"common checkpoint requires a nonempty {checkpoint / name}")
        template["initial_state"] = str(checkpoint / "state.bin")
        template["initial_dataloader_pos"] = str(checkpoint / "dataloader_pos.txt")
    if args.epochs:
        template["max_epochs"] = max(args.epochs)
    check_settings(template)
    epochs = sorted(set(args.epochs or range(1, template["max_epochs"] + 1)))
    axes = collect_axes(args)
    root = args.output_folder.resolve()
    count = math.prod(len(values) for values in axes.values())
    if count > 10000:
        raise ValueError(f"grid has {count} conditions; split into grids of at most 10000")
    trials = []
    for index, values in enumerate(itertools.product(*axes.values()), 1):
        params = dict(zip(axes, values))
        settings = {**template, **params}
        check_settings(settings)
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        label = "_".join(f"{k}={v}" for k, v in params.items())
        label = re.sub(r"[^A-Za-z0-9_.=+-]", "-", label)[:64].rstrip(".")
        name = f"trial{index:04}-{label}-{digest}"
        settings["output"] = str(root / "trials" / name)
        # Explicit --output preserves this exact folder; tag is metadata only.
        settings["tag"] = name
        trials.append({"id": index, "name": name, "parameters": params, "settings": settings})
    return {"version": 1, "exe": str(args.exe.resolve()), "cwd": str(Path.cwd()),
            "report_epochs": epochs, "axes": axes, "trials": trials}


def preflight_exe(plan: dict) -> None:
    exe = Path(plan["exe"])
    if not exe.is_file():
        raise ValueError(f"BulletOu executable not found: {exe}")
    result = subprocess.run([str(exe), "--help"], cwd=plan["cwd"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise ValueError(f"bulletou --help failed: {result.stderr or result.stdout}")
    options = set(re.findall(r"(?m)^\s+(?:-[A-Za-z],\s+)?--([a-z][a-z0-9-]*)", result.stdout))
    needed = {key.replace("_", "-") for trial in plan["trials"] for key in trial["settings"]}
    missing = sorted((needed | {"settings-file", "resume"}) - options)
    if missing:
        raise ValueError("this bulletou executable does not support: " + ", ".join("--" + k for k in missing))


@contextmanager
def grid_lock(root: Path):
    """OS releases the lock even if this runner crashes; never touches other jobs."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "grid.lock").open("a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"another grid runner/summary writer is using {root}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def trial_dir(root: Path, trial: dict) -> Path:
    name = trial["name"]
    if Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("invalid trial directory in manifest")
    return root / "trials" / name


def log_rows(directory: Path, *, live=False) -> list[dict]:
    path = directory / SUMMARY_CSV_NAME
    if not path.is_file():
        return []
    # Live readers must not accept a partially appended final record, even if
    # that prefix happens to be syntactically valid CSV.
    text = path.read_text(encoding="utf-8-sig")
    if live and text and not text.endswith("\n"):
        raise ValueError(f"{path}: waiting for a complete CSV record")
    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f, strict=True)
        if not {"epoch", "superbatch"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected ordinary BulletOu epoch/superbatch CSV")
        points = {}
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}: incomplete CSV row; original file was not modified")
            point = (int(row["epoch"]), int(row["superbatch"]))
            points[point] = row
        return [points[point] for point in sorted(points)]


def numeric(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def checkpoint_path(directory: Path, row: dict) -> str:
    value = row.get("checkpoint", "")
    if not value or value == "-":
        return ""
    path = (directory / value).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("checkpoint in summary must be inside its trial directory")
    state = path / "state.bin"
    return str(path) if state.is_file() and state.stat().st_size > 0 else ""


def is_complete(directory: Path, trial: dict, state: dict, rows=None) -> bool:
    if rows is None:
        rows = log_rows(directory)
    epoch = trial["settings"]["max_epochs"]
    group = [r for r in rows if int(r["epoch"]) == epoch]
    target = (epoch, epoch_settings(trial, epoch, group)["superbatches"])
    last = next((r for r in reversed(rows) if (int(r["epoch"]), int(r["superbatch"])) == target), None)
    # A saved final row can recover completion after a runner interruption just
    # after the child finished, before grid-state.json was updated.
    return last is not None and (state.get("status") == "done" or (
        state.get("status") != "failed" and bool(checkpoint_path(directory, last))))


def has_resume_checkpoint(directory: Path) -> bool:
    return any(p.parent.name.isdigit() and p.stat().st_size > 0
               and (p.parent / "dataloader_pos.txt").is_file()
               and (p.parent / "dataloader_pos.txt").stat().st_size > 0
               for p in directory.glob("*/state.bin"))


def restart_unsaved_trials(root: Path, plan: dict, selected: set[int]) -> None:
    """Under the grid lock, archive unsaved attempts before parsing their logs.

    A killed process can leave a partial CSV row or stale resume-config. Move
    the whole attempt intact so neither can contaminate a fresh native run.
    Completed conditions and all usable native checkpoints remain untouched.
    """
    for trial in plan["trials"]:
        if trial["id"] not in selected:
            continue
        directory = trial_dir(root, trial)
        if not directory.exists() or not any(directory.iterdir()) or has_resume_checkpoint(directory):
            continue
        check_trial_settings_file(directory, trial)
        state_path = directory / "grid-state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        try:
            if is_complete(directory, trial, state):
                continue
        except (ValueError, csv.Error):
            pass  # Preserve the truncated/malformed unsaved log in the archive.
        archive = root / "interrupted-runs" / f"{directory.name}-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
        resolved_root = root.resolve()
        if (not directory.resolve().is_relative_to(resolved_root)
                or not archive.resolve().is_relative_to(resolved_root)):
            raise ValueError("unsafe unsaved-trial archive path; nothing was moved")
        archive.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archive)
        directory.mkdir()
        trial["initial_settings"] = copy.deepcopy(trial["settings"])
        trial.pop("epoch_settings", None)
        atomic_json(directory / "grid-state.json", {
            "status": "pending", "elapsed_seconds": state.get("elapsed_seconds", 0),
            "restarted_from_archive": str(archive),
        })
        print(f"[RESTART] trial={trial['id']} no resumable checkpoint; restarting from the original initial state (epoch 1), not the interrupted sb", flush=True)
        print(f"[ARCHIVE] previous attempt preserved: {archive}", flush=True)


def training_identity(settings: dict) -> dict:
    """Compare launch files while allowing the historical epoch budget."""
    return {k: v for k, v in settings.items() if k not in {"output", "tag", "max_epochs"}}


def settings_diff(saved: dict, requested: dict) -> str:
    """Show all changed, added and removed settings."""
    def value(settings, key):
        return json.dumps(settings[key], ensure_ascii=False) if key in settings else "<not specified>"
    return "\n".join(
        f"  {key}: saved={value(saved, key)}, requested={value(requested, key)}"
        for key in sorted(saved.keys() | requested.keys())
        if key not in saved or key not in requested or saved[key] != requested[key]
    )


def changed_setting_keys(saved: dict, requested: dict) -> list[str]:
    return sorted(key for key in saved.keys() | requested.keys()
                  if key not in saved or key not in requested or saved[key] != requested[key])


def changed_setting_columns(plan: dict) -> list[str]:
    columns = list(plan.get("changed_setting_columns", []))
    # Recover changes recorded by older manifests, including removed settings.
    for trial in plan["trials"]:
        for previous in [trial.get("initial_settings", trial["settings"]),
                         *[e["settings"] for e in trial.get("epoch_settings", {}).values()]]:
            columns.extend(changed_setting_keys(previous, trial["settings"]))
    return list(dict.fromkeys(columns))


def check_trial_settings_file(directory: Path, trial: dict) -> None:
    path = directory / "bulletou-settings.json"
    if not path.exists():
        return
    saved = read_json(path)
    expected = trial.get("initial_settings", trial["settings"])
    # Keep the original launch file immutable; extensions use resume-settings.
    if (training_identity(saved) != training_identity(expected)
            or saved.get("output") != expected.get("output") or saved.get("tag") != expected.get("tag")
            or type(saved.get("max_epochs")) is not int
            or not 1 <= saved["max_epochs"] <= expected["max_epochs"]):
        raise ValueError(f"trial settings were edited: {path}; refusing to overwrite")


def rows_digest(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def epoch_settings(trial: dict, epoch: int, rows: list[dict]) -> dict:
    saved = trial.get("epoch_settings", {}).get(str(epoch))
    # A rollback/retraining changes the log: never label new results with old settings.
    if saved is not None and saved["rows_digest"] == rows_digest(rows):
        return saved["settings"]
    return trial["settings"]


def remember_completed_epoch_settings(directory: Path, trial: dict) -> None:
    try:
        rows = log_rows(directory)
    except (ValueError, csv.Error):
        if has_resume_checkpoint(directory):
            raise
        return  # The unsaved attempt will be archived and restarted.
    for epoch in sorted({int(r["epoch"]) for r in rows}):
        group = [r for r in rows if int(r["epoch"]) == epoch]
        settings = epoch_settings(trial, epoch, group)
        if int(group[-1]["superbatch"]) == (settings.get("warmup_sb", 0) if epoch == 0 else settings["superbatches"]):
            trial.setdefault("epoch_settings", {})[str(epoch)] = {
                "rows_digest": rows_digest(group), "settings": copy.deepcopy(settings),
            }


def plan_resume(root: Path, stored: dict, requested: dict) -> tuple[dict, set[int]]:
    """Read-only reconciliation. Preserve IDs, folders and unselected conditions."""
    if (stored.get("version") != 1 or any(stored.get(k) != requested.get(k) for k in ("exe", "cwd"))
            or set(stored.get("axes", {})) != set(requested["axes"])):
        raise ValueError("existing grid manifest differs: resume requires the same executable path, cwd and grid axis names")
    merged = copy.deepcopy(stored)
    merged["changed_setting_columns"] = changed_setting_columns(stored)
    selected = set()
    report_epochs = set(stored["report_epochs"]) | set(requested["report_epochs"])
    for candidate in requested["trials"]:
        matches = [t for t in merged["trials"] if t["parameters"] == candidate["parameters"]]
        if not matches:
            trial = copy.deepcopy(candidate)
            trial["id"] = max(t["id"] for t in merged["trials"]) + 1
            trial["name"] = f"trial{trial['id']:04}-" + candidate["name"].split("-", 1)[1]
            trial["settings"]["output"] = str(root / "trials" / trial["name"])
            trial["settings"]["tag"] = trial["name"]
            if trial_dir(root, trial).exists():
                raise ValueError(f"new condition output already exists: {trial_dir(root, trial)}")
            merged["trials"].append(trial)
            for key, value in trial["parameters"].items():
                if value not in merged["axes"][key]:
                    merged["axes"][key].append(value)
            selected.add(trial["id"])
            continue
        if len(matches) != 1:
            raise ValueError(f"existing grid manifest differs: no unique existing condition for {candidate['parameters']}")
        trial = matches[0]
        old, new = trial["settings"]["max_epochs"], candidate["settings"]["max_epochs"]
        defaults = {"backend": "cuda-cpp", "no_ft_factorize": False}
        incompatible = [k for k in ("arch", "backend", "no_ft_factorize")
                        if trial["settings"].get(k, defaults.get(k)) != candidate["settings"].get(k, defaults.get(k))]
        if incompatible:
            raise ValueError(
                f"trial {trial['id']}: checkpoint-incompatible settings changed: {', '.join(incompatible)}\n"
                + settings_diff({k: trial["settings"].get(k, defaults.get(k)) for k in incompatible},
                                {k: candidate["settings"].get(k, defaults.get(k)) for k in incompatible})
                + "\nUse a new grid root for a different network layout. Nothing was overwritten.")
        if new < old:
            raise ValueError(f"cannot reduce trial {trial['id']} max_epochs from {old} to {new}; use --epochs {old} or higher")
        directory = trial_dir(root, trial)
        check_trial_settings_file(directory, trial)
        updated = {**candidate["settings"], "output": trial["settings"]["output"], "tag": trial["settings"]["tag"]}
        if trial["settings"] != updated:
            merged["changed_setting_columns"] = list(dict.fromkeys([
                *merged["changed_setting_columns"], *changed_setting_keys(trial["settings"], updated),
            ]))
            remember_completed_epoch_settings(directory, trial)
            trial.setdefault("initial_settings", copy.deepcopy(trial["settings"]))
            trial["settings"] = updated
        if new > old:
            report_epochs.update(range(old + 1, new + 1))
        selected.add(trial["id"])
    merged["report_epochs"] = sorted(report_epochs)
    # IDs identify persistent outputs; list order follows this invocation's CLI.
    ordered = [next(t for t in merged["trials"] if t["parameters"] == c["parameters"])
               for c in requested["trials"]]
    merged["trials"] = ordered + [t for t in merged["trials"] if t["id"] not in selected]
    return merged, selected


def summarize(root: Path, plan: dict, epochs=None, *, trial_rows=None) -> tuple[list[str], list[dict]]:
    parameter_columns = list(dict.fromkeys([*plan["axes"], *COMMON_COLUMNS]))
    all_settings = [settings for trial in plan["trials"]
                    for settings in [trial["settings"],
                                     *[e["settings"] for e in trial.get("epoch_settings", {}).values()]]]
    parameter_columns = [key for key in parameter_columns
                         if any(key in settings for settings in all_settings)]
    parameter_columns = list(dict.fromkeys([*parameter_columns, *changed_setting_columns(plan)]))
    parameter_columns = list(dict.fromkeys([*parameter_columns,
        *(key for settings in all_settings for key, value in settings.items() if isinstance(value, dict))]))
    fields = ["trial", "epoch", "superbatch", *METRICS, *EXTREMA,
              *[name + "_sb" for name in EXTREMA], "positions",
              *parameter_columns, "status", "trial_status", "output_dir", "checkpoint"]
    # Generic grid keys must not duplicate metric/status columns.
    fields = list(dict.fromkeys(fields))
    result = []
    for trial in plan["trials"]:
        directory = trial_dir(root, trial)
        state_path = directory / "grid-state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        trial_status = state.get("status", "pending")
        rows = trial_rows[trial["id"]] if trial_rows is not None and trial["id"] in trial_rows else log_rows(directory)
        if is_complete(directory, trial, state, rows):
            trial_status = "done"
        elif trial_status == "done":
            trial_status = "incomplete"
        report_epochs = list(epochs or plan["report_epochs"])
        if trial["settings"].get("warmup_sb", 0) > 0 and 0 not in report_epochs:
            report_epochs = [0, *report_epochs]
        for epoch in report_epochs:
            if epoch > trial["settings"]["max_epochs"]:
                continue  # Unselected conditions were not extended.
            group = [row for row in rows if int(row["epoch"]) == epoch]
            last = group[-1] if group else {}
            settings = resolve_epoch_settings(epoch_settings(trial, epoch, group), epoch)
            closed = last and int(last["superbatch"]) == (settings.get("warmup_sb", 0) if epoch == 0 else settings["superbatches"])
            row = {key: settings.get(key, "") for key in parameter_columns}
            if not closed:
                row.update(trial=trial["id"], epoch=epoch,
                           status="incomplete" if trial_status == "done" else trial_status,
                           trial_status=trial_status, output_dir=str(directory))
                result.append(row)
                continue
            status = "done"
            row.update(trial=trial["id"], epoch=epoch, superbatch=last.get("superbatch", ""),
                       status=status, trial_status=trial_status,
                       output_dir=str(directory),
                       checkpoint=checkpoint_path(directory, last))
            for key in (*METRICS, "positions"):
                value = last.get(key, "")
                row[key] = value if numeric(value) is not None else ""
            for metric, name in zip(METRICS, EXTREMA):
                measured = [r for r in group if numeric(r.get(metric)) is not None]
                if measured:
                    choose = min if "loss" in metric else max
                    best = choose(measured, key=lambda r: numeric(r[metric]))
                    row[name], row[name + "_sb"] = best[metric], best["superbatch"]
            result.append(row)
    return fields, result


def write_summary(root: Path, plan: dict, path: Path, epochs=None, *, trial_rows=None) -> list[dict]:
    fields, rows = summarize(root, plan, epochs, trial_rows=trial_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
    return rows


@contextmanager
def live_summary_updates(root: Path, plan: dict, path: Path, trial: dict, *, interval=1.0):
    """Publish completed epochs while the child is alive, even if stdout is quiet.

    Stop/join before the main thread updates state or writes its final summary.
    CSV read/write failures are retried, never used to terminate training.
    """
    stop = threading.Event()
    directory = trial_dir(root, trial)

    def closed_rows(rows):
        return [r for r in rows if int(r["superbatch"]) == (
            trial["settings"].get("warmup_sb", 0) if int(r["epoch"]) == 0 else trial["settings"]["superbatches"])]

    previous = closed_rows(log_rows(directory))

    def watch():
        nonlocal previous
        warned = None
        while not stop.wait(interval):
            try:
                rows = log_rows(directory, live=True)
                closed = closed_rows(rows)
                if not closed or closed == previous:
                    continue
                # Use the same complete snapshot for detection and aggregation.
                write_summary(root, plan, path, trial_rows={trial["id"]: rows})
                previous = closed
                warned = None
                print(f"[SUMMARY] trial={trial['id']} completed_epoch={closed[-1]['epoch']} updated: {path}", flush=True)
            except (OSError, ValueError, csv.Error) as exc:
                message = str(exc)
                if message != warned:
                    print(f"[WARN] live summary update deferred; will retry: {message}", flush=True)
                    warned = message

    thread = threading.Thread(target=watch, name="grid-epoch-summary", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def print_leaders(rows: list[dict]) -> None:
    for epoch in sorted({r["epoch"] for r in rows}):
        complete = [r for r in rows if r["epoch"] == epoch and r["status"] == "done"]
        for metric in METRICS:
            measured = [r for r in complete if numeric(r[metric]) is not None]
            if measured:
                choose = min if "loss" in metric else max
                best = choose(measured, key=lambda r: numeric(r[metric]))
                print(f"[BEST] epoch={epoch} {metric}={best[metric]} trial={best['trial']} (epoch end)", flush=True)


def stop_child(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def trial_console_line(trial_id: int, line: str) -> str:
    text = f"[TRIAL {trial_id}] {line}"
    color_mode = os.environ.get("BULLETOU_COLOR", "auto").lower()
    color = ("NO_COLOR" not in os.environ and color_mode != "never"
             and (color_mode == "always" or (sys.stdout.isatty() and os.environ.get("TERM") != "dumb")))
    if color and re.match(r"\s*(?:WARN(?:ING)?:|\[WARN(?:ING)?\])", line):
        ending = "\n" if text.endswith("\n") else ""
        return "\x1b[1;33m" + text.rstrip("\n") + "\x1b[0m" + ending
    return text


def run_child(command: list[str], directory: Path, cwd: str, trial_id: int) -> tuple[int, float]:
    start = time.monotonic()
    # Append so interrupted attempts remain inspectable, including settings/build
    # banners. The child still writes its normal summary and checkpoints itself.
    with (directory / "stdout.log").open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n[COMMAND] " + subprocess.list2cmdline(command) + "\n")
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as proc:
            try:
                for line in proc.stdout:
                    print(trial_console_line(trial_id, line), end="", flush=True)
                    log.write(line)
                code = proc.wait()
            except BaseException:
                stop_child(proc)
                raise
    return code, time.monotonic() - start


def command_for(plan: dict, directory: Path, resume: bool, verbose: bool = False) -> list[str]:
    filename = "bulletou-resume-settings.json" if resume else "bulletou-run-settings.json"
    command = [plan["exe"], "--settings-file", str(directory / filename)]
    if resume:
        command.append("--resume")
    if verbose:
        command.append("--verbose")
    return command


def resume_settings(settings: dict) -> dict:
    # BulletOu rejects explicit initial-state plus --resume. The trial's own
    # resume-config/checkpoints now own both weights and dataloader position.
    return {key: value for key, value in settings.items()
            if key not in {"initial_state", "initial_dataloader_pos"}}


def record_settings_launch(directory: Path, trial: dict, resume: bool) -> None:
    path = directory / "grid-settings-history.json"
    history = read_json(path) if path.exists() else {"launches": []}
    rows = log_rows(directory)
    history["launches"].append({
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "resume": resume,
        "log_last_point_before_launch": (
            {"epoch": int(rows[-1]["epoch"]), "superbatch": int(rows[-1]["superbatch"])} if rows else None),
        "settings": copy.deepcopy(trial["settings"]),
        "effective_settings": resume_settings(trial["settings"]) if resume else copy.deepcopy(trial["settings"]),
    })
    atomic_json(path, history)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.output_folder.resolve()
    manifest_path = root / MANIFEST
    summary_path = (args.summary_csv or root / "grid_summary.csv").resolve()
    if summary_path.suffix.lower() != ".csv" or summary_path.is_relative_to(root / "trials"):
        raise ValueError("--summary-csv must be a .csv file outside trials/ (source logs are never overwritten)")
    if args.summary_only:
        plan = read_json(manifest_path)
        if plan.get("version") != 1:
            raise ValueError("unsupported grid manifest version")
        with grid_lock(root):
            rows = write_summary(root, plan, summary_path, sorted(set(args.epochs)) if args.epochs else None)
        print_leaders(rows)
        print(f"[SUMMARY] {summary_path}", flush=True)
        return 0

    plan = make_plan(args)
    stored = read_json(manifest_path) if manifest_path.is_file() else None
    selected = {t["id"] for t in plan["trials"]}
    if stored is not None and args.resume:
        plan, selected = plan_resume(root, stored, plan)
    elif stored is not None and stored != plan:
        raise ValueError("existing grid manifest differs from this plan; restore the original settings/grid or choose a different --output-folder (nothing was overwritten)")
    execution_plan = {**plan, "trials": [t for t in plan["trials"] if t["id"] in selected]}
    preflight_exe(execution_plan)
    print(f"[CONFIG] conditions={len(selected)} total_conditions={len(plan['trials'])} report_epochs={plan['report_epochs']} sequential=true", flush=True)
    if stored is not None and args.resume:
        for trial in execution_plan["trials"]:
            old = next((t for t in stored["trials"] if t["id"] == trial["id"]), None)
            if old is None:
                print(f"[ADD PLAN] trial={trial['id']} parameters={trial['parameters']} output={trial_dir(root, trial)}", flush=True)
                continue
            print(f"[RESUME PLAN] trial={trial['id']} parameters={trial['parameters']} max_epochs={old['settings']['max_epochs']}->{trial['settings']['max_epochs']} output={trial_dir(root, trial)}", flush=True)
            changes = settings_diff(old["settings"], trial["settings"])
            if changes:
                print(f"[SETTINGS CHANGED] trial={trial['id']} (applies on next launch)\n{changes}", flush=True)
    print("[CONFIG] output/output_folder/tag/resume are controlled per trial; all other common settings are preserved", flush=True)
    print(f"[CONFIG] relative teacher/input paths use cwd={plan['cwd']}", flush=True)
    objective_keys = {"wrm_target_scaling", "wrm_target_offset", "wrm_target_epsilon", "wrm_in_scaling", "wrm_in_offset",
                      "wrm_nnue2score", "scale", "fv_scale", "loss_pow_exp", "lambda",
                      "win_rate_model", "loss_sigmoid_mse", "loss_bce_with_logits"}
    if any(key in objective_keys and len(values) > 1 for key, values in plan["axes"].items()):
        print("[WARN] this grid varies loss/score conversion settings; raw loss/qloss rankings are not a common-objective comparison", flush=True)
    for trial in plan["trials"]:
        s = trial["settings"]
        if s.get("validation_rate") == -1 or s.get("quantized_validation_rate") == -1:
            print(f"[WARN] trial={trial['id']}: validation disabled; unmeasured CSV cells will be blank", flush=True)
    if args.dry_run:
        for trial in execution_plan["trials"]:
            print(f"[PLAN {trial['id']}] {json.dumps(trial['parameters'], ensure_ascii=False)}", flush=True)
            print(f"  output={trial_dir(root, trial)}", flush=True)
            print(f"  settings={json.dumps(trial['settings'], ensure_ascii=False)}", flush=True)
        print("[DRY RUN] no files written; no training started", flush=True)
        return 0

    failures = 0
    with grid_lock(root):
        if manifest_path.is_file():
            if read_json(manifest_path) != stored:
                raise ValueError("grid manifest changed before lock acquisition")
            if stored != plan:
                atomic_json(manifest_path, plan)
        else:
            if (root / "trials").exists() or summary_path.exists():
                raise ValueError("output contains trials/ or a summary but no manifest; use an empty grid root")
            atomic_json(manifest_path, plan)
        if args.resume:
            restart_unsaved_trials(root, plan, selected)
            atomic_json(manifest_path, plan)
        write_summary(root, plan, summary_path)
        print(f"[SUMMARY] initialized: {summary_path}", flush=True)
        for trial in plan["trials"]:
            if trial["id"] not in selected:
                print(f"[SKIP] trial={trial['id']} not selected; existing results retained", flush=True)
                continue
            directory = trial_dir(root, trial)
            state_path = directory / "grid-state.json"
            state = read_json(state_path) if state_path.is_file() else {}
            if is_complete(directory, trial, state):
                print(f"[SKIP] trial={trial['id']} completed: {directory}", flush=True)
                continue
            resume = has_resume_checkpoint(directory)
            if (resume or log_rows(directory)) and not args.resume:
                raise ValueError(f"trial {trial['id']} is incomplete; rerun the same command with --resume")
            directory.mkdir(parents=True, exist_ok=True)
            settings_path = directory / "bulletou-settings.json"
            check_trial_settings_file(directory, trial)
            if not settings_path.exists():
                atomic_json(settings_path, trial.get("initial_settings", trial["settings"]))
            if resume:
                atomic_json(directory / "bulletou-resume-settings.json", resume_settings(trial["settings"]))
            else:
                atomic_json(directory / "bulletou-run-settings.json", trial["settings"])
            command = command_for(plan, directory, resume, args.verbose)
            record_settings_launch(directory, trial, resume)
            old_elapsed = state.get("elapsed_seconds", 0)
            atomic_json(state_path, {"status": "running", "elapsed_seconds": old_elapsed})
            write_summary(root, plan, summary_path)
            print(f"[TRIAL {trial['id']} START] {trial['parameters']} resume={resume}", flush=True)
            print("[COMMAND] " + subprocess.list2cmdline(command), flush=True)
            started = time.monotonic()
            try:
                with live_summary_updates(root, plan, summary_path, trial):
                    code, elapsed = run_child(command, directory, plan["cwd"], trial["id"])
                rows = log_rows(directory)
                last = rows[-1] if rows else {}
                reached_end = (last.get("epoch") == str(trial["settings"]["max_epochs"])
                               and last.get("superbatch") == str(trial["settings"]["superbatches"]))
                status = "done" if code == 0 and reached_end else "failed"
                atomic_json(state_path, {"status": status, "exit_code": code,
                                       "elapsed_seconds": round(old_elapsed + elapsed, 3)})
            except BaseException:
                atomic_json(state_path, {"status": "interrupted", "elapsed_seconds": round(old_elapsed + time.monotonic() - started, 3)})
                write_summary(root, plan, summary_path)
                raise
            summaries = write_summary(root, plan, summary_path)
            print(f"[TRIAL {trial['id']} END] status={status} exit={code} elapsed={elapsed:.1f}s", flush=True)
            print(f"[SUMMARY] updated: {summary_path}", flush=True)
            if status != "done":
                failures += 1
                if not args.continue_on_error:
                    print(f"[ERROR] see {directory / 'stdout.log'}", flush=True)
                    return 1
        summaries = write_summary(root, plan, summary_path)
        print_leaders(summaries)
    print(f"[SUMMARY] {summary_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] stopped this grid's child; saved checkpoints remain. Use --resume.", flush=True)
        raise SystemExit(130)
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
