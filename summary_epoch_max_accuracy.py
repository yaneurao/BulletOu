#!/usr/bin/env python3
"""Extract per-epoch maximum accuracy from summary-learn.log.

Default behavior:
    python summary_epoch_max_accuracy.py

reads ./summary-learn.log and prints one tab-separated row containing the
maximum test_value_accuracy for each epoch, sorted by epoch.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-epoch maximum accuracy from BulletOu summary-learn.log as TSV."
    )
    parser.add_argument(
        "log",
        nargs="?",
        default=None,
        help="summary-learn.log path. Defaults to summary-learn.log in the current working directory",
    )
    parser.add_argument(
        "--column",
        default="test_value_accuracy",
        help="accuracy column name. Defaults to test_value_accuracy",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="print accuracy as percent values, e.g. 62.0168 instead of 0.620168",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="digits after the decimal point. Defaults to 6",
    )
    parser.add_argument(
        "--with-epoch",
        action="store_true",
        help="print two TSV rows: epoch numbers, then max accuracies",
    )
    parser.add_argument(
        "--per-line",
        action="store_true",
        help="print one epoch per line as: epoch<TAB>max_accuracy",
    )
    return parser.parse_args()


def resolve_log_path(log: str | None) -> Path:
    if log is None:
        return Path.cwd() / "summary-learn.log"

    path = Path(log)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def format_accuracy(value: float, *, percent: bool, digits: int) -> str:
    if percent:
        value *= 100.0
    return f"{value:.{digits}f}"


def load_epoch_max_accuracy(path: Path, column: str) -> dict[int, float]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    best_by_epoch: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or has no CSV header")
        if "epoch" not in reader.fieldnames:
            raise ValueError(f"{path} has no 'epoch' column")
        if column not in reader.fieldnames:
            raise ValueError(f"{path} has no '{column}' column")

        for line_no, row in enumerate(reader, start=2):
            epoch_text = (row.get("epoch") or "").strip()
            value_text = (row.get(column) or "").strip()
            if not epoch_text or not value_text:
                continue

            try:
                epoch = int(epoch_text)
                value = float(value_text)
            except ValueError as e:
                raise ValueError(
                    f"{path}:{line_no}: invalid epoch or {column}: "
                    f"epoch={epoch_text!r}, {column}={value_text!r}"
                ) from e

            if math.isnan(value):
                continue
            if epoch not in best_by_epoch or value > best_by_epoch[epoch]:
                best_by_epoch[epoch] = value

    if not best_by_epoch:
        raise ValueError(f"{path} has no usable '{column}' values")
    return best_by_epoch


def main() -> int:
    args = parse_args()
    path = resolve_log_path(args.log)

    try:
        best_by_epoch = load_epoch_max_accuracy(path, args.column)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    epochs = sorted(best_by_epoch)
    accuracies = [
        format_accuracy(best_by_epoch[epoch], percent=args.percent, digits=args.digits)
        for epoch in epochs
    ]

    if args.per_line:
        for epoch, accuracy in zip(epochs, accuracies):
            print(f"{epoch}\t{accuracy}")
    elif args.with_epoch:
        print("\t".join(str(epoch) for epoch in epochs))
        print("\t".join(accuracies))
    else:
        print("\t".join(accuracies))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
