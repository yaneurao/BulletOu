"""Summary CSV filenames and one-time migration for BulletOu runners."""

from pathlib import Path


SUMMARY_CSV_NAME = "summary-learn.csv"
ACCEPTED_SUMMARY_CSV_NAME = "accepted-summary-learn.csv"


def migrate_summary_logs(output_dir: Path) -> None:
    """Rename .log summaries at startup without changing their contents.

    Check all destinations before moving either file. If both extensions exist,
    the caller must resolve the ambiguity; neither history is silently replaced.
    """
    moves = []
    for name in (SUMMARY_CSV_NAME, ACCEPTED_SUMMARY_CSV_NAME):
        destination = output_dir / name
        source = destination.with_suffix(".log")
        if not source.exists():
            continue
        if destination.exists():
            raise FileExistsError(
                f"both {source} and {destination} exist; keep the intended summary "
                "and move the other file before restarting; neither file was changed"
            )
        moves.append((source, destination))
    for source, destination in moves:
        source.rename(destination)
        print(f"[SUMMARY] renamed {source} -> {destination}", flush=True)
