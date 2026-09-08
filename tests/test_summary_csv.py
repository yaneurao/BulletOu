import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import bulletou_csv
import bulletou_tuner as tuner
import summary_epoch_max_accuracy as aggregate
import tuning_parameters as tuning


class SummaryCsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def migrate(self):
        with redirect_stdout(io.StringIO()):
            bulletou_csv.migrate_summary_logs(self.root)

    def test_absent_directory_is_not_created(self):
        missing = self.root / "missing"
        bulletou_csv.migrate_summary_logs(missing)
        self.assertFalse(missing.exists())

    def test_rename_both_logs_preserves_bytes_and_is_idempotent(self):
        data = b'epoch,parameters\r\n5,"{""shared"":1}"\r\n'
        for name in (bulletou_csv.SUMMARY_CSV_NAME, bulletou_csv.ACCEPTED_SUMMARY_CSV_NAME):
            (self.root / name).with_suffix(".log").write_bytes(data)
        self.migrate()
        self.migrate()
        for name in (bulletou_csv.SUMMARY_CSV_NAME, bulletou_csv.ACCEPTED_SUMMARY_CSV_NAME):
            self.assertEqual((self.root / name).read_bytes(), data)
            self.assertFalse((self.root / name).with_suffix(".log").exists())

    def test_conflicting_accepted_file_prevents_all_moves(self):
        for name in (bulletou_csv.SUMMARY_CSV_NAME, bulletou_csv.ACCEPTED_SUMMARY_CSV_NAME):
            (self.root / name).with_suffix(".log").write_bytes(b"original")
        (self.root / bulletou_csv.ACCEPTED_SUMMARY_CSV_NAME).write_bytes(b"other")
        with self.assertRaises(FileExistsError):
            self.migrate()
        self.assertFalse((self.root / bulletou_csv.SUMMARY_CSV_NAME).exists())
        self.assertEqual((self.root / bulletou_csv.ACCEPTED_SUMMARY_CSV_NAME).read_bytes(), b"other")
        self.assertEqual((self.root / "summary-learn.log").read_bytes(), b"original")
        self.assertEqual((self.root / "accepted-summary-learn.log").read_bytes(), b"original")

    def test_tuning_resume_keeps_trial_and_commit_parameters(self):
        path = self.root / "summary-learn.log"
        params = {"shared": 1.0, "hand_axis": 0.75}
        row = dict(
            generation=3, generation_trial=4, trial_sbs=8, trial=104, status="finished",
            test_value_accuracy=0.65, test_value_loss=0.12,
            quantized_value_accuracy=0.64, quantized_value_loss=0.13,
            parameters=json.dumps(params), selection_metric=0.13, checkpoint="",
        )
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tuning.SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(row)
            writer.writerow({**row, "trial": "gen3-commit", "status": "commit_best"})
        self.migrate()
        path = self.root / bulletou_csv.SUMMARY_CSV_NAME
        tuning.upgrade_summary_csv(path)
        tuner.ensure_csv(path, tuning.SUMMARY_FIELDS)
        result, = tuning.load_completed(path)
        self.assertEqual((result.trial, result.generation, result.trial_sbs), (104, 3, 8))
        self.assertEqual(result.params, params)
        self.assertEqual(tuning.latest_commit_params(path), params)
        self.assertEqual(result.metric.qloss, 0.13)

    def test_normal_csv_consumers(self):
        path = self.root / bulletou_csv.SUMMARY_CSV_NAME
        path.write_text("epoch,superbatch,test_value_accuracy\n1,1,0.6\n1,2,0.65\n2,1,0.64\n", encoding="utf-8")
        self.assertEqual(tuner.latest_summary_row(self.root)["epoch"], "2")
        self.assertEqual(aggregate.load_epoch_max_accuracy(path, "test_value_accuracy"), {1: 0.65, 2: 0.64})
        self.assertEqual(aggregate.resolve_log_path(None).name, bulletou_csv.SUMMARY_CSV_NAME)


if __name__ == "__main__":
    unittest.main()
