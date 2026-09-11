import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import grid_search as grid


class GridSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "grid"
        self.settings_path = self.root / "common.json"
        self.common = {
            "backend": "cuda-cpp", "arch": "SFNN_halfka2_1024_7_64_k3k3",
            "teacher": "D:/teachers", "test_teacher": "C:/test.hcpe",
            "max_epochs": 2, "superbatches": 4, "lr": 0.001, "lr_min": 0.00001,
            "wrm_in_offset": 0, "wrm_target_offset": 0,
            "resume": True, "output_folder": "D:/must-not-use", "tag": "original",
        }
        grid.atomic_json(self.settings_path, self.common)
        self.argv = ["--settings-file", str(self.settings_path), "--output-folder", str(self.output),
                     "--lrs", "0.0001", "0.0002"]

    def plan(self, extra=()):
        return grid.make_plan(grid.parse_args([*self.argv, *extra]))

    def summary(self, directory, rows):
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["epoch", "superbatch", *grid.METRICS, "positions", "lr_start", "lr_end", "checkpoint"]
        with (directory / grid.SUMMARY_CSV_NAME).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def checkpoint(self, directory, number="0001"):
        cp = directory / number
        cp.mkdir(parents=True)
        (cp / "state.bin").write_bytes(b"test-state")
        (cp / "dataloader_pos.txt").write_text("12345,0", encoding="utf-8")
        (cp / "nn.bin").write_bytes(b"test-nn")
        return cp

    def metrics(self, epoch=1, sb=4, checkpoint="-"):
        return dict(epoch=epoch, superbatch=sb, test_value_accuracy="0.63", test_value_loss="0.12",
                    quantized_value_accuracy="0.62", quantized_value_loss="0.13", checkpoint=checkpoint)

    def fake_run(self, command, directory, cwd, trial_id):
        settings = grid.read_json(Path(command[2]))
        self.assertEqual(settings["wrm_in_offset"], 0)
        self.assertEqual(settings["wrm_target_offset"], 0)
        self.assertEqual(settings["output"], str(directory))
        # The CSV exists before the first child starts, even with no data yet.
        self.assertTrue((self.output / "grid_summary.csv").is_file())
        self.summary(directory, [self.metrics(epoch=epoch) for epoch in range(1, settings["max_epochs"] + 1)])
        return 0, 1.25

    def run_grid(self, extra=(), child=None):
        with patch.object(grid, "preflight_exe"), patch.object(grid, "run_child", side_effect=child or self.fake_run) as run:
            with redirect_stdout(io.StringIO()):
                code = grid.main([*self.argv, *extra])
        return code, run

    def test_product_order_and_no_settings_writeback(self):
        before = self.settings_path.read_bytes()
        plan = self.plan(["--wrm-target-scalings", "600", "1200"])
        self.assertEqual([t["parameters"] for t in plan["trials"]], [
            {"lr": 0.0001, "wrm_target_scaling": 600}, {"lr": 0.0001, "wrm_target_scaling": 1200},
            {"lr": 0.0002, "wrm_target_scaling": 600}, {"lr": 0.0002, "wrm_target_scaling": 1200}])
        self.assertEqual(plan["report_epochs"], [1, 2])
        self.assertEqual(self.settings_path.read_bytes(), before)
        for trial in plan["trials"]:
            self.assertNotIn("resume", trial["settings"])
            self.assertNotIn("output_folder", trial["settings"])
            self.assertEqual(trial["settings"]["wrm_in_offset"], 0)
        self.assertFalse(self.output.exists())

    def test_epochs_train_once_to_max(self):
        plan = self.plan(["--epochs", "5", "1", "2", "1"])
        self.assertEqual(plan["report_epochs"], [1, 2, 5])
        self.assertEqual(plan["trials"][0]["settings"]["max_epochs"], 5)

    def test_generic_grid_string_boolean_and_zero(self):
        plan = self.plan(["--grid", "sfnn-factorizer-alpha", "shared=0.5", "shared=1.0",
                          "--grid", "no-ft-factorize", "false", "true",
                          "--grid", "validation_rate", "0"])
        self.assertEqual(len(plan["trials"]), 8)
        self.assertEqual(plan["trials"][0]["settings"]["sfnn_factorizer_alpha"], "shared=0.5")
        self.assertIs(plan["trials"][0]["settings"]["no_ft_factorize"], False)
        self.assertEqual(plan["trials"][0]["settings"]["validation_rate"], 0)

    def test_duplicate_axes_or_values_rejected(self):
        for extra in (["--grid", "lr", "0.001"], ["--grid", "wrm_target_scaling", "600", "600"],
                      ["--grid", "lr"], ["--grid", "initial-state", "abc"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.plan(extra)

    def test_zero_and_none_save_rates_pass_through_without_affecting_validation(self):
        for value in (0, "none"):
            with self.subTest(value=value):
                grid.atomic_json(self.settings_path, {**self.common, "save_rate": value,
                                 "validation_rate": 1, "quantized_validation_rate": 1})
                for trial in self.plan()["trials"]:
                    self.assertEqual(trial["settings"]["save_rate"], value)
                    self.assertEqual(trial["settings"]["validation_rate"], 1)
                    self.assertEqual(trial["settings"]["quantized_validation_rate"], 1)

    def test_invalid_values_rejected(self):
        for extra in (["--grid", "wrm_target_scaling", "NaN"],
                      ["--grid", "superbatches", "0"], ["--grid", "batch_size", "1.5"],
                      ["--grid", "wrm_target_scaling", "-1"], ["--grid", "lr_min", "0.5"],
                      ["--grid", "sfnn_factorizer_alpha", "{}"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.plan(extra)

    def test_normalized_duplicate_settings_rejected(self):
        grid.atomic_json(self.settings_path, {**self.common, "lr-min": 0.00001})
        with self.assertRaisesRegex(ValueError, "duplicate normalized"):
            self.plan()

    def test_validation_count_all_and_numeric_values(self):
        for value in ("invalid", "300000", 0, -1, 1.5, True):
            with self.subTest(value=value):
                grid.atomic_json(self.settings_path, {**self.common, "test_positions": value})
                with self.assertRaisesRegex(ValueError, "positive integer or 'all'"):
                    self.plan()
                self.assertFalse(self.output.exists())
        for value in (None, "all", 300000):
            grid.atomic_json(self.settings_path, {**self.common, "test_positions": value})
            self.assertEqual(self.plan()["trials"][0]["settings"]["test_positions"], value)
        grid.atomic_json(self.settings_path, self.common)
        self.assertNotIn("test_positions", self.plan()["trials"][0]["settings"])

    def test_bundled_settings_use_explicit_all_positions(self):
        example = Path(grid.__file__).parent / "docs/examples/grid-search-bulletou-settings.json"
        settings = grid.read_json(example)
        grid.check_settings(settings)
        self.assertEqual(settings["test_positions"], "all")

    def test_common_checkpoint_both_files_and_independent_starts(self):
        cp = self.checkpoint(self.root / "base")
        plan = self.plan(["--checkpoint", str(cp)])
        self.assertEqual({t["settings"]["initial_state"] for t in plan["trials"]}, {str(cp / "state.bin")})
        self.assertEqual({t["settings"]["initial_dataloader_pos"] for t in plan["trials"]}, {str(cp / "dataloader_pos.txt")})
        (cp / "dataloader_pos.txt").unlink()
        with self.assertRaisesRegex(ValueError, "dataloader_pos"):
            self.plan(["--checkpoint", str(cp)])

    def test_dry_run_no_writes_no_training(self):
        code, run = self.run_grid(["--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 0)
        self.assertFalse(self.output.exists())

    def test_preflight_rejects_unknown_flag_before_training(self):
        plan = self.plan()
        plan["exe"] = sys.executable
        help_text = "  --settings-file <FILE>\n  --resume\n description --lr is not an option definition\n"
        with patch.object(grid.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, help_text, "")):
            with self.assertRaisesRegex(ValueError, "--lr"):
                grid.preflight_exe(plan)

    def test_csv_metric_order_extrema_and_missing_final_not_carried(self):
        plan = self.plan()
        trial = plan["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        self.checkpoint(directory)
        rows = [self.metrics(sb=1), {**self.metrics(sb=2), "quantized_value_accuracy": "0.70"},
                {**self.metrics(sb=4, checkpoint="0001"), "quantized_value_accuracy": "-", "test_value_loss": "nan"}]
        self.summary(directory, rows)
        fields, result = grid.summarize(self.output, plan)
        row = result[0]
        self.assertEqual(fields[3:7], list(grid.METRICS))
        self.assertEqual(fields[-1], "checkpoint")
        self.assertEqual(row["quantized_value_accuracy"], "")
        self.assertEqual(row["test_value_loss"], "")
        self.assertEqual(row["max_qacc"], "0.70")
        self.assertEqual(row["max_qacc_sb"], "2")
        self.assertEqual(row["checkpoint"], str(directory / "0001"))
        self.assertEqual(result[1]["status"], "pending")
        self.assertEqual(rows[-1]["quantized_value_accuracy"], "-")

    def test_unsaved_peak_no_invented_checkpoint(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        self.summary(directory, [self.metrics()])
        row = grid.summarize(self.output, plan)[1][0]
        self.assertEqual(row["checkpoint"], "")
        self.assertEqual(row["max_qacc"], "0.62")

    def test_empty_and_zero_metrics(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        row = self.metrics()
        row.update({key: "0" for key in grid.METRICS})
        self.summary(directory, [row])
        result = grid.summarize(self.output, plan)[1]
        self.assertEqual(result[0]["min_qloss"], "0")
        self.assertEqual(result[0]["test_value_accuracy"], "0")
        self.assertEqual(result[2]["test_value_accuracy"], "")

    def test_duplicate_points_use_latest_row_not_file_time(self):
        directory = self.root / "trial"
        self.summary(directory, [{**self.metrics(), "test_value_accuracy": "0.9"}, self.metrics()])
        self.assertEqual(len(grid.log_rows(directory)), 1)
        self.assertEqual(grid.log_rows(directory)[0]["test_value_accuracy"], "0.63")

    def test_truncated_row_rejected_without_touching_source(self):
        directory = self.root / "trial"
        directory.mkdir()
        path = directory / grid.SUMMARY_CSV_NAME
        path.write_text("epoch,superbatch,test_value_accuracy\n1,", encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            grid.log_rows(directory)
        self.assertEqual(path.read_bytes(), before)

    def test_checkpoint_path_cannot_escape_trial(self):
        with self.assertRaises(ValueError):
            grid.checkpoint_path(self.root / "trial", {"checkpoint": "../other"})

    def test_complete_conditions_skip_and_summary_only_needs_no_exe(self):
        code, first = self.run_grid()
        self.assertEqual((code, first.call_count), (0, 2))
        code, second = self.run_grid()
        self.assertEqual((code, second.call_count), (0, 0))
        with patch.object(grid, "preflight_exe", side_effect=AssertionError("must not run exe")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(grid.main(["--output-folder", str(self.output), "--summary-only"]), 0)
        with (self.output / "grid_summary.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["status"] == "done" for row in rows))

    def test_changed_plan_refused_without_overwriting(self):
        self.run_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        grid.atomic_json(self.settings_path, {**self.common, "wrm_in_offset": 10})
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.run_grid(["--resume"])
        self.assertEqual((self.output / grid.MANIFEST).read_bytes(), before)

    def test_interrupt_resume_uses_own_state_not_common_initial(self):
        cp = self.checkpoint(self.root / "base")
        extra = ["--checkpoint", str(cp)]

        def interrupted(command, directory, cwd, trial_id):
            self.checkpoint(directory)
            self.summary(directory, [self.metrics(epoch=1, checkpoint="0001")])
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(extra, interrupted)
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.run_grid(extra)

        def resumed(command, directory, cwd, trial_id):
            settings = grid.read_json(Path(command[2]))
            if trial_id == 1:
                self.assertIn("--resume", command)
                self.assertNotIn("initial_state", settings)
                self.assertNotIn("initial_dataloader_pos", settings)
            else:
                self.assertNotIn("--resume", command)
                self.assertEqual(settings["initial_state"], str(cp / "state.bin"))
            return self.fake_run(command, directory, cwd, trial_id)

        code, calls = self.run_grid([*extra, "--resume"], resumed)
        self.assertEqual((code, calls.call_count), (0, 2))

    def test_resume_does_not_erase_progress_without_checkpoint(self):
        def interrupted(command, directory, cwd, trial_id):
            self.summary(directory, [self.metrics(sb=2)])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(child=interrupted)
        with self.assertRaisesRegex(ValueError, "no resumable checkpoint"):
            self.run_grid(["--resume"])

    def test_failed_run_even_with_final_save_not_skipped(self):
        plan = self.plan()
        trial = plan["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        self.checkpoint(directory)
        self.summary(directory, [self.metrics(epoch=2, checkpoint="0001")])
        self.assertTrue(grid.is_complete(directory, trial, {"status": "running"}))
        self.assertFalse(grid.is_complete(directory, trial, {"status": "failed"}))

    def test_failure_stops_or_continues_and_returns_nonzero(self):
        def fail(command, directory, cwd, trial_id):
            return 2, 0.1
        code, calls = self.run_grid(child=fail)
        self.assertEqual((code, calls.call_count), (1, 1))
        code, calls = self.run_grid(["--continue-on-error"], child=fail)
        self.assertEqual((code, calls.call_count), (1, 2))

    def test_exit_zero_without_final_row_is_failure(self):
        code, calls = self.run_grid(child=lambda *args: (0, 0.1))
        self.assertEqual((code, calls.call_count), (1, 1))

    def test_nested_lock_refused(self):
        with grid.grid_lock(self.output):
            with self.assertRaises(ValueError):
                with grid.grid_lock(self.output):
                    self.fail("nested lock succeeded")
        with grid.grid_lock(self.output):
            pass

    def test_real_child_streams_stdout_and_stderr_and_preserves_exit(self):
        with redirect_stdout(io.StringIO()) as output:
            code, elapsed = grid.run_child([sys.executable, "-u", "-c",
                "import sys; print('train'); print('qvalid',file=sys.stderr); sys.exit(3)"],
                self.root, str(self.root), 7)
        self.assertEqual(code, 3)
        self.assertGreater(elapsed, 0)
        self.assertIn("[TRIAL 7] train", output.getvalue())
        self.assertIn("qvalid", (self.root / "stdout.log").read_text(encoding="utf-8"))

    def test_summary_cannot_overwrite_trial_log(self):
        with self.assertRaisesRegex(ValueError, "outside trials"):
            self.run_grid(["--summary-csv", str(self.output / "trials" / "a" / "summary-learn.csv")])

    def saved_scale_grid(self):
        argv = [*self.argv[:-3], "--wrm-target-scalings", "600", "1200", "1800"]
        plan = grid.make_plan(grid.parse_args(argv))
        self.output.mkdir()
        grid.atomic_json(self.output / grid.MANIFEST, plan)
        for t in plan["trials"]:
            directory = grid.trial_dir(self.output, t)
            self.checkpoint(directory, "0002")
            self.summary(directory, [self.metrics(epoch=1), self.metrics(epoch=2, checkpoint="0002")])
            grid.atomic_json(directory / "bulletou-settings.json", t["settings"])
            grid.atomic_json(directory / "grid-state.json", {"status": "done", "elapsed_seconds": 3.0})
        return argv, plan

    def test_extend_subset_preserves_ids_folders_and_old_csv_results(self):
        argv, old = self.saved_scale_grid()
        untouched = grid.trial_dir(self.output, old["trials"][2])
        before = {p.name: p.read_bytes() for p in untouched.iterdir() if p.is_file()}
        original = {t["id"]: (grid.trial_dir(self.output, t) / "bulletou-settings.json").read_bytes()
                    for t in old["trials"]}
        requested = [*argv[:-1], "--resume", "--epochs", "7"]  # 600 / 1200 only
        seen = []

        def resumed(command, directory, cwd, trial_id):
            seen.append(trial_id)
            self.assertIn("--resume", command)
            s = grid.read_json(Path(command[2]))
            self.assertEqual(s["max_epochs"], 7)
            self.assertEqual(directory, grid.trial_dir(self.output, old["trials"][trial_id-1]))
            self.checkpoint(directory, "0007")
            self.summary(directory, [*grid.log_rows(directory), *[
                self.metrics(epoch=e, checkpoint="0007" if e == 7 else "-") for e in range(3,8)]])
            return 0, 1.0

        with patch.object(grid, "preflight_exe"), patch.object(grid, "run_child", side_effect=resumed), redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main(requested), 0)
            self.assertEqual(grid.main(requested), 0)  # Idempotent; no second extension.
        self.assertEqual(seen, [1,2])
        merged = grid.read_json(self.output / grid.MANIFEST)
        self.assertEqual([t["settings"]["max_epochs"] for t in merged["trials"]], [7,7,2])
        self.assertEqual(merged["axes"], old["axes"])
        self.assertEqual(merged["report_epochs"], list(range(1,8)))
        self.assertEqual(before, {p.name: p.read_bytes() for p in untouched.iterdir() if p.is_file()})
        for t in old["trials"]:
            self.assertEqual(original[t["id"]], (grid.trial_dir(self.output,t) / "bulletou-settings.json").read_bytes())
        rows = grid.summarize(self.output, merged)[1]
        self.assertEqual(len(rows), 16)  # 7 + 7 + 2; no fictitious extra 1800 epochs.
        self.assertTrue(all(r["status"] == "done" for r in rows))

    def test_resume_reordered_subset_keeps_original_trial_id(self):
        argv, old = self.saved_scale_grid()
        req = grid.make_plan(grid.parse_args([*argv[:-3], "1200", "--epochs", "7", "--resume"]))
        merged, selected = grid.plan_resume(self.output, old, req)
        self.assertEqual(selected, {2})
        self.assertEqual(merged["trials"][1]["name"], old["trials"][1]["name"])
        rows = grid.summarize(self.output, merged)[1]
        self.assertEqual({r["trial_status"] for r in rows if r["trial"] == 2}, {"incomplete"})
        self.assertEqual({r["trial_status"] for r in rows if r["trial"] == 1}, {"done"})

    def test_extension_rejects_missing_checkpoint_without_writes(self):
        argv, old = self.saved_scale_grid()
        (grid.trial_dir(self.output, old["trials"][1]) / "0002" / "state.bin").unlink()
        before = (self.output / grid.MANIFEST).read_bytes()
        with self.assertRaisesRegex(ValueError, "no resumable checkpoint"):
            grid.main([*argv, "--resume", "--epochs", "7"])
        self.assertEqual(before, (self.output / grid.MANIFEST).read_bytes())

    def test_extension_rejects_changed_condition_shrink_and_new_values(self):
        argv, old = self.saved_scale_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        for arguments in ([*argv, "--resume", "--epochs", "1"],
                          [*argv[:-3], "2400", "--resume", "--epochs", "7"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                grid.main(arguments)
        grid.atomic_json(self.settings_path, {**self.common, "lr": 0.0003})
        with self.assertRaisesRegex(ValueError, "training settings changed"):
            grid.main([*argv, "--resume", "--epochs", "7"])
        self.assertEqual(before, (self.output / grid.MANIFEST).read_bytes())

    def test_extension_dry_run_and_lock_failure_do_not_write(self):
        argv, old = self.saved_scale_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        with patch.object(grid,"preflight_exe"), patch.object(grid,"run_child") as child, redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main([*argv,"--resume","--epochs","7","--dry-run"]),0)
            with grid.grid_lock(self.output):
                with self.assertRaisesRegex(ValueError, "another grid"):
                    grid.main([*argv,"--resume","--epochs","7"])
            child.assert_not_called()
        self.assertEqual(before,(self.output / grid.MANIFEST).read_bytes())

    def test_extension_resume_after_interrupt_keeps_new_target(self):
        argv, old = self.saved_scale_grid()
        requested=[*argv[:-3],"600","--resume","--epochs","7"]
        def interrupted(command,directory,cwd,trial_id):
            self.checkpoint(directory,"0003")
            self.summary(directory,[*grid.log_rows(directory),self.metrics(epoch=3,checkpoint="0003")])
            raise KeyboardInterrupt
        with patch.object(grid,"preflight_exe"), patch.object(grid,"run_child",side_effect=interrupted), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                grid.main(requested)
        def resumed(command,directory,cwd,trial_id):
            self.assertIn("--resume",command)
            self.assertEqual(grid.log_rows(directory)[-1]["epoch"],"3")
            self.assertEqual(grid.read_json(Path(command[2]))["max_epochs"],7)
            self.summary(directory,[*grid.log_rows(directory),*[self.metrics(epoch=e) for e in range(4,8)]])
            return 0,1.0
        with patch.object(grid,"preflight_exe"), patch.object(grid,"run_child",side_effect=resumed) as child, redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main(requested),0)
            self.assertEqual(child.call_count,1)


if __name__ == "__main__":
    unittest.main()
