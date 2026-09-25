import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import grid_search as grid


class GridSearchTests(unittest.TestCase):
    def test_l2_revive_grid(self):
        plan=self.plan(["--grid","sfnn-l2-revive","false","true"])
        self.assertEqual({t["settings"]["sfnn_l2_revive"] for t in plan["trials"]},{False,True})
        self.assertIn("sfnn_l2_revive",grid.COMMON_COLUMNS)
    def test_l2_revive_zero_grid(self):
        plan=self.plan(["--grid","sfnn-l2-revive-zero","false","true"])
        self.assertEqual({t["settings"]["sfnn_l2_revive_zero"] for t in plan["trials"]},{False,True})
        self.assertIn("sfnn_l2_revive_zero",grid.COMMON_COLUMNS)
    def test_bn_l2_effective_clip_grid(self):
        plan=self.plan(["--grid","sfnn-bn-l2-effective-weight-clip","false","true"])
        self.assertEqual({t["settings"]["sfnn_bn_l2_effective_weight_clip"] for t in plan["trials"]},{False,True})
        self.assertIn("sfnn_bn_l2_effective_weight_clip",grid.COMMON_COLUMNS)

    def test_bn_momentum_grid(self):
        plan=self.plan(["--grid","sfnn-bn-momentum","0.1","0.01","0.001"])
        self.assertEqual({t["settings"]["sfnn_bn_momentum"] for t in plan["trials"]},{0.1,0.01,0.001})
        self.assertIn("sfnn_bn_momentum",grid.COMMON_COLUMNS)

    def test_bn_qat_epoch_schedule(self):
        settings = {"sfnn_bn_qat": {"epoch1": False, "epoch6": True, "epoch8": False}}
        for epoch in (0, 1, 5, 6, 7, 8, 10):
            self.assertEqual(grid.resolve_epoch_settings(settings, epoch)["sfnn_bn_qat"], 6 <= epoch < 8)
        with self.assertRaisesRegex(ValueError, "true/false"):
            grid.resolve_epoch_settings({"sfnn_bn_qat": {"epoch1": False, "epoch6": 1}}, 1)
        self.common["sfnn_bn_qat"] = settings["sfnn_bn_qat"]
        grid.atomic_json(self.settings_path, self.common)
        plan = self.plan(["--lrs", "0.0001"])
        self.assertEqual(plan["trials"][0]["settings"]["sfnn_bn_qat"], settings["sfnn_bn_qat"])

    def test_nnue_bn_grid_conditions(self):
        self.common["arch"] = "NNUE_ka2_256x2_32_32"
        grid.atomic_json(self.settings_path, self.common)
        plan = self.plan(["--lrs", "0.0001", "--grid", "nnue-bn-ft", "false", "true",
                          "--grid", "nnue-bn-l1", "true", "--grid", "nnue-bn-l2", "true"])
        self.assertEqual(len(plan["trials"]), 2)
        self.assertEqual({t["settings"]["nnue_bn_ft"] for t in plan["trials"]}, {False, True})
        for trial in plan["trials"]:
            self.assertTrue(trial["settings"]["nnue_bn_l1"])
            self.assertTrue(trial["settings"]["nnue_bn_l2"])
        for name in ("ft", "l1", "l2", "gamma", "beta", "momentum", "epsilon"):
            self.assertIn("nnue_bn_" + name, grid.COMMON_COLUMNS)

    def test_ft_saturation_guard_grid_and_epoch(self):
        plan = self.plan(["--grid", "sfnn-ft-saturation-penalty", "0", "0.0001", "0.001"])
        self.assertEqual({t["settings"]["sfnn_ft_saturation_penalty"] for t in plan["trials"]}, {0, 0.0001, 0.001})
        for key in ("sfnn_ft_saturation_penalty", "sfnn_ft_saturation_rate", "sfnn_ft_saturation_patience"):
            self.assertIn(key, grid.EPOCH_SETTING_KEYS)
            self.assertEqual(grid.resolve_epoch_settings({key: {"epoch1": 1, "epoch2": 2}}, 2)[key], 2)

    def test_missing_grid_value_names_the_offending_axis(self):
        args = grid.parse_args(["--settings-file", "unused.json", "--output-folder", "unused",
                               "--grid", "sfnn-l1-effective-weight-clip", "--grid", "warmup_sb", "0"])
        with self.assertRaises(ValueError) as raised:
            grid.collect_axes(args)
        message = str(raised.exception)
        self.assertIn("--grid sfnn-l1-effective-weight-clip: missing value(s)", message)
        self.assertIn("--grid sfnn-l1-effective-weight-clip true", message)
        self.assertIn("--grid sfnn-l1-effective-weight-clip false true", message)
        self.assertNotIn("warmup_sb", message)

    def test_effective_weight_clip_grid(self):
        plan = self.plan(["--grid", "sfnn-l1-effective-weight-clip", "false", "true"])
        self.assertEqual({t["settings"]["sfnn_l1_effective_weight_clip"] for t in plan["trials"]}, {False, True})

    def test_verbose_is_forwarded_without_changing_trial_identity(self):
        normal = self.plan([])
        verbose = self.plan(["--verbose"])
        self.assertEqual(normal, verbose)
        for resume in (False, True):
            command = grid.command_for(normal, self.output, resume, True)
            self.assertEqual(command.count("--verbose"), 1)
            self.assertEqual("--resume" in command, resume)
            self.assertNotIn("--verbose", grid.command_for(normal, self.output, resume))

    def test_l1_center_grid_and_epoch_booleans(self):
        plan = self.plan(["--grid", "sfnn-l1-center", "false", "true"])
        self.assertEqual({t["settings"]["sfnn_l1_center"] for t in plan["trials"]}, {False, True})
        for key in ("sfnn_l1_center", "sfnn_l2_l3_center", "sfnn_l1_effective_weight_clip"):
            values = {key: {"epoch1": False, "epoch2": True}}
            self.assertFalse(grid.resolve_epoch_settings(values, 1)[key])
            self.assertTrue(grid.resolve_epoch_settings(values, 2)[key])
            with self.assertRaises(ValueError):
                grid.resolve_epoch_settings({key: {"epoch1": 1}}, 1)

    def test_trial_warning_color_is_console_only(self):
        line = "  WARN: clipping disabled\n"
        with patch.dict(os.environ, {"BULLETOU_COLOR": "always"}, clear=True):
            self.assertEqual(grid.trial_console_line(1, line),
                             "\x1b[1;33m[TRIAL 1]   WARN: clipping disabled\x1b[0m\n")
            self.assertEqual(grid.trial_console_line(1, "train\n"), "[TRIAL 1] train\n")
            with patch.dict(os.environ, {"NO_COLOR": "1"}):
                self.assertEqual(grid.trial_console_line(1, line), "[TRIAL 1] " + line)
        with patch.dict(os.environ, {}, clear=True), patch.object(sys.stdout, "isatty", return_value=False):
            self.assertEqual(grid.trial_console_line(1, line), "[TRIAL 1] " + line)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"BULLETOU_COLOR": "always"}, clear=True):
            output = io.StringIO()
            with redirect_stdout(output):
                code, _ = grid.run_child([sys.executable, "-c", "print('  WARN: clipping disabled')"], Path(tmp), tmp, 1)
            self.assertEqual(code, 0)
            self.assertIn("\x1b[1;33m", output.getvalue())
            self.assertNotIn("\x1b", (Path(tmp) / "stdout.log").read_text(encoding="utf-8"))

    def test_warmup_sb_grid(self):
        grid.check_settings({**self.common, "superbatches": 1, "max_epochs": 1, "warmup_sb": 1024})
        grid.check_settings({**self.common, "superbatches": 1, "max_epochs": 2, "warmup_sb": 1024})
        plan = self.plan(["--grid", "warmup_sb", "0", "1"])
        self.assertEqual({t["settings"]["warmup_sb"] for t in plan["trials"]}, {0, 1})
        self.complete_plan(plan)
        fields, rows = grid.summarize(self.output, plan)
        self.assertIn("warmup_sb", fields)
        self.assertEqual({r["warmup_sb"] for r in rows}, {0, 1})
        for invalid in [-1, 0.5, True]:
            with self.assertRaisesRegex(ValueError, "warmup_sb"):
                grid.check_settings({**self.common, "warmup_sb": invalid})

    def test_epoch_settings_numeric_boolean_inheritance_and_validation(self):
        settings = {**self.common,
                    "lr": {"epoch1": 0.0004, "epoch11": 0.0002},
                    "sfnn_qat_l1": {"epoch1": True, "epoch11": False},
                    "batches_per_update": {"epoch1": 1, "epoch11": 4}}
        grid.check_settings(settings)
        for epoch in (1, 2, 10, 11, 15):
            resolved = grid.resolve_epoch_settings(settings, epoch)
            self.assertEqual(resolved["lr"], 0.0004 if epoch < 11 else 0.0002)
            self.assertEqual(resolved["sfnn_qat_l1"], epoch < 11)
        for key, value in [("arch", {"epoch1": "foo"}), ("lr", {"epoch2": .001}),
                           ("lr", {"epoch1": .001, "epoch02": .001}),
                           ("sfnn_qat_l1", {"epoch1": 1}),
                           ("lr", {"epoch1": .001, "epoch2": -1})]:
            with self.assertRaises(ValueError):
                grid.check_settings({**self.common, key: value})

    def test_warmup_epoch_zero_summary(self):
        plan = self.plan(["--grid", "warmup_sb", "64"])
        for trial in plan["trials"]:
            directory = grid.trial_dir(self.output, trial)
            self.summary(directory, [self.metrics(epoch=0, sb=63)])
        _, rows = grid.summarize(self.output, plan)
        self.assertTrue(all(not r.get("test_value_accuracy") for r in rows if r["epoch"] == 0))
        for trial in plan["trials"]:
            self.summary(grid.trial_dir(self.output, trial), [self.metrics(epoch=0, sb=64), self.metrics(epoch=1)])
        _, rows = grid.summarize(self.output, plan)
        warmup = [r for r in rows if r["epoch"] == 0]
        self.assertEqual(len(warmup), len(plan["trials"]))
        self.assertTrue(all(r["status"] == "done" for r in warmup))
        self.assertEqual(grid.resolve_epoch_settings({"lr": {"epoch1": .001, "epoch2": .0001}}, 0)["lr"], .001)

    def test_epoch_settings_grid_override_and_summary(self):
        settings = {**self.common, "lr": {"epoch1": .0004, "epoch2": .0002},
                    "sfnn_qat_l1": {"epoch1": True, "epoch2": False}}
        grid.atomic_json(self.settings_path, settings)
        plan = self.plan()
        self.assertEqual(plan["trials"][0]["settings"]["lr"], .0001)  # explicit axis wins
        args = grid.parse_args(["--settings-file", str(self.settings_path), "--output-folder", str(self.output),
                               "--grid", "wrm_target_offset", "0"])
        plan = grid.make_plan(args)
        self.complete_plan(plan)
        _, rows = grid.summarize(self.output, plan)
        self.assertEqual([r["lr"] for r in rows], [.0004, .0002])
        self.assertEqual([r["sfnn_qat_l1"] for r in rows], [True, False])

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

    def test_summary_keeps_target_scaling_without_elapsed_seconds(self):
        plan = self.plan(["--wrm-target-scalings", "600", "1200", "1800"])
        self.complete_plan(plan)
        fields, rows = grid.summarize(self.output, plan)
        self.assertEqual(fields.count("wrm_target_scaling"), 1)
        self.assertEqual({row["wrm_target_scaling"] for row in rows}, {600, 1200, 1800})
        self.assertNotIn("elapsed_seconds", fields)
        self.assertTrue(all("elapsed_seconds" not in row for row in rows))
        self.assertEqual(fields[-1], "checkpoint")

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if predicate():
                    return
            except PermissionError:
                pass  # Windows may briefly deny reads during atomic replacement.
            time.sleep(0.01)
        self.fail("timed out waiting for live epoch summary")

    def csv_rows(self, path):
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def test_grid_updates_finished_epoch_before_child_exits(self):
        def child(command, directory, cwd, trial_id):
            self.summary(directory, [self.metrics(epoch=1), self.metrics(epoch=2, sb=1)])
            path = self.output / "grid_summary.csv"
            def published():
                rows = [r for r in self.csv_rows(path) if r["trial"] == str(trial_id)]
                return rows[0]["test_value_accuracy"] == "0.63"
            self.wait_until(published)
            rows = [r for r in self.csv_rows(path) if r["trial"] == str(trial_id)]
            self.assertEqual(rows[0]["status"], "done")
            self.assertEqual(rows[0]["trial_status"], "running")
            for row in rows[1:]:
                for metric in grid.METRICS:
                    self.assertEqual(row[metric], "")
            self.summary(directory, [self.metrics(epoch=e) for e in range(1, 11)])
            return 0, 1.0
        code, _ = self.run_grid(["--epochs", *map(str, range(1, 11))], child)
        self.assertEqual(code, 0)

    def test_live_summary_retries_partial_rows_and_locked_output(self):
        plan = self.plan()
        trial = plan["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        self.summary(directory, [self.metrics(epoch=1)])
        log = directory / grid.SUMMARY_CSV_NAME
        complete = log.read_bytes()
        self.summary(directory, [])
        path = self.output / "grid_summary.csv"
        grid.write_summary(self.output, plan, path)
        original = path.read_bytes()
        with redirect_stdout(io.StringIO()):
            with grid.live_summary_updates(self.output, plan, path, trial, interval=0.01):
                # Valid CSV prefix, but not yet a completely appended record.
                log.write_bytes(complete.rstrip(b"\r\n"))
                with self.assertRaises(ValueError):
                    grid.log_rows(directory, live=True)
                time.sleep(0.05)
                self.assertEqual(path.read_bytes(), original)
                with patch.object(grid, "write_summary", side_effect=PermissionError("locked")) as blocked:
                    log.write_bytes(complete)
                    self.wait_until(lambda: blocked.call_count > 0)
                    self.assertEqual(path.read_bytes(), original)
                self.wait_until(lambda: self.csv_rows(path)[0]["status"] == "done")
                self.summary(directory, [self.metrics(epoch=1), self.metrics(epoch=2)])
                self.wait_until(lambda: self.csv_rows(path)[1]["status"] == "done")
        # No monitor remains to race the final writer after leaving the context.
        self.assertEqual(self.csv_rows(path)[0]["test_value_accuracy"], "0.63")

    def test_bce_grid_settings_and_columns(self):
        plan = self.plan(["--grid", "loss_bce_with_logits", "false", "true"])
        self.assertEqual(len(plan["trials"]), 4)
        fields, rows = grid.summarize(self.output, plan)
        self.assertIn("loss_bce_with_logits", fields)
        self.assertEqual({r["loss_bce_with_logits"] for r in rows}, {False, True})
        for bad in ({"wrm_in_offset": 270}, {"win_rate_model": True},
                    {"loss_sigmoid_mse": True}, {"loss_pow_exp": 3}):
            with self.assertRaises(ValueError):
                grid.check_settings({**self.common, "loss_bce_with_logits": True, **bad})

    def test_bce_error_weight_grid(self):
        grid.atomic_json(self.settings_path, {**self.common, "loss_bce_with_logits": True})
        plan = self.plan(["--grid", "bce_error_weight_k", "0", "1", "2"])
        self.assertEqual(len(plan["trials"]), 6)
        fields, rows = grid.summarize(self.output, plan)
        self.assertEqual(fields.count("bce_error_weight_k"), 1)
        self.assertEqual({r["bce_error_weight_k"] for r in rows}, {0, 1, 2})
        for k in [-1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                grid.check_settings({**self.common, "loss_bce_with_logits": True, "bce_error_weight_k": k})
        with self.assertRaisesRegex(ValueError, "requires loss_bce"):
            grid.check_settings({**self.common, "bce_error_weight_k": 1})

    def test_target_epsilon_grid_and_summary(self):
        plan = self.plan(["--wrm-target-epsilons", "0", "0.005", "0.01"])
        self.complete_plan(plan)
        fields, rows = grid.summarize(self.output, plan)
        self.assertEqual(fields.count("wrm_target_epsilon"), 1)
        self.assertEqual({row["wrm_target_epsilon"] for row in rows}, {0, 0.005, 0.01})
        for value in [-0.01, 0.5, float("nan"), float("inf"), True, "0.01"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                grid.check_settings({**self.common, "wrm_target_epsilon": value})
        with self.assertRaises(ValueError):
            grid.check_settings({**self.common, "wrm_target_epsilon": 0.01, "loss_sigmoid_mse": True})

    def test_norm_loss_grid_keeps_common_settings_and_reports_strength(self):
        plan = self.plan(["--grid", "sfnn_norm_loss_strength", "0", "0.00001", "0.0001"])
        self.assertEqual(len(plan["trials"]), 6)
        fields, rows = grid.summarize(self.output, plan)
        self.assertEqual(fields.count("sfnn_norm_loss_strength"), 1)
        self.assertEqual({r["sfnn_norm_loss_strength"] for r in rows}, {0, 1e-5, 1e-4})
        for trial in plan["trials"]:
            self.assertEqual(trial["settings"]["wrm_target_offset"], self.common["wrm_target_offset"])

    def test_l2_l3_center_grid_boolean_and_summary(self):
        plan = self.plan(["--grid", "sfnn-l2-l3-center", "false", "true"])
        fields, rows = grid.summarize(self.output, plan)
        self.assertEqual(fields.count("sfnn_l2_l3_center"), 1)
        self.assertEqual({row["sfnn_l2_l3_center"] for row in rows}, {False, True})
        single = self.plan(["--grid", "sfnn-l2-l3-center", "true"])
        self.assertTrue(all(t["settings"]["sfnn_l2_l3_center"] is True for t in single["trials"]))

    def test_glorot_and_centering_cartesian_grid(self):
        plan=self.plan(["--grid","sfnn-init-l2-l3-glorot","false","true",
                        "--grid","sfnn-l2-l3-center","false","true"])
        fields,rows=grid.summarize(self.output,plan)
        self.assertEqual(fields.count("sfnn_init_l2_l3_glorot"),1)
        self.assertEqual({(r["sfnn_init_l2_l3_glorot"],r["sfnn_l2_l3_center"]) for r in rows},
                         {(False,False),(False,True),(True,False),(True,True)})

    def test_batch_norm_axes_and_summary(self):
        plan=self.plan(["--grid","sfnn-bn-ft","false","true",
                        "--grid","sfnn-bn-l1","false","true",
                        "--grid","sfnn-bn-l2","false","true",
                        "--grid","sfnn-bn-gamma","0.25"])
        self.assertEqual(len(plan["trials"]),16)  # fixture also varies lr over two values
        fields,rows=grid.summarize(self.output,plan)
        for key in ("sfnn_bn_ft","sfnn_bn_l1","sfnn_bn_l2","sfnn_bn_gamma"):
            self.assertEqual(fields.count(key),1)
        self.assertEqual(len({(r["sfnn_bn_ft"],r["sfnn_bn_l1"],r["sfnn_bn_l2"]) for r in rows}),8)

    def complete_plan(self, plan):
        for trial in plan["trials"]:
            directory = grid.trial_dir(self.output, trial)
            self.summary(directory, [self.metrics(epoch=e) for e in range(1, trial["settings"]["max_epochs"] + 1)])
            grid.atomic_json(directory / "grid-state.json", {"status": "done"})

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
        self.summary(directory, [*rows, self.metrics(epoch=2)])
        grid.atomic_json(directory / "grid-state.json", {"status": "done"})
        fields, result = grid.summarize(self.output, plan)
        row = result[0]
        self.assertEqual(fields[3:7], list(grid.METRICS))
        self.assertEqual(fields[-1], "checkpoint")
        self.assertEqual(row["quantized_value_accuracy"], "")
        self.assertEqual(row["test_value_loss"], "")
        self.assertEqual(row["max_qacc"], "0.70")
        self.assertEqual(row["max_qacc_sb"], "2")
        self.assertEqual(row["checkpoint"], str(directory / "0001"))
        self.assertEqual(result[1]["status"], "done")
        self.assertEqual(len(result), 4)
        self.assertEqual(rows[-1]["quantized_value_accuracy"], "-")

    def test_summary_omits_final_sb_lr_but_keeps_configured_lr(self):
        plan = self.plan()
        trial = plan["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        self.summary(directory, [{**self.metrics(), "lr_start": "0.000015", "lr_end": "0.000014"}])
        source = (directory / grid.SUMMARY_CSV_NAME).read_bytes()
        path = self.output / "grid_summary.csv"
        grid.write_summary(self.output, plan, path)
        rows = self.csv_rows(path)
        for row in rows:
            self.assertNotIn("lr_start", row)
            self.assertNotIn("lr_end", row)
        self.assertEqual(float(rows[0]["lr"]), trial["settings"]["lr"])
        self.assertEqual(float(rows[0]["lr_min"]), trial["settings"]["lr_min"])
        self.assertEqual((directory / grid.SUMMARY_CSV_NAME).read_bytes(), source)

    def test_unsaved_peak_no_invented_checkpoint(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        self.summary(directory, [self.metrics(), self.metrics(epoch=2)])
        grid.atomic_json(directory / "grid-state.json", {"status": "done"})
        row = grid.summarize(self.output, plan)[1][0]
        self.assertEqual(row["checkpoint"], "")
        self.assertEqual(row["max_qacc"], "0.62")

    def test_empty_and_zero_metrics(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        row = self.metrics()
        row.update({key: "0" for key in grid.METRICS})
        self.summary(directory, [row, self.metrics(epoch=2)])
        grid.atomic_json(directory / "grid-state.json", {"status": "done"})
        result = grid.summarize(self.output, plan)[1]
        self.assertEqual(result[0]["min_qloss"], "0")
        self.assertEqual(result[0]["test_value_accuracy"], "0")
        self.assertEqual(len(result), 4)

    def test_duplicate_points_use_latest_row_not_file_time(self):
        directory = self.root / "trial"
        self.summary(directory, [{**self.metrics(), "test_value_accuracy": "0.9"}, self.metrics()])
        self.assertEqual(len(grid.log_rows(directory)), 1)
        self.assertEqual(grid.log_rows(directory)[0]["test_value_accuracy"], "0.63")

    def test_summary_keeps_completed_epochs_and_blanks_partial_epochs(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        self.summary(directory, [self.metrics(epoch=1), self.metrics(epoch=2, sb=2)])
        source = (directory / grid.SUMMARY_CSV_NAME).read_bytes()
        for status in ("pending", "running", "interrupted", "failed", "done"):
            with self.subTest(status=status):
                grid.atomic_json(directory / "grid-state.json", {"status": status})
                path = self.output / "grid_summary.csv"
                result = grid.write_summary(self.output, plan, path)
                self.assertEqual(len(result), 4)
                self.assertEqual(result[0]["lr"], 0.0001)
                self.assertEqual(result[0]["status"], "done")
                self.assertEqual(result[0]["test_value_accuracy"], "0.63")
                self.assertEqual(result[0]["trial_status"], "incomplete" if status == "done" else status)
                with path.open(encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    self.assertIn("test_value_accuracy", reader.fieldnames)
                    for row in reader:
                        if row["trial"] == "1" and row["epoch"] == "1":
                            continue
                        for key in (*grid.METRICS, *grid.EXTREMA, "checkpoint", "superbatch", "positions"):
                            self.assertEqual(row[key], "")
        self.assertEqual((directory / grid.SUMMARY_CSV_NAME).read_bytes(), source)

    def test_summary_recovers_saved_final_but_excludes_failed_trial(self):
        plan = self.plan()
        directory = grid.trial_dir(self.output, plan["trials"][0])
        self.checkpoint(directory)
        self.summary(directory, [self.metrics(epoch=1, sb=2), self.metrics(epoch=2, checkpoint="0001")])
        grid.atomic_json(directory / "grid-state.json", {"status": "interrupted"})
        rows = grid.summarize(self.output, plan)[1]
        self.assertEqual([r["epoch"] for r in rows if r["status"] == "done"], [2])
        self.assertEqual(rows[0]["trial_status"], "done")
        grid.atomic_json(directory / "grid-state.json", {"status": "failed"})
        rows = grid.summarize(self.output, plan)[1]
        self.assertEqual(rows[1]["status"], "done")
        self.assertEqual(rows[1]["trial_status"], "failed")
        self.assertEqual(rows[1]["test_value_accuracy"], "0.63")

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
            self.run_grid()
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

    def test_resume_archives_unsaved_progress_and_restarts_initial_state(self):
        def interrupted(command, directory, cwd, trial_id):
            self.summary(directory, [self.metrics(sb=2)])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(child=interrupted)
        old = grid.trial_dir(self.output, self.plan()["trials"][0])
        original_log = (old / grid.SUMMARY_CSV_NAME).read_bytes()
        (old / "resume-config.txt").write_text("stale-pointer", encoding="utf-8")
        def restarted(command, directory, cwd, trial_id):
            self.assertNotIn("--resume", command)
            self.assertFalse((directory / "resume-config.txt").exists())
            self.assertEqual(grid.log_rows(directory), [])
            return self.fake_run(command, directory, cwd, trial_id)
        code, run = self.run_grid(["--resume"], restarted)
        self.assertEqual((code, run.call_count), (0,2))
        archives = list((self.output / "interrupted-runs").iterdir())
        self.assertEqual(len(archives),1)
        self.assertEqual((archives[0] / grid.SUMMARY_CSV_NAME).read_bytes(), original_log)
        self.assertEqual((archives[0] / "resume-config.txt").read_text(), "stale-pointer")

    def test_unsaved_restart_retains_common_initial_checkpoint(self):
        cp = self.checkpoint(self.root / "base")
        extra = ["--checkpoint", str(cp)]
        def interrupted(command,directory,cwd,trial_id):
            self.summary(directory,[self.metrics(sb=2)])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(extra,interrupted)
        def restarted(command,directory,cwd,trial_id):
            self.assertNotIn("--resume",command)
            s=grid.read_json(Path(command[2]))
            self.assertEqual(s["initial_state"],str(cp / "state.bin"))
            self.assertEqual(s["initial_dataloader_pos"],str(cp / "dataloader_pos.txt"))
            return self.fake_run(command,directory,cwd,trial_id)
        self.assertEqual(self.run_grid([*extra,"--resume"],restarted)[0],0)

    def test_unsaved_restart_handles_truncated_csv_and_dry_run_is_read_only(self):
        def interrupted(command,directory,cwd,trial_id):
            self.summary(directory,[self.metrics(sb=2)])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(child=interrupted)
        directory=grid.trial_dir(self.output,self.plan()["trials"][0])
        path=directory / grid.SUMMARY_CSV_NAME
        path.write_text("epoch,superbatch,test_value_accuracy\n1,",encoding="utf-8")
        with patch.object(grid,"preflight_exe"),redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main([*self.argv,"--resume","--dry-run"]),0)
        self.assertFalse((self.output / "interrupted-runs").exists())
        self.assertEqual(self.run_grid(["--resume"])[0],0)
        archive=next((self.output / "interrupted-runs").iterdir())
        self.assertEqual((archive / grid.SUMMARY_CSV_NAME).read_text(),"epoch,superbatch,test_value_accuracy\n1,")

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
        self.assertEqual(merged["trials"][0]["name"], old["trials"][1]["name"])
        self.assertEqual([t["id"] for t in merged["trials"]], [2, 1, 3])
        rows = grid.summarize(self.output, merged)[1]
        self.assertEqual([r["status"] for r in rows if r["trial"] == 2],
                         ["done", "done", "incomplete", "incomplete", "incomplete", "incomplete", "incomplete"])
        self.assertEqual({r["trial_status"] for r in rows if r["trial"] == 1}, {"done"})

    def test_extension_with_missing_checkpoint_restarts_only_that_condition(self):
        argv, old = self.saved_scale_grid()
        (grid.trial_dir(self.output, old["trials"][1]) / "0002" / "state.bin").unlink()
        def child(command,directory,cwd,trial_id):
            self.assertEqual("--resume" in command,trial_id != 2)
            return self.fake_run(command,directory,cwd,trial_id)
        with patch.object(grid,"preflight_exe"),patch.object(grid,"run_child",side_effect=child),redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main([*argv,"--resume","--epochs","7"]),0)
        archives=list((self.output / "interrupted-runs").iterdir())
        self.assertEqual(len(archives),1)
        self.assertTrue(archives[0].name.startswith(old["trials"][1]["name"]))

    def test_resume_adds_conditions_without_changing_existing_results(self):
        argv, old = self.saved_scale_grid()
        before = {p: p.read_bytes() for p in (self.output / "trials").rglob("*") if p.is_file()}
        requested = [*argv[:-3], "2400", "600", "3000", "--resume"]
        seen = []
        def child(command, directory, cwd, trial_id):
            self.assertNotIn("--resume", command)
            seen.append(trial_id)
            self.summary(directory, [self.metrics(epoch=1), self.metrics(epoch=2)])
            return 0, 1.0
        with patch.object(grid, "preflight_exe"), patch.object(grid, "run_child", side_effect=child), redirect_stdout(io.StringIO()):
            self.assertEqual(grid.main([*requested, "--dry-run"]), 0)
            self.assertEqual(grid.read_json(self.output / grid.MANIFEST), old)
            self.assertEqual(grid.main(requested), 0)
            self.assertEqual(grid.main(requested), 0)
        self.assertEqual(seen, [4, 5])
        merged = grid.read_json(self.output / grid.MANIFEST)
        self.assertEqual([t for t in merged["trials"] if t["id"] <= 3], old["trials"])
        self.assertEqual([t["id"] for t in merged["trials"]], [4, 1, 5, 2, 3])
        self.assertEqual([r["trial"] for r in grid.summarize(self.output, merged)[1]],
                         [4, 4, 1, 1, 5, 5, 2, 2, 3, 3])
        self.assertEqual(merged["axes"]["wrm_target_scaling"], [600, 1200, 1800, 2400, 3000])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual({r["trial"] for r in grid.summarize(self.output, merged)[1]}, {1,2,3,4,5})

    def test_resume_error_lists_changes_without_writing(self):
        argv, old = self.saved_scale_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        grid.atomic_json(self.settings_path, {**self.common, "arch": "SFNN_halfka2_256_8_32"})
        with self.assertRaises(ValueError) as error:
            grid.main([*argv, "--resume", "--epochs", "3"])
        message = str(error.exception)
        self.assertIn("checkpoint-incompatible settings changed: arch", message)
        self.assertIn('requested="SFNN_halfka2_256_8_32"', message)
        self.assertNotIn("max_epochs: saved=", message)
        self.assertEqual(before, (self.output / grid.MANIFEST).read_bytes())
        self.assertEqual(grid.settings_diff({"batches_per_update": 1}, {"batches_per_update": 4}),
                         "  batches_per_update: saved=1, requested=4")
        self.assertIn("saved=null, requested=<not specified>", grid.settings_diff({"x": None}, {}))

    def test_extension_rejects_changed_condition_and_shrink(self):
        argv, old = self.saved_scale_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        for arguments in ([*argv, "--resume", "--epochs", "1"],):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                grid.main(arguments)
        grid.atomic_json(self.settings_path, {**self.common, "no_ft_factorize": True})
        with self.assertRaisesRegex(ValueError, "checkpoint-incompatible"):
            grid.main([*argv, "--resume", "--epochs", "7"])
        self.assertEqual(before, (self.output / grid.MANIFEST).read_bytes())

    def test_resume_common_changes_preserve_historical_epochs_and_original_launch(self):
        argv, old = self.saved_scale_grid()
        directory = grid.trial_dir(self.output, old["trials"][0])
        original = (directory / "bulletou-settings.json").read_bytes()
        grid.atomic_json(self.settings_path, {
            **self.common, "lr": 0.0003, "batches_per_update": 4,
            "superbatches": 8, "save_rate": 0, "validation_rate": 1, "max_epochs": 3,
        })
        requested = [*argv[:-3], "600", "--resume"]
        seen = []
        def child(command, path, cwd, trial_id):
            self.assertEqual(trial_id, 1)
            self.assertIn("--resume", command)
            settings = grid.read_json(Path(command[2]))
            self.assertEqual(settings["lr"], 0.0003)
            self.assertEqual(settings["batches_per_update"], 4)
            self.assertEqual(settings["superbatches"], 8)
            self.assertEqual(settings["wrm_target_scaling"], 600)
            self.summary(path, [*grid.log_rows(path), self.metrics(epoch=3, sb=8)])
            seen.append(trial_id)
            return 0, 1
        with patch.object(grid, "preflight_exe"), patch.object(grid, "run_child", side_effect=child), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(grid.main(requested), 0)
            self.assertEqual(grid.main(requested), 0)
        self.assertEqual(seen, [1])
        self.assertIn("[SETTINGS CHANGED]", out.getvalue())
        self.assertEqual((directory / "bulletou-settings.json").read_bytes(), original)
        rows = [r for r in self.csv_rows(self.output / "grid_summary.csv") if r["trial"] == "1"]
        self.assertEqual([r["lr"] for r in rows], ["0.001", "0.001", "0.0003"])
        self.assertEqual([r["superbatches"] for r in rows], ["4", "4", "8"])
        self.assertEqual([r["batches_per_update"] for r in rows], ["", "", "4"])
        self.assertTrue(all(r["status"] == "done" for r in rows))
        history = grid.read_json(directory / "grid-settings-history.json")["launches"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["settings"]["lr"], 0.0003)
        self.assertEqual(history[0]["log_last_point_before_launch"], {"epoch": 2, "superbatch": 4})

    def test_new_grid_condition_uses_new_common_settings_without_altering_old_results(self):
        argv, old = self.saved_scale_grid()
        grid.atomic_json(self.settings_path, {**self.common, "lr": 0.0003})
        req = grid.make_plan(grid.parse_args([*argv[:-3], "2400", "--resume"]))
        merged, selected = grid.plan_resume(self.output, old, req)
        self.assertEqual(selected, {4})
        self.assertEqual(merged["trials"][0]["settings"]["lr"], 0.0003)
        self.assertEqual(merged["trials"][1:], old["trials"])

    def test_changed_settings_become_columns_with_historical_values(self):
        argv, old = self.saved_scale_grid()
        grid.atomic_json(self.settings_path, {**self.common, "sfnn_qat": True, "max_epochs": 3})
        requested = grid.make_plan(grid.parse_args([*argv, "--resume"]))
        merged, _ = grid.plan_resume(self.output, old, requested)
        trial = merged["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        self.summary(directory, [*grid.log_rows(directory), self.metrics(epoch=3)])
        fields, rows = grid.summarize(self.output, merged)
        rows = [r for r in rows if r["trial"] == trial["id"]]
        self.assertEqual(fields.count("sfnn_qat"), 1)
        self.assertEqual([r["sfnn_qat"] for r in rows], ["", "", True])
        self.assertEqual([r["max_epochs"] for r in rows], [2, 2, 3])
        self.assertEqual(fields[-1], "checkpoint")
        # Removing the option retains its column and past explicit value.
        grid.atomic_json(self.settings_path, {**self.common, "max_epochs": 4})
        requested = grid.make_plan(grid.parse_args([*argv, "--resume"]))
        newer, _ = grid.plan_resume(self.output, merged, requested)
        fields, rows = grid.summarize(self.output, newer)
        rows = [r for r in rows if r["trial"] == trial["id"]]
        self.assertIn("sfnn_qat", fields)
        self.assertEqual([r["sfnn_qat"] for r in rows], ["", "", True, ""])

    def test_changed_column_survives_reverting_before_any_epoch_completes(self):
        old = self.plan()
        grid.atomic_json(self.settings_path, {**self.common, "sfnn_qat": True})
        merged, _ = grid.plan_resume(self.output, old, self.plan())
        grid.atomic_json(self.settings_path, self.common)
        reverted, _ = grid.plan_resume(self.output, merged, self.plan())
        self.assertIn("sfnn_qat", reverted["changed_setting_columns"])
        fields, rows = grid.summarize(self.output, reverted)
        self.assertIn("sfnn_qat", fields)
        self.assertTrue(all(row["sfnn_qat"] == "" for row in rows))

    def test_older_manifest_changes_are_discovered_for_summary_only(self):
        plan = self.plan()
        trial = plan["trials"][0]
        trial["initial_settings"] = {**trial["settings"], "sfnn_qat": False}
        trial["settings"]["sfnn_qat"] = True
        fields, _ = grid.summarize(self.output, plan)
        self.assertIn("sfnn_qat", fields)

    def test_common_changes_without_checkpoint_restart_with_new_launch_settings(self):
        def interrupted(command, directory, cwd, trial_id):
            self.summary(directory, [self.metrics(sb=2)])
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_grid(child=interrupted)
        grid.atomic_json(self.settings_path, {**self.common, "batches_per_update": 4})
        def restarted(command, directory, cwd, trial_id):
            self.assertNotIn("--resume", command)
            settings = grid.read_json(Path(command[2]))
            self.assertEqual(settings["batches_per_update"], 4)
            self.assertEqual(grid.log_rows(directory), [])
            return self.fake_run(command, directory, cwd, trial_id)
        code, calls = self.run_grid(["--resume"], restarted)
        self.assertEqual((code, calls.call_count), (0, 2))
        plan = grid.read_json(self.output / grid.MANIFEST)
        self.assertEqual(plan["trials"][0]["initial_settings"]["batches_per_update"], 4)
        self.assertNotIn("epoch_settings", plan["trials"][0])
        archives = list((self.output / "interrupted-runs").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertNotIn("batches_per_update", grid.read_json(archives[0] / "bulletou-settings.json"))

    def test_grid_axis_overrides_json_and_retains_trial_identity(self):
        self.run_grid()
        old = grid.read_json(self.output / grid.MANIFEST)
        grid.atomic_json(self.settings_path, {**self.common, "lr": 0.02, "validation_rate": 1})
        req = self.plan(["--resume"])
        merged, selected = grid.plan_resume(self.output, old, req)
        self.assertEqual([t["settings"]["lr"] for t in merged["trials"]], [0.0001, 0.0002])
        self.assertEqual([t["name"] for t in merged["trials"]], [t["name"] for t in old["trials"]])
        self.assertEqual(selected, {1, 2})

    def test_rollback_recomputed_epoch_does_not_reuse_old_settings(self):
        argv, old = self.saved_scale_grid()
        grid.atomic_json(self.settings_path, {**self.common, "lr": 0.0003})
        req = grid.make_plan(grid.parse_args([*argv, "--resume", "--epochs", "3"]))
        merged, _ = grid.plan_resume(self.output, old, req)
        trial = merged["trials"][0]
        directory = grid.trial_dir(self.output, trial)
        rows = grid.log_rows(directory)
        rows[-1]["test_value_loss"] = "0.11"
        self.summary(directory, rows)
        result = [r for r in grid.summarize(self.output, merged)[1] if r["trial"] == 1]
        self.assertEqual(result[0]["lr"], 0.001)
        self.assertEqual(result[1]["lr"], 0.0003)

    def test_extension_dry_run_and_lock_failure_do_not_write(self):
        argv, old = self.saved_scale_grid()
        before = (self.output / grid.MANIFEST).read_bytes()
        grid.atomic_json(self.settings_path, {**self.common, "lr": 0.0003, "batches_per_update": 4})
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
