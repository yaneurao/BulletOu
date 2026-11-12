import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grid_search


class FtCheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def checkpoint(self, epoch, sb, factorizer, *, total=100):
        directory = self.root / f"{epoch:04}-{sb:04}sb"
        directory.mkdir()
        (directory / "state.bin").write_bytes(b"state")
        (directory / "dataloader_pos.txt").write_text("12,0\n")
        (directory / "learn.log").write_text(f"epoch,superbatch\n{epoch},{sb}\n")
        (directory / "epoch_superbatches.txt").write_text(str(total))
        (directory / "bulletou-settings.json").write_text(json.dumps({
            "superbatches": total, "sfnn_ft_factorizer": factorizer,
        }))
        return directory

    def settings(self, factorizer):
        return {"superbatches": 100, "sfnn_ft_factorizer": factorizer}

    def test_change_after_checkpoint_keeps_on_and_allows_fold_off(self):
        self.checkpoint(4, 100, {"epoch1": True, "epoch5": False})
        self.assertTrue(grid_search.ft_factorizer_schedule_compatible(
            self.root, self.settings({"epoch1": True})))
        self.assertTrue(grid_search.ft_factorizer_schedule_compatible(
            self.root, self.settings({"epoch1": True, "epoch5": False})))

    def test_off_checkpoint_cannot_resume_with_factorizer_on(self):
        self.checkpoint(4, 100, True)
        self.checkpoint(5, 100, {"epoch1": True, "epoch5": False})
        self.assertFalse(grid_search.ft_factorizer_schedule_compatible(
            self.root, self.settings({"epoch1": True})))

    def test_mid_epoch_uses_current_epoch_and_missing_metadata_fails_closed(self):
        d = self.checkpoint(4, 50, True)
        self.assertTrue(grid_search.ft_factorizer_schedule_compatible(
            self.root, self.settings({"epoch1": True, "epoch5": False})))
        (d / "bulletou-settings.json").unlink()
        self.assertFalse(grid_search.ft_factorizer_schedule_compatible(
            self.root, self.settings({"epoch1": True})))

    def test_summary_is_backed_up_only_when_rolling_back(self):
        self.checkpoint(4, 100, True)
        summary = self.root / grid_search.SUMMARY_CSV_NAME
        original = b"eval,epoch,superbatch\na,4,100\na,5,1\n"
        summary.write_bytes(original)
        backup = grid_search.backup_summary_before_rollback(self.root)
        self.assertIsNotNone(backup)
        self.assertEqual(backup.read_bytes(), original)
        self.assertEqual(summary.read_bytes(), original)
        summary.write_bytes(b"eval,epoch,superbatch\na,4,100\n")
        self.assertIsNone(grid_search.backup_summary_before_rollback(self.root))


if __name__ == "__main__":
    unittest.main()
