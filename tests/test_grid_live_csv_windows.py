import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import grid_search


@unittest.skipUnless(os.name == "nt", "Windows file locking semantics")
class LiveSummarySharingTests(unittest.TestCase):
    def test_live_reader_waits_for_writer_lock(self):
        import msvcrt
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary-learn.csv"
            summary.write_text("epoch,superbatch\n1,1\n", encoding="utf-8")
            lock_path = root / "summary-learn.lock"
            result = []
            with lock_path.open("a+b") as lock:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                worker = threading.Thread(target=lambda: result.extend(grid_search.log_rows(root, live=True)))
                worker.start()
                try:
                    time.sleep(0.2)
                    self.assertTrue(worker.is_alive())
                    replacement = root / "replacement.csv"
                    replacement.write_text("epoch,superbatch\n1,2\n", encoding="utf-8")
                    os.replace(replacement, summary)
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                    worker.join(10)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, [{"epoch": "1", "superbatch": "2"}])


if __name__ == "__main__":
    unittest.main()
