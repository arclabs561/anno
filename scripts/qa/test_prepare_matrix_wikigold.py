import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("prepare_matrix_wikigold.sh")


class PinnedWikiGoldTest(unittest.TestCase):
    def test_failed_refresh_leaves_no_loadable_corpus(self):
        # Both a failed transfer and a successful transfer with the wrong
        # content must leave the following PR sampler with no WikiGold cell.
        for curl_status in (22, 0):
            with self.subTest(curl_status=curl_status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                datasets = root / "datasets"
                datasets.mkdir()
                (datasets / "WikiGold.cache").write_text("Paris B-LOC\n")
                curl = root / "curl"
                curl.write_text(f'#!/usr/bin/env bash\nprintf "wrong download" > "${{!#}}"\nexit {curl_status}\n')
                curl.chmod(0o755)
                env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", ANNO_CACHE_DIR=tmp)
                run = subprocess.run(["bash", SCRIPT], env=env, capture_output=True)
                self.assertNotEqual(run.returncode, 0)
                self.assertEqual(list(datasets.iterdir()), [])
