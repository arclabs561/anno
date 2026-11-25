"""Exercise the production release checks without a registry or publishing credentials."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rust_release as release


class ReleaseTest(unittest.TestCase):
    def test_only_synchronized_release_ref_or_main_pass(self):
        metadata = {
            "packages": [{"name": name, "id": name, "version": "0.16.0"} for name in release.CRATES],
            "workspace_members": list(release.CRATES),
        }
        for ref in ("refs/tags/v0.16.0", "refs/heads/main"):
            self.assertEqual(release.release_version(metadata, ref), "0.16.0")
        for ref in ("refs/tags/v0.15.0", "refs/tags/vgarbage", "refs/heads/feature"):
            with self.assertRaises(ValueError):
                release.release_version(metadata, ref)
        metadata["packages"][1]["version"] = "0.15.0"
        with self.assertRaises(ValueError):
            release.release_version(metadata, "refs/heads/main")
        metadata["packages"].pop(1)
        with self.assertRaises(ValueError):
            release.release_version(metadata, "refs/heads/main")

    def test_index_requires_exact_name_version_digest_and_unyanked(self):
        valid = dict(name="anno", vers="0.16.0", yanked=False, cksum="a" * 64)
        for changes, expectation in [({}, True), ({"vers": "0.15.0"}, False),
                                     ({"name": "other"}, None), ({"yanked": True}, None),
                                     ({"cksum": "b" * 64}, None)]:
            with self.subTest(changes=changes):
                row = io.BytesIO(json.dumps({**valid, **changes}).encode())
                with patch.object(release, "urlopen", return_value=row):
                    if expectation is None:
                        with self.assertRaises(ValueError):
                            release.indexed("anno", "0.16.0", "a" * 64)
                    else:
                        self.assertEqual(release.indexed("anno", "0.16.0", "a" * 64), expectation)

    def test_partial_publish_and_retry_need_registry_evidence(self):
        payload = b"candidate archive"
        digest = hashlib.sha256(payload).hexdigest()
        for observations, cargo_code, expected_commands, completes in [
            ([True], 0, ["package"], True),
            ([False, False, True], 0, ["package", "publish"], True),
            ([False, True], 101, ["package", "publish"], True),
            ([False] * 7, 0, ["package", "publish"], False),
            ([False] * 7, 101, ["package", "publish"], False),
        ]:
            with self.subTest(observations=observations, cargo_code=cargo_code), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                (target / "package").mkdir()
                (target / "package" / "anno-0.16.0.crate").write_bytes(payload)
                with patch.object(release.subprocess, "run") as cargo, \
                     patch.object(release, "indexed", side_effect=observations) as index, \
                     patch.object(release.time, "sleep"):
                    cargo.return_value.returncode = cargo_code
                    if completes:
                        release.publish("anno", "0.16.0", target)
                    else:
                        with self.assertRaises(RuntimeError):
                            release.publish("anno", "0.16.0", target)
                    self.assertTrue(all(call.args == ("anno", "0.16.0", digest) for call in index.call_args_list))
                    self.assertEqual([call.args[0][1] for call in cargo.call_args_list], expected_commands)


if __name__ == "__main__":
    unittest.main()
