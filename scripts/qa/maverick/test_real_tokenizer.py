"""Check the tokenizer attached to the verified, offline Maverick worker model."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from download_maverick_weights import PINS
from maverick_worker import load_model


class WorkerTokenizer(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MAVERICK_MODEL_PATH"), "requires verified Maverick weights")
    def test_preserves_trained_unicode_normalization(self):
        tokenizer = load_model(os.environ["MAVERICK_MODEL_PATH"], PINS).tokenizer
        # Recorded on the same pinned spm.model with Transformers 4.48.3.
        cases = [
            ("oﬃce", "office", [832], [(0, 4)]),
            ("2040‑02‑29", "2040‐02‐29", [50187, 8325, 4159, 8325, 3873],
             [(0, 4), (4, 5), (5, 7), (7, 8), (8, 10)]),
            ("1\u00a002", "1 02", [376, 7732], [(0, 1), (1, 4)]),
            ("２０４０", "2040", [50187], [(0, 4)]),
            ("½", "1⁄2", [14624], [(0, 1)]),
            ("［ＵＮＫ］", "[UNK]", [3], [(0, 5)]),
            ("[SPEAKER_START]", "[SPEAKER_START]", [128001], [(0, 15)]),
            ("[SPEAKER_END]", "[SPEAKER_END]", [128002], [(0, 13)]),
        ]
        sequence_ids = [1]  # [CLS]
        for source, normalized, ids, offsets in cases:
            sequence_ids.extend(ids)
            with self.subTest(source=source):
                self.assertEqual(tokenizer.backend_tokenizer.normalizer.normalize_str(source), normalized)
                encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
                self.assertEqual(encoded["input_ids"], ids)
                self.assertEqual(encoded["offset_mapping"], offsets)
        sequence_ids.append(2)  # [SEP]
        sequence = tokenizer([case[0] for case in cases], is_split_into_words=True)
        self.assertEqual(sequence["input_ids"], sequence_ids)


if __name__ == "__main__":
    unittest.main()
