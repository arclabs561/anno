"""Check that a speaker-pair diagnostic cannot pass by dropping mentions."""
import unittest

from scripts.qa_speaker_coref import score, transcript


class SpeakerCorefTest(unittest.TestCase):
    def test_known_labels_and_missing_predictions(self):
        for meeting, same, different in [("es2002a", 3, 12), ("es2003a", 4, 24)]:
            with self.subTest(meeting=meeting):
                text, gold, turns = transcript(meeting, "speaker")
                for turn in turns:
                    self.assertEqual(text[turn["start"]:turn["end"]], turn["text"])
                doc = {"text": text, "tracks": {}, "signals": []}
                empty = score(doc, text, gold)
                self.assertEqual(empty["unambiguously_tracked"], 0)
                self.assertEqual(empty["same_speaker"]["unresolved"], same)
                self.assertEqual(empty["different_speaker"]["unresolved"], different)

                for signal_id, (speaker, (start, end)) in enumerate(gold):
                    doc["signals"].append({
                        "id": signal_id, "surface": text[start:end],
                        "location": {"Text": {"start": start, "end": end}},
                    })
                    track = doc["tracks"].setdefault(speaker, {"signals": []})
                    track["signals"].append({"signal_id": signal_id})
                correct = score(doc, text, gold)
                self.assertEqual(correct["same_speaker"]["correct"], same)
                self.assertEqual(correct["different_speaker"]["correct"], different)

                # Multi-track membership also abstains on every affected pair.
                doc["tracks"]["extra"] = {"signals": [{"signal_id": 0}]}
                ambiguous = score(doc, text, gold)
                self.assertEqual(ambiguous["unambiguously_tracked"], len(gold) - 1)
                self.assertEqual(
                    ambiguous["same_speaker"]["unresolved"] + ambiguous["different_speaker"]["unresolved"],
                    len(gold) - 1,
                )


if __name__ == "__main__":
    unittest.main()
