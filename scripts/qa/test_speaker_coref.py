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

                # One missing span leaves precisely its incident pairs unresolved.
                first_channel = gold[0][0]
                first_ref = doc["tracks"][first_channel]["signals"].pop(0)
                partial = score(doc, text, gold)
                self.assertEqual(partial["same_speaker"]["unresolved"], 1)
                self.assertEqual(partial["different_speaker"]["unresolved"], len(gold) - 2)
                doc["tracks"][first_channel]["signals"].append(first_ref)

                # Multi-track membership also abstains on every affected pair.
                doc["tracks"]["extra"] = {"signals": [{"signal_id": 0}]}
                ambiguous = score(doc, text, gold)
                self.assertEqual(ambiguous["unambiguously_tracked"], len(gold) - 1)
                self.assertEqual(
                    ambiguous["same_speaker"]["unresolved"] + ambiguous["different_speaker"]["unresolved"],
                    len(gold) - 1,
                )

    def test_malformed_predictions(self):
        text, gold, _ = transcript("es2002a", "speaker")
        start, end = gold[0][1]
        signal = {"id": 0, "surface": text[start:end], "location": {"Text": {"start": start, "end": end}}}
        doc = {"text": text, "signals": [signal], "tracks": {"a": {"signals": [{"signal_id": 0}]}}}
        for fields, message in [
            ({"text": text + " "}, "Prediction text differs"),
            ({"signals": [dict(signal, surface="different"), signal]}, "Duplicate signal ID"),
            ({"signals": []}, "missing signal"),
            ({"signals": [dict(signal, surface="different")]}, "surface disagrees"),
            ({"signals": [dict(signal, surface=text[-1:], location={"Text": {"start": -1, "end": len(text)}})]},
             "Invalid text span"),
            ({"signals": [dict(signal, surface=text[-1:], location={"Text": {"start": len(text)-1, "end": len(text)+1}})]},
             "Invalid text span"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                score({**doc, **fields}, text, gold)


if __name__ == "__main__":
    unittest.main()
