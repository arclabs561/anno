"""Exercise the optional CorPipe document adapter with CorefUD output."""

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "crates/anno-cli/src/cli/commands")
)
from corpipe import ground

CONLLU = """# newdoc
# global.Entity = eid-etype-head-other
# sent_id = 1
# text = Éva du Rhin.
1	Éva	Éva	PROPN	_	_	0	root	_	Entity=(e1--1)|TokenRange=0:3
2-3	du	_	_	_	_	_	_	_	TokenRange=4:6
2	de	de	ADP	_	_	4	case	_	_
3	le	le	DET	_	_	4	det	_	Entity=(e2--2
4	Rhin	Rhin	PROPN	_	_	1	nmod	_	Entity=e2)|TokenRange=7:11
4.1	_	_	_	_	_	_	_	4:dep	Entity=(e1--1)
5	.	.	PUNCT	_	_	1	punct	_	TokenRange=11:12

"""


@unittest.skipUnless(
    importlib.util.find_spec("udapi"), "install optional CorPipe dependencies"
)
class GroundCorpipe(unittest.TestCase):
    def test_source_ranges_and_implicit_mentions(self):
        clusters = {c["id"]: c["mentions"] for c in ground("Éva du Rhin.", CONLLU)}
        eva, implicit = clusters["e1"]
        self.assertEqual((eva["start"], eva["end"], eva["text"]), (0, 3, "Éva"))
        self.assertNotIn("start", implicit)
        self.assertEqual(
            implicit["tokens"],
            [{"alignment": "empty_node", "sentence": "1", "word_id": "4.1"}],
        )
        contraction = clusters["e2"][0]
        self.assertEqual(contraction["text"], "du Rhin")
        self.assertEqual(contraction["tokens"][0]["text"], "du")
        self.assertEqual(contraction["tokens"][0]["alignment"], "multiword_token")
        self.assertEqual(contraction["tokens"][0]["word_id"], "3")

    def test_reject_out_of_bounds_input(self):
        with self.assertRaisesRegex(ValueError, "Out-of-bounds"):
            ground("Éva", CONLLU)


if __name__ == "__main__":
    unittest.main()
