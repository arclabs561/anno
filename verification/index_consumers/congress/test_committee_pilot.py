# SPDX-License-Identifier: MIT OR Apache-2.0
"""Identity hiding, prior-only reads and the same-publication join in the replay."""

import unittest

from committee_pilot import Directory, run


def edition(revision, day, upserts):
    return {
        "revision": revision,
        "published_at": f"2022-01-{day:02d}T12:00:00Z",
        "upsert": upserts,
        "remove": [],
    }


def member(key, first, last, chamber="rep"):
    return {
        "id": key,
        "first": first,
        "last": last,
        "official_full": None,
        "chamber": chamber,
    }


class PublicationOrderTests(unittest.TestCase):
    def test_distinct_chambers_and_tied_names(self):
        view = Directory(
            {
                k: member(k, "River", "Bell", chamber)
                for k, chamber in [("a", "rep"), ("b", "sen")]
            }
        )
        self.assertEqual(view.rank("River Bell", "rep")[:2], ("a", 1))
        self.assertEqual(view.rank("River Bell", "sen")[:2], ("b", 1))
        self.assertEqual(view.rank("River Bell", None)[:2], (None, 1))
        self.assertEqual(view.rank("Other", "rep"), (None, 0, frozenset()))

    def test_same_timestamp_requires_same_revision(self):
        catalog = [
            edition("base", 1, [member("a", "Rae", "Vale")]),
            edition("change", 6, [member("b", "Kit", "Aster")]),
        ]
        for revision, expect_atomic in [("change", 1), ("other-version", 0)]:
            printings = [edition(revision, 6, [["HSAG", "Kit Aster", 0, "b", False]])]
            policies = run(printings, catalog)["policy"]
            self.assertEqual(policies["prior-1"]["counts"]["correct"], 0)
            self.assertEqual(policies["atomic-1"]["counts"]["correct"], expect_atomic)
            self.assertEqual(policies["weekly-1"]["counts"]["correct"], 1)

    def test_later_evidence_does_not_change_earlier_rank(self):
        printings = [edition("print", 2, [["HSAG", "Rae Vale", 0, "a", False]])]
        earlier = edition("base", 1, [member("a", "Rae", "Vale")])
        future = edition("future", 8, [member("b", "Rae", "Vale")])
        for catalog in ([earlier], [earlier, future]):
            policies = run(printings, catalog)["policy"]
            self.assertEqual(policies["prior-1"]["counts"]["correct"], 1)
            self.assertEqual(policies["atomic-1"]["counts"]["correct"], 1)


if __name__ == "__main__":
    unittest.main()
