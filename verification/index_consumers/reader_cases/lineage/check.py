# SPDX-License-Identifier: MIT OR Apache-2.0
"""Offline reader controls using fictional records and two catalog editions."""

from dataclasses import replace

from evaluate import Record, build_names, read_reference, transition


def active(display, *aliases):
    return Record(display, (display, *aliases), "active", ())


def check():
    old = {
        "retired": active("Anchor Well"),
        "target": active("Anchor Well"),
        "decoy": active("Research Wharf"),
        "citation": Record("Former program", ("Former program",), "withdrawn", ("retired",)),
        "east": active("East Institute", "North Port"),
        "west": active("West Institute", "North Port"),
        "port": active("North Port"),
        "lab": active("Vale Study Center", "Vale Lab"),
    }
    new = {
        **old,
        "retired": replace(old["retired"], status="withdrawn", successors=("target",)),
        "east": active("North Port"),
        "lab": active("Vale Lab"),
        "new": active("New Discovery"),
    }
    previous = {i: read_reference(old, i) for i in old}
    assert previous["citation"] == ("withdrawn", (("retired", "active"),))
    assert read_reference(new, "citation") == ("withdrawn", (("retired", "withdrawn"),))

    def run(label):
        indexes = build_names(old)
        current, counts = transition(
            old, new, indexes, previous, "first", "second", {"retired": (label, [1])}, None
        )
        assert current == {i: read_reference(new, i) for i in new}
        assert indexes == build_names(new)
        assert counts["stale_via_target_change_only"] == 1
        assert counts["changed_active_name.queries"] == 2
        assert counts["changed_active_name.display.wrong"] == 1
        assert counts["changed_active_name.all_names.ambiguous_with_target"] == 1
        assert counts["changed_active_name.all_names.correct"] == 1
        assert counts["first_seen_active.all_names.empty"] == 1
        return counts

    original = run("target")
    swapped = run("decoy")
    assert original["issue_target_at_intake.all_names.correct"] == 1
    assert original["explicit_duplicate.all_names.correct"] == 1
    assert swapped["issue_target_at_intake.all_names.correct"] == 0
    assert swapped["issue_target_at_intake.all_names.wrong"] == 1
    assert swapped["explicit.new_snapshot_disagrees"] == 1
    print(
        "ROR reader controls passed for ambiguity, stale references, bulk replay and swapped labels"
    )


if __name__ == "__main__":
    check()
