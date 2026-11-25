"""Read dated geographic changes or scoped artist credits in a captured panel."""

import argparse
import json
from datetime import date
from pathlib import Path

SAMPLE = Path(__file__).with_name("sample.json")


def area(panel, edition, code, effective_on):
    """Read direct successor rows published in one edition. Preserve the cited code."""
    day = date.fromisoformat(effective_on).isoformat()
    snapshot = panel["ons"].get(edition)
    result = {"edition": edition, "as_of": day, "cited_code": code}
    if snapshot is None or day[:7] > edition:
        return dict(result, status="outside_edition")
    if code not in snapshot["origins"]:
        return dict(result, status="outside_panel")
    original = snapshot["areas"][code]
    result.update(cited_area=original, changes=[])
    if day < original["operational"]:
        return dict(result, status="before_code")
    if original["terminated"] is None or day <= original["terminated"]:
        return dict(result, status="active")
    rows = [r for r in snapshot["changes"] if r["from"] == code]
    for row in rows:
        if row["effective"] is None or row["effective"] <= day:
            target = row["to"]
            result["changes"].append({
                "code": target, "area": snapshot["areas"].get(target),
                "effective": row["effective"], "order": row["order"],
            })
    if any(c["effective"] is None or c["code"] is None for c in result["changes"]):
        return dict(result, status="incomplete_change")
    if not result["changes"]:
        return dict(result, status="no_dated_target")
    return dict(result, status="one_target" if len(result["changes"]) == 1 else "multiple_targets")


def music(panel, entity, question, role=None, attribute=None):
    """Require a complete typed entity key; cross to a work only through its own key."""
    record = panel["musicbrainz"].get(entity)
    result = {"entity": entity}
    if record is None:
        return dict(result, status="outside_panel")
    if question == "billing":
        if "billing" not in record:
            return dict(result, status="outside_scope")
        text = "".join(c["credited"] + c["join"] for c in record["billing"])
        return dict(result, status="listed", text=text, artists=record["billing"])
    if question == "works":
        if not entity.startswith("recording/"):
            return dict(result, status="outside_scope")
        return dict(result, status="listed" if record["works"] else "unasserted", works=record["works"])
    if question != "role" or not role:
        raise ValueError("A role query requires a relationship name")
    rows = [r for r in record["roles"] if r["role"] == role and (attribute is None or attribute in r["attributes"])]
    return dict(result, status="listed" if rows else "unasserted", artists=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sample", type=Path, default=SAMPLE)
    commands = cli.add_subparsers(dest="question", required=True)
    geo = commands.add_parser("area")
    geo.add_argument("edition")
    geo.add_argument("code")
    geo.add_argument("as_of")
    for name in ("billing", "works", "role"):
        command = commands.add_parser(name)
        command.add_argument("entity")
        if name == "role":
            command.add_argument("role")
            command.add_argument("--attribute")
    args = cli.parse_args()
    panel = json.loads(args.sample.read_text(encoding="utf-8"))
    if args.question == "area":
        result = area(panel, args.edition, args.code, args.as_of)
    else:
        result = music(panel, args.entity, args.question, getattr(args, "role", None), getattr(args, "attribute", None))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
