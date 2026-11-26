"""Read both help modes for every command and print a Markdown option index."""
import argparse
import re
import subprocess

GLOBAL = {"--config", "--color", "--help", "--version"}


def visit(binary, path):
    for mode in ("-h", "--help"):
        help_text = subprocess.check_output([binary, *path, mode], text=True)
        if "Usage: anno" not in help_text:
            raise ValueError(f"Missing usage for {path} {mode}")

    children = []
    options = []
    section = ""
    for line in help_text.splitlines():
        if re.fullmatch(r"[A-Z][\w ]*:", line):
            section = line
        elif section == "Commands:":
            match = re.match(r"^  ([\w-]+)\s+", line)
            if match and match[1] != "help":
                children.append(match[1])
        elif section == "Options:":
            match = re.match(r"^ {2,6}(-[\w], )?(--[\w-]+)\b", line)
            if match and match[2] not in GLOBAL:
                options.append(f"`{match[0].strip()}`")
    usage = re.search(r"^Usage: (.+)$", help_text, re.MULTILINE)[1]
    print(f"| `{usage}` | {', '.join(options) or '—'} |")
    for child in children:
        visit(binary, [*path, child])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    args = parser.parse_args()
    print("""# Command and option index

Generated from a build with `eval`, `discourse`, and `extractor-readability`.
Use `anno help <command>` for values and examples in your installation.
Every route accepts `--config`, `--color`, `-h, --help` and `-V, --version`.
Parent options appear on the parent row. Put them before the nested action.
See the [quickstart](QUICKSTART.md) for profile applicability.

| Invocation | Local options |
| --- | --- |""")
    visit(args.binary, [])
