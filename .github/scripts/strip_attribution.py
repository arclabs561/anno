"""Remove AI attribution footers and session links from pull request text.

Run by .github/workflows/strip-attribution.yml. Agent tooling appends a
footer to pull request descriptions after the agent writes them, and
repository settings do not stop it (anthropics/claude-code#107892). An edit
made outside the agent's session stays, so this job makes that edit.

- pull_request_target (opened, edited, reopened): the description of that pull request.
- pull_request_review (submitted, edited): that review.
- workflow_dispatch: every pull request in the repository and its reviews.

Only whole lines that are a footer, a bare session link or a session trailer
are removed, then any trailing "---" separator and blank lines.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

LINE = re.compile(
    r"^\s*(?:"
    r"_?(?:\U0001f916\s*)?generated (?:by|with) \[?claude code\]?(?:\([^)]*\))?\.?_?"
    r"|<?https://claude\.ai/code/session_\S+?>?"
    r"|claude-session:\s*\S+"
    r")\s*$",
    re.IGNORECASE)


def strip(body: str | None) -> str | None:
    """The body without attribution lines; None when nothing changes."""
    if not body:
        return None
    lines = body.replace("\r\n", "\n").split("\n")
    kept = [ln for ln in lines if not LINE.match(ln)]
    if len(kept) == len(lines):
        return None
    while kept and kept[-1].strip() in ("", "---", "***", "___"):
        kept.pop()
    return "\n".join(kept) + "\n"


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def call(self, method: str, path: str, data: dict | None = None):
        req = urllib.request.Request(
            f"https://api.github.com/repos/{self.repo}/{path}", method=method,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read() or b"null")

    def pages(self, path: str):
        page = 1
        while True:
            sep = "&" if "?" in path else "?"
            items = self.call("GET", f"{path}{sep}per_page=100&page={page}")
            if not items:
                return
            yield from items
            page += 1


def fix_pr(gh, number: int, body: str | None) -> bool:
    new = strip(body)
    if new is None:
        return False
    gh.call("PATCH", f"pulls/{number}", {"body": new})
    print(f"pull request #{number}: removed attribution")
    return True


def fix_review(gh, number: int, review: dict) -> bool:
    new = strip(review.get("body"))
    if new is None:
        return False
    try:
        gh.call("PUT", f"pulls/{number}/reviews/{review['id']}", {"body": new})
    except urllib.error.HTTPError as exc:
        # Only a review's author can edit it; GitHub refuses the repository token.
        print(f"::warning::review {review['id']} on #{number}: could not edit ({exc.code}); edit it by hand")
        return False
    print(f"review {review['id']} on #{number}: removed attribution")
    return True


def run(gh, event_name: str, event: dict) -> int:
    changed = 0
    if event_name in ("pull_request_target", "pull_request"):
        pr = event["pull_request"]
        changed += fix_pr(gh, pr["number"], pr.get("body"))
    elif event_name == "pull_request_review":
        changed += fix_review(gh, event["pull_request"]["number"], event["review"])
    elif event_name == "workflow_dispatch":
        for pr in gh.pages("pulls?state=all"):
            changed += fix_pr(gh, pr["number"], pr.get("body"))
            for review in gh.pages(f"pulls/{pr['number']}/reviews"):
                changed += fix_review(gh, pr["number"], review)
    print(f"{changed} item(s) changed")
    return changed


def main() -> int:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as fh:
        event = json.load(fh)
    run(GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"]), event_name, event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
