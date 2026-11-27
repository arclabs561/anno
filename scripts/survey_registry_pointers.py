"""Check the public landing URLs of the registered datasets without reading corpus contents."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from dataset_file_survey import CONTROLLED, local_source_ids

FIELDS = (
    "id",
    "url",
    "checked_utc",
    "pointer_status",
    "method",
    "http_status",
    "final_host",
    "content_type",
)


def probe(url):
    # Header-only responses suffice. Some hosts explicitly reject HEAD.
    for method in ("HEAD", "GET"):
        headers = {
            "User-Agent": "Anno registry source audit (github.com/arclabs561/anno)"
        }
        if method == "GET":
            headers["Range"] = "bytes=0-0"
        try:
            response = urlopen(Request(url, method=method, headers=headers), timeout=15)
        except HTTPError as exc:
            response = exc
        except (URLError, OSError, ValueError):
            return dict(pointer_status="connection failed", method=method)
        with response:
            code = response.status
            if method == "HEAD" and code in (400, 403, 404, 405, 410, 501):
                continue
            final_host = urlsplit(response.geturl()).hostname or ""
            if 200 <= code < 300 and final_host in {
                "accounts.google.com",
                "login.microsoftonline.com",
            }:
                status = "sign-in page"
            elif 200 <= code < 300:
                status = "responded"
            elif code in (404, 410):
                status = "missing at this URL"
            elif code in (401, 403):
                status = "request denied"
            else:
                status = "other HTTP response"
            return dict(
                pointer_status=status,
                method=method,
                http_status=code,
                final_host=final_host,
                content_type=(
                    response.headers.get_content_type()
                    if "content-type" in response.headers
                    else ""
                ),
            )


def survey(rows, workers, date):
    local = local_source_ids()
    observations = []
    for row in rows:
        obs = dict.fromkeys(FIELDS, "")
        obs.update(id=row["id"], url=row["url"], checked_utc=date)
        if row["id"] in local:
            obs["pointer_status"] = "local-source policy"
        elif row["access"] in CONTROLLED:
            obs["pointer_status"] = f"registry access: {row['access']}"
        elif not row["url"]:
            obs["pointer_status"] = "no registry URL"
        elif urlsplit(row["url"]).scheme not in {"https", "http"}:
            obs["pointer_status"] = "non-HTTP registry URL"
        observations.append(obs)
    urls = sorted({obs["url"] for obs in observations if not obs["pointer_status"]})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = dict(zip(urls, pool.map(probe, urls)))
    for obs in observations:
        if not obs["pointer_status"]:
            obs.update(outcomes[obs["url"]])
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.inventory.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    observations = survey(
        rows, args.workers, datetime.now(timezone.utc).date().isoformat()
    )
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations)


if __name__ == "__main__":
    main()
