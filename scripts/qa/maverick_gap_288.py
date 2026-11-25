"""Run the fixed named row, 256 fresh GAP controls, and 48 historical controls against a verified, offline Maverick worker.

Place beside scripts/qa/coref_gap.py. Only first-party aggregate counts and ordered output hashes are printed.
Pass --test and --validation with GAP TSV files fetched from commit
83135f26025cdd705c04f77e940d0c9201116b80; all file digests are verified.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

from coref_gap import DATA_SHA, check_file, measure

VALIDATION = (246, 134, 313, 195, 2, 118, 25, 289, 158, 84, 71, 344, 151, 64, 316, 430, 208, 89, 226, 432, 60, 99, 368, 187, 30, 53, 272, 372, 204, 451, 38, 440, 380, 364, 429, 5, 41, 32, 194, 420, 231, 20, 446, 173, 108, 322, 141, 7, 122, 217, 345, 294, 448, 260, 375, 274, 404, 276, 8, 296, 154, 301, 147, 4, 348, 152, 443, 425, 110, 74, 238, 167, 337, 286, 9, 86, 92, 29, 224, 81, 58, 184, 171, 308, 48, 27, 323, 155, 85, 3, 33, 46, 56, 35, 113, 119)
TEST = (875, 1706, 1576, 1755, 947, 4, 1709, 972, 620, 1407, 1395, 887, 797, 1371, 529, 843, 43, 1803, 266, 1734, 130, 562, 701, 1298, 1304, 1445, 1884, 628, 158, 1911, 208, 888, 1000, 673, 925, 1051, 471, 1381, 36, 650, 1981, 1380, 1578, 974, 138, 1472, 1345, 874, 1655, 496, 1851, 1208, 793, 1248, 1078, 1782, 1226, 886, 1705, 1907, 532, 1723, 771, 1569, 427, 836, 295, 651, 1965, 937, 1295, 1944, 1896, 1847, 1485, 272, 248, 574, 652, 21, 476, 1912, 1558, 170, 1009, 1070, 1626, 894, 85, 1374, 904, 601, 109, 1114, 1303, 1018, 1927, 210, 853, 288, 1501, 858, 147, 813, 607, 1717, 1926, 1454, 638, 331, 829, 1093, 1858, 1922, 1898, 508, 220, 1123, 970, 1621, 1731, 11, 137, 1541, 1108, 1135, 1658, 1196, 572, 1049, 226, 1476, 1737, 603, 1978, 626, 1460, 502, 1622, 1564, 1260, 48, 691, 365, 863, 297, 1025, 1012, 878, 1259, 995, 685, 1850, 561, 175, 598, 514, 237, 1659, 881)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    check_file(args.test, DATA_SHA)
    check_file(args.validation, "2d784f66b390404f554704b9aef6dcde8845e79dda9886b8391cf7e9a24fdb98")
    ids = ["test-1845"] + [f"validation-{n}" for n in VALIDATION] + [f"test-{n}" for n in TEST]
    digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    assert digest == "a76cd45d822c8d7abd003984f648b164d4fda1859427a238daa4eb4c631982da"
    historical = [f"validation-{n}" for n in (297, 186, 310, 101, 266, 365, 162, 350, 405, 398, 189, 378, 111, 90, 228, 436, 435, 72, 23, 399, 357, 150, 66, 268, 212, 340, 112, 114, 203, 129, 280, 298, 6, 93, 269, 51, 62, 209, 293, 31, 73, 180, 247, 76, 146, 320, 307, 165)]
    assert hashlib.sha256("\n".join(historical).encode()).hexdigest() == "12f3c9f1e2bd0117ed5bc42621d625c586e3e18f2e9e8141e3aace4790cd9560"
    ids.extend(historical)
    rows = {}
    for path in (args.test, args.validation):
        with path.open(newline="", encoding="utf8") as f:
            rows.update({r["ID"]: r for r in csv.DictReader(f, delimiter="\t")})
    docs = []
    for key in ids:
        row = rows[key]
        spans = []
        for s in ("Pronoun", "A", "B"):
            lo = int(row[s + "-offset"])
            hi = lo + len(row[s])
            assert row["Text"][lo:hi] == row[s]
            spans.append((lo, hi))
        docs.append((key, row["Text"], spans, [row[k + "-coref"] == "TRUE" for k in ("A", "B")]))
    inputs = "".join(json.dumps(dict(id=i, text=t)) + "\n" for i,t,_,_ in docs)
    env = dict(os.environ, ANNO_NO_DOWNLOADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    out = subprocess.run([args.binary.resolve(), "maverick"], input=inputs, stdout=subprocess.PIPE,
                         check=True, text=True, env=env)
    results = [json.loads(line) for line in out.stdout.splitlines()]
    assert len(results) == len(docs)
    for label,s in (("named",slice(1)),("validation",slice(1,97)),("test",slice(97,257)),("historical_48",slice(257,None))):
        scored = measure(docs[s], results[s], "maverick")
        decisions = [(record["id"], bits) for record, bits in zip(results[s], scored["per_id"], strict=True)]
        partitions = [(record["id"], sorted(sorted(cluster) for cluster in record["clusters"])) for record in results[s]]
        def digest(rows):
            return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode("ascii")).hexdigest()
        print(json.dumps(dict(panel=label, counts=scored["counts"],
                              decisions_sha256=digest(decisions), partitions_sha256=digest(partitions))))

if __name__ == "__main__":
    main()
