import csv
from pathlib import Path
import survey_slue_manifest
import survey_scrolls_qmsum
fields="id registry_url repository revision path split status reason source_url sha256 bytes metric unit value".split()
base=dict(id="SLUE",registry_url="https://github.com/asappresearch/slue-toolkit",repository="asappresearch/slue-toolkit",revision="ed23039507c8f01b704a5c43ed89b6e808a49405")
rows=survey_slue_manifest.survey(base)
rows+=survey_scrolls_qmsum.survey("https://github.com/tau-nlp/scrolls",Path("/tmp"))
with open("public-release-counts.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(Path("public-release-counts.csv").read_text())

if not any(row["id"] == "SLUE" and row["status"] == "ok" for row in rows):
    import urllib.request
    import urllib.error
    import xml.etree.ElementTree as ET
    for method in ("HEAD","GET"):
        try:
            req=urllib.request.Request(survey_slue_manifest.SOURCE,method=method)
            with urllib.request.urlopen(req,timeout=25) as resp:
                print("VoxPopuli ordinary", method, resp.status, "type",resp.headers.get("Content-Type"),"bytes",resp.headers.get("Content-Length"))
        except urllib.error.HTTPError as e:
            code=None
            if method=="GET":
                try: code=ET.fromstring(e.read(1800)).findtext("Code")
                except ET.ParseError: pass
            print("VoxPopuli ordinary", method,e.code,"S3 code",code)
        except OSError as e:
            print("VoxPopuli ordinary", method,type(e).__name__)
