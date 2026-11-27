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
