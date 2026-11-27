"""Readers for publisher releases whose web pages and archive sizes hide the actual units."""

import io
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

import ncerb_survey

AG_SPLITS = {"42188205": "train", "42188211": "dev", "42188208": "test"}
HAREM = {
    "CDPrimeiroHAREMprimeiroevento": "first HAREM, first event",
    "CDPrimeiroHAREMMiniHAREM": "first HAREM, MiniHAREM",
    "CDSegundoHAREM": "second HAREM",
}
QUOREF_REVISION = "41fed5bc359f81c2f28259e6572cd1f9aab26e29"
THYME_REVISION = "b669d5e225c26a1eddf542ac2dd07bbe3ea28ba6"
BC2GM_ORIGINAL = "https://raw.githubusercontent.com/spyysalo/bc2gm-corpus/018161681145daf751af037cb890f3a54f62a74d/original-data"
TE3_PLATINUM = "https://api.bitbucket.org/2.0/repositories/leondz/te3-platinum/src/8f38eacd49c98e0a5d5a7d972c0e0b18a48ea4ea"
TE3_FILES = (
    "AP_20130322.tml",
    "CNN_20130321_821.tml",
    "CNN_20130322_1003.tml",
    "CNN_20130322_1243.tml",
    "CNN_20130322_248.tml",
    "CNN_20130322_314.tml",
    "WSJ_20130318_731.tml",
    "WSJ_20130321_1145.tml",
    "WSJ_20130322_159.tml",
    "WSJ_20130322_804.tml",
    "bbc_20130322_1150.tml",
    "bbc_20130322_1353.tml",
    "bbc_20130322_1600.tml",
    "bbc_20130322_332.tml",
    "bbc_20130322_721.tml",
    "nyt_20130321_china_pollution.tml",
    "nyt_20130321_cyprus.tml",
    "nyt_20130321_sarkozy.tml",
    "nyt_20130321_women_senate.tml",
    "nyt_20130322_strange_computer.tml",
)
SOURCES = {
    "AgCNER": [f"https://ndownloader.figshare.com/files/{file}" for file in AG_SPLITS],
    "AMIMeeting": [
        "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    ],
    "AnEM": ["https://www.nactem.ac.uk/anatomy/data/AnEM-1.0.4.tar.gz"],
    # The corpus distributor preserves the original NLM files and public-domain notice.
    "BC2GMFull": [
        f"{BC2GM_ORIGINAL}/{split}/{member}"
        for split in ("train", "test")
        for member in (f"{split}.in", "GENE.eval", "ALTGENE.eval")
    ],
    "CHEMDNER": [
        "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-CHEMDNER-corpus_v2.BioC.xml.gz"
    ],
    "CoQAEntities": [
        "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json?download=1",
        "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json",
    ],
    "CUAD": ["https://zenodo.org/api/records/4595826/files/CUAD_v1.zip/content"],
    "DrugProtBioCreative": [
        "https://zenodo.org/api/records/5119892/files/drugprot-training-development-test-background.zip/content"
    ],
    "HAREM": [
        f"https://www.linguateca.pt/aval_conjunta/HAREM/{name}.xml" for name in HAREM
    ],
    "NCERB": ncerb_survey.SOURCES,
    "NLMChem": ["https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem/NLM-Chem-corpus.zip"],
    "QUOREF": [
        f"https://huggingface.co/datasets/allenai/quoref/resolve/{QUOREF_REVISION}/default/{split}/0000.parquet"
        for split in ("train", "validation")
    ],
    "RAMS": ["https://nlp.jhu.edu/rams/RAMS_1.0c.tar.gz"],
    "SemEval2013Task91": [
        "https://zenodo.org/api/records/10560167/files/DDICorpus-2013.zip/content"
    ],
    "TempEval3": [f"{TE3_PLATINUM}/{name}" for name in TE3_FILES],
    # The THYME authors distribute only offset annotations here. The notes use a separate DUA.
    "THYME": [f"https://codeload.github.com/stylerw/thymedata/zip/{THYME_REVISION}"],
}
BIO = re.compile(r"O|[BI]-.+")
NITE = "{http://nite.sourceforge.net/}"


def bio(lines, columns):
    records = tokens = tagged = 0
    in_sequence = False
    for line in lines:
        fields = line.split()
        if not fields:
            in_sequence = False
            continue
        if len(fields) != columns or not BIO.fullmatch(fields[-1]):
            raise ValueError("Unrecognized BIO token row")
        if columns == 4 and not (int(fields[1]) < int(fields[2])):
            raise ValueError("AnEM token has inconsistent offsets")
        records += not in_sequence
        in_sequence = True
        tokens += 1
        tagged += fields[-1] != "O"
    if not tokens:
        raise ValueError("Empty BIO file")
    return dict(
        unit="nonempty BIO sequences", records=records, tokens=tokens, tagged=tagged
    )


def anem(path):
    rows = []
    seen = set()
    with tarfile.open(path) as archive:
        for split in ("train", "test"):
            prefix = "AnEM-1.0.4/test/"
            with archive.extractfile(prefix + split + "-files.list") as listing:
                fragments = listing.read().splitlines()
            if len(set(fragments)) != len(fragments) or seen.intersection(fragments):
                raise ValueError("Overlapping AnEM train/test source fragments")
            seen.update(fragments)
            with archive.extractfile(prefix + "AnEM." + split) as f:
                counts = bio(io.TextIOWrapper(f, encoding="utf-8"), 4)
            rows.append(
                dict(
                    counts,
                    split_or_subset=split,
                    reason=f"{len(fragments)} article fragments. The alternate development partition reuses training data.",
                )
            )
    return rows


def ami(path):
    tagged = words = remainder = mentions = 0
    with zipfile.ZipFile(path) as z:
        ne = {
            name.removeprefix("namedEntities/").removesuffix(".ne.xml")
            for name in z.namelist()
            if name.startswith("namedEntities/") and name.endswith(".ne.xml")
        }
        tracks = {
            name.removeprefix("words/").removesuffix(".words.xml")
            for name in z.namelist()
            if name.startswith("words/") and name.endswith(".words.xml")
        }
        if not ne or not ne <= tracks:
            raise ValueError("AMI entity layer lacks corresponding word files")
        for track in sorted(tracks):
            filename = track + ".words.xml"
            speech = list(ET.fromstring(z.read("words/" + filename)))
            word_positions = {i for i, entry in enumerate(speech) if entry.tag == "w"}
            if track not in ne:
                remainder += len(word_positions)
                continue
            words += len(word_positions)
            positions = {entry.attrib[NITE + "id"]: i for i, entry in enumerate(speech)}
            if len(positions) != len(speech):
                raise ValueError("Duplicate AMI alignment ID")
            marked = set()
            for entity in ET.fromstring(z.read("namedEntities/" + track + ".ne.xml")):
                if entity.tag != "named-entity":
                    raise ValueError("Unexpected AMI entity annotation")
                children = entity.findall(NITE + "child")
                if not children:
                    raise ValueError("AMI named entity without word links")
                mentions += 1
                for child in children:
                    link = re.fullmatch(
                        r"([^#]+)#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?",
                        child.attrib["href"],
                    )
                    if not link or link[1] != filename:
                        raise ValueError("Invalid AMI word reference")
                    start, end = positions[link[2]], positions[link[3] or link[2]]
                    if start > end:
                        raise ValueError("Reversed AMI word reference")
                    marked.update(range(start, end + 1))
            tagged += len(marked & word_positions)
    return [
        dict(
            split_or_subset="speaker transcripts with a released named-entity layer",
            unit="speaker transcripts",
            records=len(ne),
            tokens=words,
            tagged=tagged,
            reason=f"{mentions} named entities. Count each linked word position once, including transcribed punctuation.",
        ),
        dict(
            split_or_subset="speaker transcripts without a released named-entity layer",
            unit="speaker transcripts",
            records=len(tracks - ne),
            tokens=remainder,
            reason="These word files supply no entity-density denominator.",
        ),
    ]


def coqa(path, split):
    data = json.loads(path.read_bytes())
    if data["version"] != "1.0":
        raise ValueError("Unexpected CoQA version")
    transcripts = set()
    count = 0
    for doc in data["data"]:
        questions = [q["turn_id"] for q in doc["questions"]]
        answers = [a["turn_id"] for a in doc["answers"]]
        if (
            questions != answers
            or len(set(questions)) != len(questions)
            or doc["id"] in transcripts
        ):
            raise ValueError("Unmatched CoQA turns or repeated conversation")
        transcripts.add(doc["id"])
        count += len(questions)
    return dict(
        split_or_subset=split,
        unit="question-answer turns",
        records=count,
        reason=f"{len(transcripts)} conversations. Alternate answers describe the same turns. The test labels are held back.",
    )


def cuad(path):
    with zipfile.ZipFile(path) as z:
        with z.open("CUAD_v1/CUAD_v1.json") as f:
            contracts = json.load(f)["data"]
    ids, categories = set(), set()
    positives = spans = 0
    for contract in contracts:
        for paragraph in contract["paragraphs"]:
            for question in paragraph["qas"]:
                if (
                    question["id"] in ids
                    or bool(question["answers"]) == question["is_impossible"]
                ):
                    raise ValueError("Invalid CUAD question ID or answerability label")
                ids.add(question["id"])
                categories.add(question["question"])
                positives += bool(question["answers"])
                spans += len(question["answers"])
    return dict(
        split_or_subset="complete CUAD v1 QA JSON",
        unit="contract-clause questions",
        records=len(ids),
        reason=f"{len(contracts)} contracts and {len(categories)} clause questions per contract in this release. "
        f"{positives} positive questions with {spans} answer spans. PDF and text copies excluded.",
    )


def rams(path):
    rows = []
    with tarfile.open(path) as archive:
        for split in ("train", "dev", "test"):
            records = tokens = links = 0
            with archive.extractfile(f"RAMS_1.0c/data/{split}.jsonlines") as f:
                for line in f:
                    row = json.loads(line)
                    n = sum(len(s) for s in row["sentences"])
                    if row["split"] != split or len(row["evt_triggers"]) != 1:
                        raise ValueError("Invalid RAMS event window")
                    trigger = tuple(row["evt_triggers"][0][:2])
                    for a, b, _role in row["gold_evt_links"]:
                        if tuple(a) != trigger or not (0 <= b[0] <= b[1] < n):
                            raise ValueError("Invalid RAMS argument link")
                    records += 1
                    tokens += n
                    links += len(row["gold_evt_links"])
            rows.append(
                dict(
                    split_or_subset=split,
                    unit="event-centered windows",
                    records=records,
                    tokens=tokens,
                    reason=f"{links} gold trigger-argument links. Windows can reuse passage tokens.",
                )
            )
    return rows


def nlmchem(path):
    rows = []
    seen = set()
    with zipfile.ZipFile(path) as z:
        corpus = set(z.read("FINAL_v1/pmcids_corpus.txt").decode().split())
        for split in ("train", "dev", "test"):
            ids = z.read(f"FINAL_v1/pmcids_{split}.txt").decode().split()
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("NLM-Chem split list repeats a document")
            seen.update(ids)
            mentions = 0
            for ident in ids:
                (doc,) = ET.fromstring(z.read(f"FINAL_v1/ALL/{ident}_v1.xml")).findall(
                    "document"
                )
                if doc.findtext("id") != ident:
                    raise ValueError("NLM-Chem document disagrees with split list")
                mentions += len(doc.findall("./passage/annotation"))
            rows.append(
                dict(
                    split_or_subset=split,
                    unit="full-text articles",
                    records=len(ids),
                    reason=f"{mentions} BioC passage annotations. Splits use the publisher's PMC ID lists.",
                )
            )
        if corpus != seen:
            raise ValueError("NLM-Chem split union differs from the corpus list")
    return rows


def bc2gm(path, source):
    split, member = source.rsplit("/", 2)[1:]
    ids = set()
    records = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if member.endswith(".in"):
                identifier, text = line.rstrip("\r\n").split(" ", 1)
                if not text.strip() or identifier in ids:
                    raise ValueError("Missing BC2GM sentence or duplicate identifier")
            else:
                identifier, offsets, mention = line.rstrip("\r\n").split("|", 2)
                start, end = map(int, offsets.split())
                if not identifier or not mention or not (0 <= start <= end):
                    raise ValueError("Invalid BC2GM original mention offsets")
            ids.add(identifier)
            records += 1
    if not records:
        raise ValueError("Empty BC2GM original file")
    if member.endswith(".in"):
        unit = "source sentences"
        note = "Includes sentences without a gene mention."
    elif member == "GENE.eval":
        unit = "gold gene mentions"
        note = f"{len(ids)} distinct sentence IDs carry gold annotations."
    else:
        unit = "acceptable alternate gene mentions"
        note = f"{len(ids)} distinct sentence IDs have alternate answers accepted by the scorer."
    return dict(
        split_or_subset=f"{split}/{member}",
        unit=unit,
        records=records,
        reason=f"{note} Original NLM release. BC2GM is a registration of the same task.",
    )


def chemdner(path):
    rows = []
    seen = set()
    with tarfile.open(path) as archive:
        for split in ("training", "development", "evaluation"):
            member = f"BC7T2-CHEMDNER-corpus-{split}.BioC.xml"
            counts = Counter()
            documents = 0
            with archive.extractfile(member) as part:
                # The declaration uses "utf8". Decode the validated UTF-8 bytes before
                # passing them to Expat, which otherwise rejects some multibyte characters.
                for _, doc in ET.iterparse(io.TextIOWrapper(part, encoding="utf-8")):
                    if doc.tag != "document":
                        continue
                    ident = doc.findtext("id")
                    if not ident or ident in seen:
                        raise ValueError("CHEMDNER repeated or missing PubMed ID")
                    seen.add(ident)
                    documents += 1
                    for ann in doc.findall("./passage/annotation"):
                        label = ann.findtext("infon[@key='type']")
                        if label not in {"Chemical", "MeSH_Indexing_Chemical"}:
                            raise ValueError("Unrecognized CHEMDNER BioC annotation")
                        counts[label] += 1
                    doc.clear()
            rows.append(
                dict(
                    split_or_subset=split,
                    unit="CHEMDNER abstracts",
                    records=documents,
                    reason=f"{counts['Chemical']} chemical mention annotations. "
                    f"{counts['MeSH_Indexing_Chemical']} additional MeSH indexing entries in NLM's BioC release. "
                    "BC4CHEMD is a registration of the same shared task.",
                )
            )
    return rows


def drugprot(path):
    rows = []
    with zipfile.ZipFile(path) as z:
        for split in ("training", "development", "test-background"):
            base = "drugprot-gs-training-development/" + split + "/"
            prefix = (
                "test_background_"
                if split == "test-background"
                else "drugprot_" + split + "_"
            )
            # The publisher spells the inner training and development filenames 'abstracs'.
            papers = "abstracts" if split == "test-background" else "abstracs"

            def fields(name, columns):
                with z.open(base + prefix + name + ".tsv") as stream:
                    for line in stream:
                        values = line.rstrip(b"\r\n").split(b"\t")
                        if len(values) != columns or not values[0].isdigit():
                            raise ValueError("Unexpected DrugProt TSV row")
                        yield values

            ids = [r[0] for r in fields(papers, 3)]
            known = set(ids)
            if len(known) != len(ids):
                raise ValueError("Repeated DrugProt abstract ID within a split")
            counts = Counter()
            for name, size in (("entities", 6), ("relations", 4)):
                if split == "test-background" and name == "relations":
                    continue
                for row in fields(name, size):
                    if row[0] not in known:
                        raise ValueError("DrugProt annotation without an abstract")
                    counts[name] += 1
            note = (
                f"{counts['relations']} gold relations."
                if split != "test-background"
                else "Blinded test/background pool. The release supplies no gold relation labels for it."
            )
            rows.append(
                dict(
                    split_or_subset=split,
                    unit="abstracts",
                    records=len(ids),
                    reason=f"{counts['entities']} entity mentions. {note}",
                )
            )
    return rows


def drugner(path):
    groups = defaultdict(Counter)
    seen = set()
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.endswith(".xml"):
                continue
            match = re.fullmatch(
                r"DDICorpus/(Train|Test/Test for DrugNER task|Test/Test for DDI Extraction task)/(DrugBank|MedLine)/[^/]+\.xml",
                member,
            )
            if not match:
                raise ValueError("Unknown DDI 2013 annotation partition")
            part, collection = match.groups()
            doc = ET.fromstring(z.read(member))
            ident = doc.attrib["id"]
            if doc.tag != "document" or ident in seen:
                raise ValueError("Unexpected or duplicate DDI 2013 document")
            seen.add(ident)
            group = groups[part, collection]
            group["documents"] += 1
            for sentence in doc.findall("sentence"):
                group["sentences"] += 1
                for entity in sentence.findall("entity"):
                    if entity.attrib["type"] not in {
                        "drug",
                        "drug_n",
                        "group",
                        "brand",
                    }:
                        raise ValueError("Unknown DrugNER entity class")
                    group["mentions"] += 1
    rows = []
    for split, part in (
        ("shared training", "Train"),
        ("task 9.1 test", "Test/Test for DrugNER task"),
    ):
        for collection in ("DrugBank", "MedLine"):
            group = groups[part, collection]
            excluded = groups["Test/Test for DDI Extraction task", collection][
                "documents"
            ]
            if not group["documents"] or not excluded:
                raise ValueError("Missing DDI 2013 partition")
            rows.append(
                dict(
                    split_or_subset=f"{split}, {collection}",
                    unit="annotated documents",
                    records=group["documents"],
                    reason=f"{group['mentions']} pharmacological entity mentions in {group['sentences']} sentences. "
                    f"The separate task 9.2 test contains {excluded} {collection} documents and is excluded.",
                )
            )
    return rows


def thyme(path):
    layers = defaultdict(Counter)
    docs = defaultdict(set)
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            rel = name.split("/", 1)[1]
            match = re.fullmatch(
                r"coloncancer/(Train|Dev|Test)/([^/]+)/([^/]+)\.(Temporal-Relation|Temporal-Entity|Coreference|UMLS-Entity)\.gold\.completed\.xml",
                rel,
            )
            if not match or match[2] != match[3]:
                raise ValueError("Unrecognized THYME annotation file")
            split, note, _, layer = match.groups()
            ann = ET.fromstring(z.read(name)).find("annotations")
            if ann is None:
                raise ValueError("Missing Anafora annotations")
            docs[split].add(note)
            layers[split][layer, "files"] += 1
            for entry in ann:
                if entry.tag not in {"entity", "relation"} or not entry.findtext(
                    "type"
                ):
                    raise ValueError("Unexpected Anafora annotation")
                layers[split][layer, entry.tag] += 1
    if set(docs) != {"Train", "Dev", "Test"}:
        raise ValueError("Missing THYME annotation split")
    rows = []
    for split in ("Train", "Dev", "Test"):
        counts = layers[split]
        totals = []
        for layer in (
            "Temporal-Relation",
            "Temporal-Entity",
            "Coreference",
            "UMLS-Entity",
        ):
            files = counts[layer, "files"]
            if files:
                totals.append(
                    f"{layer} {files} files/{counts[layer, 'entity']} entities/{counts[layer, 'relation']} relations"
                )
        rows.append(
            dict(
                split_or_subset="colon cancer, " + split,
                unit="note IDs with public gold annotation XML",
                records=len(docs[split]),
                reason=". ".join(totals)
                + ". Layers can overlap. No source notes released here.",
            )
        )
    return rows


def quoref(path, split):
    import pyarrow.parquet as pq

    file = pq.ParquetFile(path)
    ids = set()
    count = 0
    for batch in file.iter_batches(columns=["id", "answers"], batch_size=500):
        for row in batch.to_pylist():
            answer = row["answers"]
            if (
                row["id"] in ids
                or not answer["text"]
                or len(answer["text"]) != len(answer["answer_start"])
            ):
                raise ValueError("Invalid Quoref question or answer spans")
            ids.add(row["id"])
            count += len(answer["text"])
    return dict(
        split_or_subset=split,
        unit="questions with gold answers",
        records=len(ids),
        reason=f"{count} answer spans in the publisher's Parquet conversion. The held-out test answers are absent.",
    )


def platinum(path, name):
    root = ET.parse(path).getroot()
    body = root.find("TEXT")
    if root.tag != "TimeML" or body is None or not root.findtext("DOCID"):
        raise ValueError("Invalid TempEval-3 Platinum TimeML document")
    events = list(body.iter("EVENT"))
    times = list(body.iter("TIMEX3"))
    datelines = root.findall("./DCT/TIMEX3")
    links = root.findall("TLINK")
    for elements, key in ((events, "eid"), (times + datelines, "tid"), (links, "lid")):
        ids = [e.attrib[key] for e in elements]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate TimeML annotation ID")
    return dict(
        split_or_subset=f"Platinum evaluation, {name}",
        unit="adjudicated TimeML documents",
        records=1,
        reason=f"Body events {len(events)}, body time expressions {len(times)}, "
        f"document creation times {len(datelines)} and temporal links {len(links)}. "
        "TimeBank/AQUAINT training, silver data and intermediate annotators excluded.",
    )


def measure(ident, source, path):
    if ident == "TempEval3":
        return [platinum(path, source.rsplit("/", 1)[1])]
    if ident == "BC2GMFull":
        return [bc2gm(path, source)]
    if ident == "AgCNER":
        with path.open(encoding="utf-8-sig") as f:
            return [
                dict(
                    bio(f, 2),
                    split_or_subset=AG_SPLITS[source.rsplit("/", 1)[1]],
                    reason="BIO rows use the publisher's segmented token units.",
                )
            ]
    if ident == "CoQAEntities":
        return [coqa(path, "train" if "train-v1" in source else "dev")]
    if ident == "CUAD":
        return [cuad(path)]
    if ident == "HAREM":
        name = source.rsplit("/", 1)[1].removesuffix(".xml")
        root = ET.parse(path).getroot()
        docs = [d.attrib["DOCID"] for d in root.findall("DOC")]
        if root.tag != "colHAREM" or not docs or len(docs) != len(set(docs)):
            raise ValueError("Invalid HAREM document collection")
        return [
            dict(
                split_or_subset=HAREM[name],
                unit="gold XML documents",
                records=len(docs),
                reason=f"{sum(1 for _ in root.iter('EM'))} EM marks including alternative annotation analyses.",
            )
        ]
    if ident == "NCERB":
        return ncerb_survey.measure(source, path)
    if ident == "QUOREF":
        return [quoref(path, source.split("/")[-2])]
    return {
        "AMIMeeting": ami,
        "AnEM": anem,
        "CHEMDNER": chemdner,
        "DrugProtBioCreative": drugprot,
        "NLMChem": nlmchem,
        "RAMS": rams,
        "SemEval2013Task91": drugner,
        "THYME": thyme,
    }[ident](path)
