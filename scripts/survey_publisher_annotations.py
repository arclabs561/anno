"""Reviewed corpus-to-file mappings and parsers for the GitHub source survey.

File names select a documented representation of each release. SCOPES records
source layers and aliases for individual registry IDs. MAVEN and MAVEN-ARG
combine their publisher-linked files with commit-pinned repository metadata.
"""

import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from threading import Lock

# Author releases identified from dataset titles, papers and release READMEs.
# LGL is the published copy in Gritta et al.'s comparative geoparsing release.
AUTHOR_REPOS = {
    "AgriNER": "Tec4Tric/AgriNER",
    "AnEM": "ncbi/AIONER",
    "CoMTA": "Khan/tutoring-accuracy-dataset",
    "CoNLL2002": "nltk/nltk_data",
    "CoNLL2002Dutch": "nltk/nltk_data",
    "CoNLL2002Spanish": "nltk/nltk_data",
    "CopticScriptorium": "UniversalDependencies/UD_Coptic-Scriptorium",
    "DrugProtBioCreative": "ncbi/AIONER",
    "NLMChem": "ncbi/AIONER",
    "FinNER": "mpsilfve/finer-data",
    "GeoWebNews": "milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation",
    "LGL": "milangritta/Pragmatic-Guide-to-Geoparsing-Evaluation",
    "LegalCore": "WeiKangda/LegalCore",
    "MHERCL": "arianna-graciotti/historical-entity-linking",
    "MathDial": "eth-nlped/mathdial",
    "MixRED": "acddca/MixRED",
    "PolyIE": "jerry3027/PolyIE",
    "RISeC": "YiweiJiang2015/RISeC",
    "ScrollsQMSum": "Yale-LILY/QMSum",
}

# FantasyCoref publishes original annotated public-domain texts. ISNotes
# grants research access to its standoff annotations separately from OntoNotes.
PUBLIC_ANNOTATIONS = {
    "FantasyCoref",
    "ISNotes",
    "CopticScriptorium",
    "CoNLL2002Dutch",
    "CoNLL2002Spanish",
}

PATHS = {
    "AgriNER": r"AgriNer_Code/(train|test)_data\.json",
    "BioNERLLaMA": r"datasets/(BC2GM|BC5CDR-chemical|BC5CDR-disease|NCBI-disease|NLMChem|NLMGene)_(train|test)\.txt",
    "AnEM": r"data/conll/AnEM_(train)\.conll",
    "CoMTA": r"CoMTA_dataset\.json",
    "CopticScriptorium": r"cop_scriptorium-ud-(train|dev|test)\.conllu",
    "DrugProtBioCreative": r"data/pubtator/DrugProt_(Test|Train)\.PubTator",
    "NLMChem": r"data/pubtator/NLMchem_(Test|TrainDev)\.PubTator",
    "BELB": r"src/belb/assets/offsets/([^/]+)\.json",
    "ScrollsQMSum": r"data/ALL/(?:(train)/[^/]+\.json|jsonl/(val|test)\.jsonl)",
    "CSN": r"resources/annotationStore\.csv",
    "CodeSearchNet": r"resources/annotationStore\.csv",
    "FantasyCoref": r"data/(?:(GFT_split1)/(train|dev|test)|(AFT_original)/(test))\.english\.v4_gold_conll",
    "FinNER": r"data/(digitoday\.2014\.(?:train|dev)|digitoday\.2015\.test|wikipedia\.test)\.csv",
    "GeoWebNews": r"data/GWN\.xml",
    "LGL": r"data/Corpora/lgl\.xml",
    "LegalCore": r"data/data\.jsonl",
    "MAVEN": r"docid2topic\.json",
    "MELO": r"data/processed/melo/([^/]+)/annotations\.tsv",
    "MHERCL": r"benchmark/v0\.1/mhercl_v012_nodup_final\.tsv",
    "MathDial": r"data/(train|test)\.jsonl",
    "PolyIE": r"Cleaned_data/Final_v2/(train|test|validation)_split\.txt",
    # The combined code-mixed set is derived from the three variants. Keep the
    # variant-level records. EN and ZH are aligned monolingual versions.
    "MixRED": r"(EN|ZH|EN_ZH/(?:entity|inter-sent|intra-sent))/(train_annotated|dev|test)\.jsonl",
    "MultiWOZNER": r"data/MultiWOZ_2\.2/(train|dev|test)/dialogues_\d+\.json",
    "TaggedPBCEsperanto": r"corpora/conllu/epo-eng-tagged-epo-x-bible_parsed\.conllu",
    "TaggedPBCKlingon": r"corpora/conllu/tlh-eng-tagged-tlh-x-bible_parsed\.conllu",
    "TwitterGMNER": r"data_processing/MNER_processing/twitter-GMNER/new(train|dev|test)\.txt",
}

# A single download checks thousands of small files without one HTTP request
# per document. The capture (if present) distinguishes the actual directories.
ARCHIVE_PATHS = {
    "CASIE": r"data/annotation/\d+\.json",
    "CRAFTCoref": r"coreference-annotation/knowtator-2/\d+\.xml",
    "CRAFTCorpusCoref": r"coreference-annotation/knowtator-2/\d+\.xml",
    "ISNotes": r"ISAnnotationWithoutTrace/wsj_\d+_entity_level\.xml",
    "Mahanama": r"data/mahanama_conllu/(volume_\d+)/[^/]+\.conllu",
    "NNE": r"ACL2019 Paper/Resources/no-text/(\d\d)/wsj_\d+\.mrg",
    "RISeC": r"data/(train|dev|test)/[^/]+\.ann",
}

SCOPES = {
    "AnEM": "NCBI AIONER's released training conversion; artificial sentence boundary tokens excluded",
    "BioNERLLaMA": "publisher sentences with inline entity markup; no density for a file with unbalanced source markup",
    "BELB": "corrected entity mention spans in eight BELB input corpora; full corpora and knowledge bases excluded",
    "ScrollsQMSum": "original QMSum source meetings and queries; these questions also occur in the separately measured SCROLLS transformation",
    "CSN": "CodeSearchNet challenge relevance judgments; source-code function corpus excluded",
    "CodeSearchNet": "CodeSearchNet challenge relevance judgments; source-code function corpus excluded",
    "CopticScriptorium": "publisher UD treebank only; other Scriptorium corpus editions and layers excluded",
    "CRAFTCoref": "same CRAFT coreference files as CRAFTCorpusCoref",
    "CRAFTCorpusCoref": "same CRAFT coreference files as CRAFTCoref",
    "DrugProtBioCreative": "NCBI AIONER's PubTator entity layer; chemical-protein relation labels excluded",
    "ISNotes": "publisher's without-trace standoff layer only; licensed OntoNotes text excluded",
    "MELO": "judged occupation pairs by retrieval configuration; occupations may recur across configurations",
    "MHERCL": "final deduplicated release; invalid BIO labels preclude whole-file entity density",
    "MixRED": "English, Chinese and three code-mixing variants; combined duplicate copy excluded",
    "NNE": "released indices and nested spans; Penn Treebank source words excluded",
    "NLMChem": "NCBI AIONER's PubTator conversion of the NLM-Chem entity layer",
    "TwitterGMNER": "publisher tweet text linked to image IDs; bounding boxes and image bytes excluded",
}

EXCLUSION_REASONS = {
    "DeepFashion2": "publisher requires its application form and archive password for training/validation annotations; GitHub example model predictions and factsheets are not gold labels",
    "GLUECoS": "publisher's Original_Data directories contain split IDs; hydrated examples and external task-specific downloads were not measured",
    "ReasoningNER": "publisher provides held-out dataset configurations; linked NER-CoT instruction training records belong to a different part of the study",
    "BookSumCoref": "BookSum releases book-summary alignments; a matching gold coreference layer was not identified",
    "CEREC": "the registry's Chinese extraction corpus could not be verified; the identically named Enron email coreference corpus is a different task and language",
    "RockNER": "publisher requires accepting terms via its form to obtain OntoRock",
    "SciCoRadar": "repository points to base SciCo data; the separate SciCo-Radar contextual definitions were not measured",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def conllu(text, entity=False):
    tokens = sentences = mentions = tagged = depth = 0
    for line in text.splitlines():
        sentences += line.startswith("# sent_id = ")
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        require(len(fields) == 10, "expected ten CoNLL-U fields")
        if not fields[0].isdigit():  # skip multiword wrappers and empty nodes
            require(re.fullmatch(r"\d+[-.]\d+", fields[0]), "unexpected CoNLL-U ID")
            continue
        tokens += 1
        if entity:
            label = next(
                (a for a in fields[-1].split("|") if a.startswith("Entity=")), ""
            )
            opening, closing = label.count("("), label.count(")")
            tagged += bool(depth or opening)
            mentions += opening
            depth += opening - closing
            require(depth >= 0, "unmatched closing entity bracket")
    require(tokens and sentences and not depth, "empty or incomplete CoNLL-U document")
    result = {"tokens": tokens, "annotated segments": sentences}
    if entity:
        result.update(
            {"documents": 1, "mention annotations": mentions, "tagged tokens": tagged}
        )
    return result


def nne(text):
    # The author releases aligned token indices and POS without the WSJ text.
    # Ends are inclusive. Count the union so a nested span covers a token once.
    lines = text.splitlines()
    totals = Counter(documents=1)
    i = 0
    while i < len(lines):
        indices = lines[i].split()
        if not indices:
            i += 1
            continue
        require(i + 2 < len(lines), "incomplete NNE sentence")
        require(
            indices == [str(n) for n in range(len(indices))],
            "invalid NNE token indices",
        )
        require(len(lines[i + 1].split()) == len(indices), "NNE POS alignment mismatch")
        covered = set()
        for span in filter(None, lines[i + 2].split("|")):
            match = re.fullmatch(r"(\d+),(\d+) (\S+)", span)
            require(match, "invalid NNE annotation")
            start, end = map(int, match.groups()[:2])
            require(start <= end < len(indices), "NNE span outside sentence")
            covered.update(range(start, end + 1))
            totals["mention annotations"] += 1
        totals["tokens"] += len(indices)
        totals["tagged tokens"] += len(covered)
        totals["annotated segments"] += 1
        i += 3
    require(totals["tokens"], "empty NNE document")
    return totals


def token_labels(dataset, text):
    tokens = tagged = mentions = unknown = 0
    for line in text.splitlines():
        if not line.strip() or (
            dataset == "MHERCL" and line.startswith(("#document_", "#sent_text:"))
        ):
            continue
        fields = line.split("\t")
        if dataset == "FinNER":
            # The publication includes -DOCSTART- and page boundary markers.
            if fields[0] in {
                "",
                "-DOCSTART-",
                "<PAGEBREAK>",
                "<HEADLINE>",
                "<INGRESS>",
                "<BODY>",
            }:
                continue
            require(len(fields) == 3, "unexpected FiNER token columns")
            labels = fields[1:]
        else:
            require(len(fields) == 5, "unexpected MHERCL token columns")
            labels = fields[1:2]
        tokens += 1
        if not all(re.fullmatch(r"O|[BI]-.+", tag) for tag in labels):
            require(dataset == "MHERCL", "invalid FiNER BIO tag")
            unknown += 1
            continue
        tagged += any(l != "O" for l in labels)
        mentions += sum(l.startswith("B-") for l in labels)
    if unknown:
        return {
            "tokens": tokens,
            "unrecognized BIO labels": unknown,
            "mention starts with recognized BIO labels": mentions,
        }
    return {"tokens": tokens, "tagged tokens": tagged, "mention annotations": mentions}


def marked_sentences(text):
    totals = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        # Remove markup while preserving the publisher's whitespace tokenization.
        # A token may touch punctuation immediately outside its entity markup.
        plain = []
        ranges = []
        offset = 0
        starts = []
        invalid = False
        for part in re.split(r"(<mark>|</mark>)", line):
            if part == "<mark>":
                starts.append(offset)
            elif part == "</mark>":
                if starts and starts[-1] < offset:
                    ranges.append((starts.pop(), offset))
                else:
                    invalid = True
            else:
                offset += len(part)
                plain.append(part)
        words = list(re.finditer(r"\S+", "".join(plain)))
        totals["tokens"] += len(words)
        totals["annotated segments"] += 1
        if starts or invalid:
            totals["sentences with unbalanced markup"] += 1
            continue
        totals["tagged tokens"] += sum(
            any(w.start() < end and start < w.end() for start, end in ranges)
            for w in words
        )
        totals["mention annotations"] += len(ranges)
    if totals["sentences with unbalanced markup"]:
        # A complete file token denominator cannot be paired with a numerator
        # computed only on the subset of sentences with balanced markup.
        totals["tagged tokens in balanced sentences"] = totals.pop("tagged tokens", 0)
        totals["mentions in balanced sentences"] = totals.pop("mention annotations", 0)
    return totals


def file_metrics(dataset, raw):
    text = raw.decode("utf-8-sig")
    if dataset in {"FinNER", "MHERCL"}:
        return token_labels(dataset, text)
    if dataset in {
        "TaggedPBCEsperanto",
        "TaggedPBCKlingon",
        "Mahanama",
        "CopticScriptorium",
    }:
        return conllu(text, entity=dataset == "Mahanama")
    if dataset == "NNE":
        return nne(text)
    if dataset == "BioNERLLaMA":
        return marked_sentences(text)
    if dataset in {"NLMChem", "DrugProtBioCreative"}:
        ids = {
            line.split("|", 1)[0]
            for line in text.splitlines()
            if re.match(r"^[^|\t]+\|t\|", line)
        }
        spans = [line.split("\t") for line in text.splitlines() if "\t" in line]
        require(ids, "missing PubTator title records")
        return {
            "documents": len(ids),
            "mention annotations": sum(
                len(s) >= 5 and s[0] in ids and s[1].isdigit() and s[2].isdigit()
                for s in spans
            ),
        }
    if dataset == "AnEM":
        words = [
            l.split("\t")
            for l in text.splitlines()
            if l.strip() and not l.startswith(("<ALL>\t", "</ALL>\t"))
        ]
        require(
            all(len(w) == 2 and re.fullmatch(r"O|[BI]-.+", w[1]) for w in words),
            "invalid AnEM BIO columns",
        )
        return {"tokens": len(words), "tagged tokens": sum(w[1] != "O" for w in words)}
    if dataset == "TwitterGMNER":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        require(
            len(lines) % 2 == 0
            and all(line.startswith("IMGID:") for line in lines[::2]),
            "unpaired image ID and tweet",
        )
        return {
            "documents": len(lines) // 2,
            "tokens": sum(len(l.split()) for l in lines[1::2]),
        }
    if dataset == "FantasyCoref":
        docs = mentions = tokens = ends = 0
        for line in text.splitlines():
            if line.startswith("#begin document "):
                docs += 1
            elif line.startswith("#end document"):
                ends += 1
            elif line.strip():
                fields = line.split()
                require(
                    len(fields) >= 12 and fields[2].isdigit(),
                    "invalid gold CoNLL token line",
                )
                tokens += 1
                mentions += fields[-1].count("(")
        require(docs and ends == docs, "unbalanced gold CoNLL document markers")
        return {"documents": docs, "tokens": tokens, "mention annotations": mentions}
    if dataset in {"CodeSearchNet", "CSN"}:
        reader = csv.DictReader(io.StringIO(text))
        require(
            {"Language", "Query", "GitHubUrl", "Relevance"}
            <= set(reader.fieldnames or []),
            "unexpected judgment columns",
        )
        rows = list(reader)
        scored = sum(r["Relevance"].strip() in {"0", "1", "2", "3"} for r in rows)
        return {
            "classification examples": scored,
            "unscored judgment rows": len(rows) - scored,
        }
    if dataset == "MELO":
        rows = csv.reader(io.StringIO(text), delimiter="\t")
        count = 0
        for query, _, concept, relevance in rows:
            require(
                query.startswith("Q")
                and concept.startswith("C")
                and relevance.isdigit(),
                "invalid MELO query/concept judgment",
            )
            count += 1
        return {"relation examples": count}
    if dataset in {"CRAFTCoref", "CRAFTCorpusCoref"}:
        doc = ET.fromstring(raw).find("document")
        require(doc is not None and doc.get("id"), "missing Knowtator document")
        classes = [a.attrib["id"] for a in doc.findall("annotation/class")]
        require(classes, "missing Knowtator coreference annotations")
        return {
            "documents": 1,
            "noun phrase markables": classes.count("Noun Phrase"),
            "identity chain annotations": classes.count("IDENTITY chain"),
        }
    if dataset in {"GeoWebNews", "LGL"}:
        articles = ET.fromstring(raw).findall("article")
        require(
            articles
            and all(
                a.find("text") is not None and a.find("toponyms") is not None
                for a in articles
            ),
            "invalid article or toponym layer",
        )
        return {
            "documents": len(articles),
            "mention annotations": sum(
                len(a.findall("toponyms/toponym")) for a in articles
            ),
        }
    if dataset == "ISNotes":
        root = ET.fromstring(raw)
        marks = root.findall("{*}markable")
        require(
            marks and all(m.get("span") and m.get("id") for m in marks),
            "invalid ISNotes markable",
        )
        return {
            "documents": 1,
            "mention annotations": sum(m.get("mention") == "yes" for m in marks),
        }
    if dataset == "RISeC":
        annotations = [l.split("\t") for l in text.splitlines() if l.strip()]
        require(all(len(a) >= 2 for a in annotations), "invalid Brat line")
        return {
            "documents": 1,
            "mention annotations": sum(a[0].startswith("T") for a in annotations),
            "relations": sum(a[0].startswith("R") for a in annotations),
        }
    if dataset in {"MathDial", "MixRED", "LegalCore"}:
        records = [json.loads(l) for l in text.splitlines() if l.strip()]
        if dataset == "MathDial":
            require(
                all(r["question"] and r["conversation"] for r in records),
                "missing tutor dialogue",
            )
            return {
                "dialogues": len(records),
                "turns": sum(len(r["conversation"].split("|EOM|")) for r in records),
            }
        if dataset == "LegalCore":
            return {
                "documents": len(records),
                "tokens": sum(len(r["tokens"]) for r in records),
                "mention annotations": sum(
                    len(e["mention"]) for r in records for e in r["events"]
                ),
            }
        return {
            "documents": len(records),
            "mention annotations": sum(len(v) for r in records for v in r["vertexSet"]),
            "relations": sum(len(r["labels"]) for r in records),
        }
    if dataset == "ScrollsQMSum":
        try:
            records = [json.loads(text)]
        except json.JSONDecodeError:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        counts = Counter()
        for meeting in records:
            turns = meeting["meeting_transcripts"]
            require(
                turns and all("speaker" in t and "content" in t for t in turns),
                "missing QMSum transcript turn",
            )
            queries = meeting["general_query_list"] + meeting["specific_query_list"]
            require(
                queries and all("query" in q and "answer" in q for q in queries),
                "missing QMSum source query/summary",
            )
            counts.update(
                {"documents": 1, "questions": len(queries), "turns": len(turns)}
            )
        return counts
    data = json.loads(text)
    if dataset == "CoMTA":
        require(
            all(
                d["test_id"] is not None
                and all(t["role"] and "content" in t for t in d["data"])
                for d in data
            ),
            "invalid CoMTA dialogue",
        )
        return {"dialogues": len(data), "turns": sum(len(d["data"]) for d in data)}
    if dataset == "PolyIE":
        counts = Counter()
        for text_parts, label_parts, relation_parts in zip(
            data["text"], data["label"], data["relation"], strict=True
        ):
            counts["documents"] += 1
            for words, labels, relations in zip(
                text_parts, label_parts, relation_parts, strict=True
            ):
                require(
                    len(words) == len(labels)
                    and all(re.fullmatch(r"O|[BI]-.+", tag) for tag in labels),
                    "unaligned PolyIE token or label",
                )
                counts["tokens"] += len(words)
                counts["tagged tokens"] += sum(tag != "O" for tag in labels)
                counts["mention annotations"] += sum(
                    tag.startswith("B-") for tag in labels
                )
                counts["relations"] += len(relations)
        return counts
    if dataset == "BELB":
        counts = Counter()
        for correction in data:
            start, end = correction["start_patch"], correction["end_patch"]
            require(
                correction["id"] and correction["text_patch"],
                "missing BELB source mention",
            )
            if (start, end) == (-1, -1):
                counts["corrections without replacement offsets"] += 1
            else:
                require(0 <= start < end, "invalid corrected mention offsets")
                counts["mention annotations"] += 1
        return counts
    if dataset == "AgriNER":
        require(
            all(d["text"] and isinstance(d["entities"], list) for d in data),
            "missing AgriNER entity layer",
        )
        return {
            "documents": len(data),
            "mention annotations": sum(len(r["entities"]) for r in data),
            "relations": sum(len(r["relations"]) for r in data),
        }
    if dataset == "CASIE":
        require(
            data["sourcefile"] and isinstance(data["cyberevent"]["hopper"], list),
            "invalid CASIE document",
        )
        events = [e for h in data["cyberevent"]["hopper"] for e in h["events"]]
        require(all(e["nugget"] and e["type"] for e in events), "missing event nugget")
        return {
            "documents": 1,
            "event nuggets": len(events),
            "annotated argument pairs": sum(len(e.get("argument", [])) for e in events),
        }
    if dataset == "MAVEN":
        require(
            data
            and all(
                re.fullmatch(r"[a-f0-9]{32}", k) and isinstance(v, str) and v
                for k, v in data.items()
            ),
            "invalid document-topic assignment",
        )
        return {"classification examples": len(data)}
    if dataset == "MultiWOZNER":
        require(
            all(
                d["dialogue_id"]
                and all(t["speaker"] and "utterance" in t for t in d["turns"])
                for d in data
            ),
            "invalid MultiWOZ dialogue",
        )
        return {
            "dialogues": len(data),
            "turns": sum(len(d["turns"]) for d in data),
            "character-offset slot annotations": sum(
                "start" in s and "exclusive_end" in s
                for d in data
                for t in d["turns"]
                for f in t["frames"]
                for s in f["slots"]
            ),
        }
    raise ValueError("unknown publisher format " + dataset)


_DOWNLOAD_LOCKS = defaultdict(Lock)


def cached_download(cache, key, source, expected_size=None):
    dest = cache / key
    with _DOWNLOAD_LOCKS[key]:
        if (
            dest.exists()
            and expected_size is not None
            and dest.stat().st_size != expected_size
        ):
            dest.unlink()
        if not dest.exists():
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                try:
                    with urllib.request.urlopen(source, timeout=180) as res:
                        shutil.copyfileobj(res, tmp)
                    tmp.close()
                    if (
                        expected_size is not None
                        and Path(tmp.name).stat().st_size != expected_size
                    ):
                        raise ValueError("download size differs from pinned tree")
                    Path(tmp.name).replace(dest)
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
    return dest


def archive_metrics(base, tree, cache):
    """Measure a git archive. Each row's digest/bytes identify the whole ZIP.

    Path is the containing directory in that ZIP. Group all matching documents
    under a real directory, verify each file against the pinned tree, and store
    the number of source members actually read alongside the annotation counts.
    """
    dataset, repo, rev = base["id"], base["repository"], base["revision"]
    source = f"https://codeload.github.com/{repo}/zip/{rev}"
    group_counts = defaultdict(Counter)
    try:
        paths = {
            p: size
            for p, size in tree["files"].items()
            if re.fullmatch(ARCHIVE_PATHS[dataset], p)
        }
        require(
            paths and not tree.get("truncated"),
            "complete pinned source-file list unavailable",
        )
        dest = cached_download(cache, rev + ".zip", source)
        with dest.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with zipfile.ZipFile(dest) as archive:
            members = {z.filename.partition("/")[2]: z for z in archive.infolist()}
            for path, size in sorted(paths.items()):
                member = members[path]
                require(
                    member.file_size == size,
                    "archive file size differs from pinned tree",
                )
                counts = file_metrics(dataset, archive.read(member))
                parent = str(Path(path).parent)
                group_counts[parent].update(counts)
                group_counts[parent]["source members"] += 1
        result = []
        for parent, counts in sorted(group_counts.items()):
            name = Path(parent).name
            split = name if dataset in {"Mahanama", "RISeC"} else "all"
            if dataset == "NNE":
                split = (
                    "dev"
                    if name == "02"
                    else "test"
                    if name in {"23", "24"}
                    else "train"
                )
            for unit, value in counts.items():
                result.append(
                    dict(
                        base,
                        status="ok",
                        reason=SCOPES.get(dataset, ""),
                        path=parent + "/",
                        split=split,
                        source_url=source,
                        sha256=digest,
                        bytes=dest.stat().st_size,
                        metric="count",
                        unit=unit,
                        value=value,
                    )
                )
        return result
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as exc:
        return [
            dict(
                base,
                status="unmeasured",
                source_url=source,
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]


# The publisher READMEs at the pinned revisions link these public Drive files.
# A Drive ID is mutable, so a replay also verifies each independent byte digest.
MAVEN_FILES = {
    "train": (
        "12UQpitu4bce_6hwaxgzELN3OcKDX6XfO",
        "232e88b1cdb5ce8d7042bca74586bcb01777d76babc5a5c27e351ea70b7995bb",
    ),
    "valid": (
        "19VMZr3W1giIIJQzvSDP8QrDRXthi45Xx",
        "058e6e12868c0b2908c34ccaa3453e1415a76f517019a1a4f0da52192719b5cc",
    ),
    "test": (
        "14qK80ifhKvyfd93ux-19sh5iM_laFKkW",
        "0dca55bf694b90ba959b27a7a71fbd3ff4892c2ac62f456b252206710aacc928",
    ),
}
MAVEN_ARG_ZIP = (
    "1JmSbAimW4STxp0eqsCfN_JE9MAfCy4hB",
    "d535cee2ea29293e5f20339a851be021328197a1b3148dc771510324f7ec53f1",
)
CONLL2002 = {
    "CoNLL2002": ("esp", "ned"),
    "CoNLL2002Dutch": ("ned",),
    "CoNLL2002Spanish": ("esp",),
}


def maven_counts(lines, dataset):
    counts = Counter()
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        require(d["id"] and "title" in d, "MAVEN document has no identifier or title")
        counts["documents"] += 1
        if dataset == "MAVEN":
            counts["tokens"] += sum(len(s["tokens"]) for s in d["content"])
            if "events" not in d:  # Organizer withholds the event labels on test.
                counts["unlabeled trigger candidates"] += len(d["candidates"])
                continue
        else:
            counts["entity mention annotations"] += sum(
                len(e["mention"]) for e in d["entities"]
            )
            if "events" not in d:
                counts["mention annotations"] += len(d["event_mentions"])
                continue
            counts["annotated argument pairs"] += sum(
                len(args) for e in d["events"] for args in e["argument"].values()
            )
        counts["mention annotations"] += sum(len(e["mention"]) for e in d["events"])
    require(counts["documents"], "empty MAVEN source")
    return counts


def evidence_rows(base, path, split, source, digest, size, counts, scope=""):
    return [
        dict(
            base,
            path=path,
            split=split,
            status="ok",
            reason=scope,
            source_url=source,
            sha256=digest,
            bytes=size,
            metric="count",
            unit=unit,
            value=value,
        )
        for unit, value in counts.items()
    ]


def maven_release(base, cache):
    dataset = base["id"]
    files = MAVEN_FILES if dataset == "MAVEN" else {"archive": MAVEN_ARG_ZIP}
    result = []
    for name, (file_id, expected) in files.items():
        source = (
            "https://drive.usercontent.google.com/download?id="
            + file_id
            + "&export=download&confirm=t"
        )
        try:
            local = cached_download(cache, expected, source)
            with local.open("rb") as f:
                require(
                    hashlib.file_digest(f, "sha256").hexdigest() == expected,
                    "publisher file digest changed",
                )
            size = local.stat().st_size
            if dataset == "MAVEN":
                with local.open("rb") as f:
                    counts = maven_counts(f, dataset)
                scope = (
                    "Organizer releases candidate triggers without test event labels"
                    if name == "test"
                    else ""
                )
                result.extend(
                    evidence_rows(
                        base,
                        name + ".jsonl",
                        name,
                        source,
                        expected,
                        size,
                        counts,
                        scope,
                    )
                )
            else:
                with zipfile.ZipFile(local) as archive:
                    for split in ("train", "valid", "test"):
                        path = split + ".jsonl"
                        with archive.open(path) as f:
                            counts = maven_counts(f, dataset)
                        scope = (
                            "Organizer releases entity/event mentions without test argument-role labels"
                            if split == "test"
                            else ""
                        )
                        result.extend(
                            evidence_rows(
                                base, path, split, source, expected, size, counts, scope
                            )
                        )
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            result.append(
                dict(
                    base,
                    path=name,
                    source_url=source,
                    status="unmeasured",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return result


def conll2002(base, tree, cache):
    """NLTK distributes the task README and all six original language/split files."""
    path = "packages/corpora/conll2002.zip"
    source = f"https://raw.githubusercontent.com/{base['repository']}/{base['revision']}/{path}"
    records = []
    try:
        local = cached_download(cache, base["revision"] + "-conll2002.zip", source)
        size = local.stat().st_size
        require(size == tree["files"][path], "NLTK package differs from pinned tree")
        with local.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        with zipfile.ZipFile(local) as z:
            for language in CONLL2002[base["id"]]:
                for filename, split in (
                    ("train", "train"),
                    ("testa", "dev"),
                    ("testb", "test"),
                ):
                    member = "conll2002/" + language + "." + filename
                    lines = z.read(member).decode("utf-8").splitlines()
                    tokens = tagged = mentions = sentences = 0
                    in_sentence = False
                    for line in lines:
                        words = line.split()
                        if not words or words[0] == "-DOCSTART-":
                            in_sentence = False
                            continue
                        require(
                            len(words) in {2, 3}
                            and re.fullmatch(r"O|[BI]-(?:PER|ORG|LOC|MISC)", words[-1]),
                            "invalid CoNLL-2002 token label",
                        )
                        sentences += not in_sentence
                        in_sentence = True
                        tokens += 1
                        tagged += words[-1] != "O"
                        mentions += words[-1].startswith("B-")
                    require(tokens, "empty CoNLL-2002 member")
                    counts = {
                        "annotated segments": sentences,
                        "tokens": tokens,
                        "tagged tokens": tagged,
                        "mention annotations": mentions,
                    }
                    records.extend(
                        evidence_rows(
                            base,
                            member,
                            language + "/" + split,
                            source,
                            digest,
                            size,
                            counts,
                            "NLTK's shared-task release; language-specific registry IDs are subsets of CoNLL2002",
                        )
                    )
        return records
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
        return [
            dict(
                base,
                source_url=source,
                status="unmeasured",
                reason=f"{type(exc).__name__}: {exc}",
            )
        ]
