"""Offline regressions for registry path suggestions from operator caches."""
import importlib.util
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATE = load("enrich_dataset_registry")
APPLY = load("apply_registry_enrichment")


def propose(names, filenames):
    datasets = {name: {"name": title, "s3_path": None} for name, title in names}
    files = {filename: {"path": f"datasets/{filename}"} for filename in filenames}
    return GENERATE.match_cache_to_registry(datasets, files)


class CacheEnrichmentTests(unittest.TestCase):
    def test_short_filename_does_not_claim_unrelated_datasets(self):
        names = [("AutomotiveNER", "Automotive NER"), ("BasqueNER", "Basque NER")]
        self.assertEqual(propose(names, ["e_ner.csv"]), {})
        old = {"BasqueNER": {"s3_path": "datasets/e_ner.csv", "match_type": "substring_name"}}
        source = '    BasqueNER { categories: ["ner"] }'
        self.assertEqual(APPLY.update_registry(source, old, {}), (source, 0))

    def test_only_an_unambiguous_complete_basename_is_a_candidate(self):
        one = [("NorthEast", "North East")]
        match = propose(one, ["north_east.csv"])
        self.assertEqual(match["NorthEast"]["filename"], "north_east.csv")
        self.assertFalse(propose(one, ["north_east.csv", "north-east.json"]))
        self.assertFalse(propose(one + [("Another", "North-East")], ["north_east.csv"]))
