import unittest

from corefud_retrieval import score
from predicted_heads import component, intervals, native_counts, span_heads


class PredictedHeadsTests(unittest.TestCase):
    def test_unique_aligned_governor(self):
        text = 'A dog ran'
        tokens = [dict(form=word, char_onset=a, char_offset=b, sentence_ID=0)
                  for word, a, b in [('A', 0, 1), ('dog', 2, 5), ('ran', 6, 9)]]
        query = [(0, 5, 0, 1)]
        self.assertEqual(span_heads(tokens, [1, 2, -1], query, text), {(0, 5): (2, 5)})
        self.assertEqual(span_heads(tokens, [2, 2, -1], query, text), {})
        self.assertEqual(span_heads(tokens, [1, 2, -1], [(1, 5, 0, 1)], text), {})
        tokens[1]['sentence_ID'] = 1
        self.assertEqual(span_heads(tokens, [1, 2, -1], query, text), {})

    def test_retrieval_and_mergers_with_unlabeled_parser_input(self):
        # The parser-side features are fixed while we vary gold identity.
        doc = {'text': 'A fox. Crow. The fox', 'mentions': [
            dict(onset=0, offset=5, COREF='a', start_token=0, end_token=1),
            dict(onset=7, offset=11, COREF='b', start_token=3, end_token=3),
            dict(onset=13, offset=20, COREF='a', start_token=5, end_token=6),
        ]}
        heads = {(0, 5): (2, 5), (7, 11): (7, 11), (13, 20): (17, 20)}
        result = score(doc, heads)
        self.assertEqual((result['gained'], result['lost']), (1, 0))
        self.assertEqual((result['head_backoff.predicted_pairs'], result['head_backoff.correct_pairs']), (1, 1))
        self.assertEqual(result['surface.predicted_pairs'], 0)
        self.assertEqual(result['gold_pairs'], 1)
        # A coreference reference swap changes correctness, never model membership.
        doc['mentions'][-1]['COREF'] = 'b'
        other = score(doc, heads)
        self.assertEqual((other['gained'], other['lost']), (0, 1))
        self.assertEqual((other['head_backoff.predicted_pairs'], other['head_backoff.correct_pairs']), (1, 0))

    def test_native_false_pairs_for_unannotated_output_mentions(self):
        doc = dict(text='Ada he she', mentions=[dict(onset=0, offset=3, COREF='x'),
                                               dict(onset=4, offset=6, COREF='x')],
                   tokens=[dict(form=w, char_onset=a, char_offset=b, sentence_ID=0)
                           for w, a, b in [('Ada', 0, 3), ('he', 4, 6), ('she', 7, 10)]])
        raw = [[(0, 3), (4, 6), (7, 10)]]
        counts = native_counts(doc, [-1, 0, 0], raw)
        self.assertEqual(counts['native_correct_pairs'], 1)
        self.assertEqual(counts['native_predicted_pairs'], 3)
        self.assertEqual(counts['native_exact_mentions'], 2)
        self.assertEqual(counts['native_spans.surface.predicted_pairs'], 0)

    def test_distinct_works_share_a_component(self):
        self.assertEqual(component('nl_openboek-corefud-dev_14'), component('nl_openboek-corefud-dev_93'))
        with self.assertRaises(ValueError):
            component('nl_openboek-corefud-test_14')

    def test_empty_or_zero_query_confidence_interval(self):
        self.assertIsNone(intervals([]))
        no_queries = {'all.queries': 0, 'gained': 0, 'lost': 0}
        self.assertIsNone(intervals([('component', no_queries)]))


if __name__ == '__main__':
    unittest.main()
