import unittest

from src.absa.data import Example, SpanLabel
from src.absa.metrics import evaluate_exact
from src.absa.tags import bio_to_spans, spans_to_bio


class AbsaCoreTests(unittest.TestCase):
    def test_bio_roundtrip_for_simple_span(self):
        ex = Example("Pin khỏe nhưng camera tệ", [SpanLabel(0, 8, "BATTERY#POSITIVE")])

        tags, token_spans = spans_to_bio(ex)
        spans = bio_to_spans(tags, token_spans)

        self.assertEqual(tags[:2], ["B-BATTERY#POSITIVE", "I-BATTERY#POSITIVE"])
        self.assertEqual(spans, [SpanLabel(0, 8, "BATTERY#POSITIVE")])

    def test_exact_metric(self):
        gold = [Example("abc", [SpanLabel(0, 3, "GENERAL#POSITIVE")])]
        pred = [Example("abc", [SpanLabel(0, 3, "GENERAL#POSITIVE")])]

        result = evaluate_exact(gold, pred)

        self.assertEqual(result["micro"]["tp"], 1)
        self.assertEqual(result["micro"]["fp"], 0)
        self.assertEqual(result["micro"]["fn"], 0)
        self.assertEqual(result["micro"]["f1"], 1.0)

    def test_metric_clamps_gold_offsets(self):
        gold = [Example("abc", [SpanLabel(0, 5, "GENERAL#POSITIVE")])]
        pred = [Example("abc", [SpanLabel(0, 3, "GENERAL#POSITIVE")])]

        result = evaluate_exact(gold, pred)

        self.assertEqual(result["micro"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
