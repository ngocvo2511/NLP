import unittest

from src.absa.data import Example, SpanLabel
from src.absa.metrics import evaluate_exact
from src.absa.postprocess import postprocess_spans
from src.absa.tags import bio_to_spans, spans_to_bio, spans_to_tags, tags_to_spans
from src.absa.train_transformer import ids_to_spans


class AbsaCoreTests(unittest.TestCase):
    def test_bio_roundtrip_for_simple_span(self):
        ex = Example("Pin khỏe nhưng camera tệ", [SpanLabel(0, 8, "BATTERY#POSITIVE")])

        tags, token_spans = spans_to_bio(ex)
        spans = bio_to_spans(tags, token_spans)

        self.assertEqual(tags[:2], ["B-BATTERY#POSITIVE", "I-BATTERY#POSITIVE"])
        self.assertEqual(spans, [SpanLabel(0, 8, "BATTERY#POSITIVE")])

    def test_bilou_roundtrip_for_single_and_multi_token_spans(self):
        ex = Example(
            "Pin khỏe camera tệ",
            [
                SpanLabel(0, 8, "BATTERY#POSITIVE"),
                SpanLabel(9, 15, "CAMERA#NEGATIVE"),
            ],
        )

        tags, token_spans = spans_to_tags(ex, scheme="bilou")
        spans = tags_to_spans(tags, token_spans)

        self.assertEqual(tags[:3], ["B-BATTERY#POSITIVE", "L-BATTERY#POSITIVE", "U-CAMERA#NEGATIVE"])
        self.assertEqual(spans, ex.labels)

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

    def test_postprocess_merges_nearby_same_label_and_filters_short_spans(self):
        spans = [
            SpanLabel(0, 3, "BATTERY#POSITIVE"),
            SpanLabel(5, 10, "BATTERY#POSITIVE"),
            SpanLabel(12, 13, "CAMERA#NEGATIVE"),
        ]

        result = postprocess_spans(spans, "pin x trau y c", min_chars=4, merge_gap=2)

        self.assertEqual(result, [SpanLabel(0, 10, "BATTERY#POSITIVE")])

    def test_ids_to_spans_supports_bilou_tags(self):
        id2label = {
            0: "O",
            1: "B-BATTERY#POSITIVE",
            2: "L-BATTERY#POSITIVE",
            3: "U-CAMERA#NEGATIVE",
        }
        pred_ids = [1, 2, 0, 3]
        offsets = [(0, 3), (4, 8), (9, 12), (13, 19)]

        result = ids_to_spans(pred_ids, offsets, id2label)

        self.assertEqual(result, {(0, 8, "BATTERY#POSITIVE"), (13, 19, "CAMERA#NEGATIVE")})


if __name__ == "__main__":
    unittest.main()
