import json
import tempfile
import unittest
from pathlib import Path

from ner_utils import LABELS, parse_prediction, read_bio, score_predictions
from prepare_data import prepare_data


def entity(text="口苦", kind="临床表现", start=0, end=2):
    return {"text": text, "type": kind, "start": start, "end": end}


class BioTests(unittest.TestCase):
    def test_preserves_space_and_flushes_final_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bio"
            path.write_text("　 O\n口 B-临床表现\n苦 I-临床表现", encoding="utf-8")
            rows = read_bio(path, "train")
        self.assertEqual(rows[0]["input"], "　口苦")
        self.assertEqual(rows[0]["output"], [entity(start=1, end=3)])

    def test_rejects_illegal_bio_with_line_number(self):
        for content in [
            "口 I-临床表现\n",
            "口 B-中药\n苦 I-临床表现\n",
            "口 B-不存在\n",
            "口苦 O\n",
        ]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bio"
                path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "行"):
                    read_bio(path, "train")

    def test_sentence_boundary_and_empty_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bio"
            path.write_text("口 B-临床表现\n\n\n好 O\n", encoding="utf-8")
            rows = read_bio(path, "test")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["id"], rows[1]["id"])
        self.assertEqual(rows[1]["output"], [])

    def test_prepare_quarantines_conflicts_and_excludes_eval_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 甲重复；乙标注冲突；丙与验证重合；丁与测试重合。
            (root / "medical.train").write_text(
                "甲 O\n\n甲 O\n\n乙 O\n\n乙 B-中药\n\n丙 O\n\n丁 O\n", encoding="utf-8"
            )
            (root / "medical.dev").write_text("丙 O\n\n戊 O\n", encoding="utf-8")
            (root / "medical.test").write_text("丁 O\n\n戊 O\n", encoding="utf-8")
            audit = prepare_data(root, root / "processed")
            train = [
                json.loads(x) for x in (root / "processed/train.jsonl").read_text().splitlines()
            ]
            self.assertEqual([x["input"] for x in train], ["甲"])
            self.assertEqual(audit["processed"]["train"]["samples"], 1)
            self.assertEqual(len(audit["excluded_train"]), 5)
            self.assertEqual(audit["dev_test_overlap_texts"], ["戊"])
            self.assertEqual(audit["processed"]["dev"]["samples"], 2)


class MetricsTests(unittest.TestCase):
    def record(self, text, gold, raw, sample_id="test-1"):
        return {"id": sample_id, "input": text, "output": gold, "raw_output": raw}

    def test_exact_span_not_character_credit(self):
        gold = [entity("慢性胃炎", "西医诊断", 0, 4)]
        raw = json.dumps([entity("胃炎", "西医诊断", 2, 4)], ensure_ascii=False)
        report = score_predictions([self.record("慢性胃炎", gold, raw)])
        row = report["per_type"]["西医诊断"]
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["f1"]), (0, 1, 1, 0))

    def test_repeated_words_have_distinct_positions_and_deduplicate_predictions(self):
        a, b = entity(), entity(start=3, end=5)
        report = score_predictions([self.record("口苦，口苦", [a, b], json.dumps([a, a]))])
        row = report["per_type"]["临床表现"]
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 0, 1))
        self.assertAlmostEqual(report["micro"]["f1"], 2 / 3)

    def test_wrong_type_is_false_positive_and_false_negative(self):
        report = score_predictions(
            [self.record("口苦", [entity()], json.dumps([entity(kind="中药")]))]
        )
        self.assertEqual(report["per_type"]["中药"]["fp"], 1)
        self.assertEqual(report["per_type"]["临床表现"]["fn"], 1)

    def test_invalid_json_is_not_dropped(self):
        report = score_predictions([self.record("口苦", [entity()], "```json\n[]\n```")])
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["micro"]["fn"], 1)
        self.assertEqual(report["parse_failure_rate"], 1)

    def test_invalid_items_count_as_fp_including_unknown_type(self):
        items = [entity(), entity(text="发热"), entity(kind="未知"), "坏条目"]
        report = score_predictions([self.record("口苦", [entity()], json.dumps(items))])
        self.assertEqual(report["micro"]["tp"], 1)
        self.assertEqual(report["micro"]["fp"], 3)
        self.assertEqual(report["unknown_type_fp"], 2)
        self.assertEqual(report["per_type"]["临床表现"]["fp"], 1)
        self.assertAlmostEqual(report["micro"]["f1"], 0.4)

    def test_empty_sets_and_all_ten_labels_are_reported(self):
        report = score_predictions([self.record("好", [], "[]")])
        self.assertEqual(set(report["per_type"]), set(LABELS))
        self.assertEqual(report["micro"]["f1"], 0)
        self.assertEqual(report["macro_f1"], 0)

    def test_parser_rejects_non_arrays_and_inconsistent_text(self):
        self.assertTrue(parse_prediction("口苦", "{}")["parse_error"])
        result = parse_prediction("口苦", json.dumps([entity(text="发热")]))
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["invalid_by_type"]["临床表现"], 1)

    def test_parser_repairs_unique_offset_and_deduplicates(self):
        raw = json.dumps([entity(start=1, end=3), entity(start=2, end=4)])
        result = parse_prediction("患者口苦", raw)
        self.assertEqual(result["entities"], [entity(start=2, end=4)])
        self.assertEqual(result["errors"], [])

    def test_parser_computes_first_match_with_or_without_model_offsets(self):
        raw = json.dumps([{"text": "口苦", "type": "临床表现"}, entity(start=3, end=5)])
        result = parse_prediction("口苦、口苦", raw)
        self.assertEqual(result["entities"], [entity(start=0, end=2)])
        self.assertEqual(result["invalid_by_type"], {})

    def test_duplicate_sample_ids_cannot_hide_false_negatives(self):
        record = self.record("口苦", [entity()], "[]")
        with self.assertRaisesRegex(ValueError, "ID"):
            score_predictions([record, record])


if __name__ == "__main__":
    unittest.main()
