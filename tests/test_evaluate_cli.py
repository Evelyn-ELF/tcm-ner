import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ner_utils import score_predictions, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


class EvaluationCliTests(unittest.TestCase):
    def test_micro_aggregates_entity_counts_instead_of_sentence_f1(self):
        gold = {"text": "药", "type": "中药", "start": 0, "end": 1}
        second = {"text": "方", "type": "方剂", "start": 1, "end": 2}
        records = [
            {
                "id": "a",
                "input": "药",
                "output": [gold],
                "raw_output": json.dumps([gold]),
            },
            {"id": "b", "input": "药方", "output": [gold, second], "raw_output": "[]"},
        ]
        result = score_predictions(records)
        self.assertEqual(result["micro"]["f1"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3) / 10)

    def test_rescore_uses_authoritative_gold_and_reports_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gold = {"text": "口苦", "type": "临床表现", "start": 0, "end": 2}
            write_jsonl(root / "data.jsonl", [{"id": "a", "input": "口苦", "output": [gold]}])
            # 预测文件中的真实标签不可信，指定 --data 后应采用原始标签。
            write_jsonl(
                root / "predictions.jsonl",
                [{"id": "a", "input": "口苦", "output": [], "raw_output": "[]"}],
            )
            write_jsonl(root / "dev.jsonl", [{"id": "dev-a", "input": "口苦", "output": [gold]}])
            command = [
                sys.executable,
                str(ROOT / "evaluate.py"),
                "--predictions",
                str(root / "predictions.jsonl"),
                "--data",
                str(root / "data.jsonl"),
                "--dev-data",
                str(root / "dev.jsonl"),
                "--output-dir",
                str(root / "reports"),
            ]
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "reports/test_metrics.json").read_text())
            self.assertEqual(report["micro"]["fn"], 1)
            extra = json.loads((root / "reports/test_without_dev_overlap_metrics.json").read_text())
            self.assertEqual(extra["excluded_overlap_samples"], 1)
            self.assertEqual(extra["sample_count"], 0)
            # 漏掉任何测试 ID 时应失败，不能生成看似更高的指标。
            write_jsonl(root / "predictions.jsonl", [])
            result = subprocess.run(command, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
