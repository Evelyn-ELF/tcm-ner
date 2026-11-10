"""保留原始字符，将 BIO 标注转换为 SFT JSONL 并审计训练数据。"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ner_types import Sample
from ner_utils import LABELS, TEMPLATE_VERSION, read_bio, write_json, write_jsonl


def _stats(samples: Sequence[Sample]) -> dict[str, Any]:
    entities = Counter(e["type"] for s in samples for e in s["output"])
    return {
        "samples": len(samples),
        "unique_texts": len({s["input"] for s in samples}),
        "empty_samples": sum(not s["output"] for s in samples),
        "max_characters": max((len(s["input"]) for s in samples), default=0),
        "entities": {kind: entities[kind] for kind in LABELS},
    }


def _annotation(sample: Sample) -> str:
    return json.dumps(sample["output"], ensure_ascii=False, sort_keys=True)


def prepare_data(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """读取三个 BIO 集合、清洗训练集并写出 JSONL 和审计报告。

    保留评估集原始标注；任一集合为空或清洗后训练集为空时抛出 ValueError。
    输出目录自动创建，同名输出文件会被覆盖。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    paths = {split: data_dir / f"medical.{split}" for split in ("train", "dev", "test")}
    raw = {split: read_bio(path, split) for split, path in paths.items()}
    if any(not samples for samples in raw.values()):
        raise ValueError("训练、验证及测试集均不能为空")
    eval_texts = {s["input"] for split in ("dev", "test") for s in raw[split]}
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in raw["train"]:
        groups[sample["input"]].append(sample)
    kept, excluded = [], []
    for text, group in groups.items():
        conflict = len({_annotation(s) for s in group}) > 1
        for i, sample in enumerate(group):
            reason = (
                "与评估集原文重合"
                if text in eval_texts
                else "同文标注冲突"
                if conflict
                else "重复文本及标注"
                if i
                else None
            )
            if reason:
                excluded.append({**sample, "reason": reason})
            else:
                kept.append(sample)
    order = {s["id"]: i for i, s in enumerate(raw["train"])}
    kept.sort(key=lambda s: order[s["id"]])
    if not kept:
        raise ValueError("清洗后训练集为空，请检查数据审计规则")
    processed = {**raw, "train": kept}
    conflicts = {}
    for split, samples in raw.items():
        by_text: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            by_text[sample["input"]].append(sample)
        conflicts[split] = [
            group for group in by_text.values() if len({_annotation(s) for s in group}) > 1
        ]
    overlap = sorted({s["input"] for s in raw["dev"]} & {s["input"] for s in raw["test"]})
    audit = {
        "rules_version": "tcm-ner-clean-v1",
        "template_version": TEMPLATE_VERSION,
        "source_sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in paths.items()
        },
        "raw": {split: _stats(samples) for split, samples in raw.items()},
        "processed": {split: _stats(samples) for split, samples in processed.items()},
        "excluded_train": excluded,
        "annotation_conflicts": conflicts,
        "dev_test_overlap_texts": overlap,
    }
    for split, samples in processed.items():
        write_jsonl(output_dir / f"{split}.jsonl", samples)
    write_json(output_dir / "labels.json", list(LABELS))
    write_json(output_dir / "data_audit.json", audit)
    return audit


def main() -> None:
    """解析命令行参数并执行对应任务，错误向命令行调用方传播。"""
    parser = argparse.ArgumentParser(description="转换中医药 BIO 数据并检查重复、冲突与泄漏")
    parser.add_argument("--data-dir", default="data", help="原始 BIO 文件目录")
    parser.add_argument("--output-dir", default="data/processed", help="处理结果目录")
    args = parser.parse_args()
    audit = prepare_data(args.data_dir, args.output_dir)
    print(json.dumps(audit["processed"], ensure_ascii=False, indent=2))
    print(
        f"训练排除 {len(audit['excluded_train'])} 条；完整审计：{args.output_dir}/data_audit.json"
    )


if __name__ == "__main__":
    main()
