"""保留原始字符，将 BIO 标注转换为 SFT JSONL 并审计训练数据。"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ner_types import Sample
from ner_utils import LABELS, TEMPLATE_VERSION, read_bio, validate_samples, write_json, write_jsonl


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


def _apply_revisions(data_dir: Path, processed: dict[str, list[Sample]]) -> list[str]:
    revisions_path = data_dir / "annotation_revisions.json"
    revisions = (
        json.loads(revisions_path.read_text(encoding="utf-8")) if revisions_path.exists() else {}
    )
    applied = []
    for split, samples in processed.items():
        revised_samples = []
        for sample in samples:
            revision = revisions.get(sample["id"])
            if revision is not None:
                if sample["input"] != revision["input"] or sample["output"] != revision["output"]:
                    raise ValueError(f"修订与原始样本不一致：{sample['id']}")
                sample = {
                    **sample,
                    "input": revision["revised_input"],
                    "output": revision["revised_output"],
                }
                applied.append(sample["id"])
            revised_samples.append(sample)
        validate_samples(revised_samples)
        processed[split] = revised_samples
    if set(revisions) != set(applied):
        raise ValueError("修订清单包含未应用的样本编号")
    return applied


def _apply_exclusions(data_dir: Path, processed: dict[str, list[Sample]]) -> list[dict[str, Any]]:
    exclusions_path = data_dir / "split_exclusions.json"
    manual_exclusions = (
        json.loads(exclusions_path.read_text(encoding="utf-8")) if exclusions_path.exists() else {}
    )
    excluded_after_review = []
    for split, samples in processed.items():
        retained = []
        for sample in samples:
            reason = manual_exclusions.get(sample["id"])
            if reason is None:
                retained.append(sample)
            else:
                excluded_after_review.append({**sample, "reason": reason})
        if not retained:
            raise ValueError(f"应用样本排除清单后集合为空：{split}")
        processed[split] = retained
    if set(manual_exclusions) != {sample["id"] for sample in excluded_after_review}:
        raise ValueError("样本排除清单包含未匹配的样本编号")
    return excluded_after_review


def _deduplicate(
    raw: dict[str, list[Sample]],
) -> tuple[dict[str, list[Sample]], list[dict[str, Any]], list[dict[str, Any]]]:
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
    dev_texts = {sample["input"] for sample in raw["dev"]}
    test: list[Sample] = []
    excluded_test: list[dict[str, Any]] = []
    for sample in raw["test"]:
        if sample["input"] in dev_texts:
            excluded_test.append({**sample, "reason": "与验证集原文重合"})
        else:
            test.append(sample)
    if not test:
        raise ValueError("排除与验证集重合的样本后，测试集为空")
    processed = {**raw, "train": kept, "test": test}
    return processed, excluded, excluded_test


def prepare_data(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """读取三个 BIO 集合并应用去重、标注修订和样本排除规则。

    集合为空、修订未匹配或清洗后仍有跨集原文重合时抛出 ValueError。
    输出目录自动创建，同名输出文件会被覆盖。
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    paths = {split: data_dir / f"medical.{split}" for split in ("train", "dev", "test")}
    raw = {split: read_bio(path, split) for split, path in paths.items()}
    if any(not samples for samples in raw.values()):
        raise ValueError("训练、验证及测试集均不能为空")
    processed, excluded, excluded_test = _deduplicate(raw)
    applied = _apply_revisions(data_dir, processed)
    excluded_after_review = _apply_exclusions(data_dir, processed)
    texts = {split: {s["input"] for s in samples} for split, samples in processed.items()}
    if texts["train"] & (texts["dev"] | texts["test"]) or texts["dev"] & texts["test"]:
        raise ValueError("清洗后出现跨集合原文重合")
    conflicts = {}
    for split, samples in raw.items():
        by_text: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            by_text[sample["input"]].append(sample)
        conflicts[split] = [
            group for group in by_text.values() if len({_annotation(s) for s in group}) > 1
        ]
    audit = {
        "rules_version": "tcm-ner-clean-v4",
        "template_version": TEMPLATE_VERSION,
        "source_sha256": {
            split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in paths.items()
        },
        "raw": {split: _stats(samples) for split, samples in raw.items()},
        "processed": {split: _stats(samples) for split, samples in processed.items()},
        "applied_annotation_revisions": applied,
        "excluded_train": excluded,
        "excluded_after_review": excluded_after_review,
        "excluded_test": excluded_test,
        "annotation_conflicts": conflicts,
        "dev_test_overlap_texts": sorted({sample["input"] for sample in excluded_test}),
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
