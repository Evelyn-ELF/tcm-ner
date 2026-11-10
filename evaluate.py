"""完整测试集生成及实体评估，也可对保存的预测离线重新计分。"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ner_types import Sample
from ner_utils import (
    flatten_metrics,
    print_report,
    read_jsonl,
    save_report,
    score_predictions,
    validate_samples,
    write_jsonl,
)


def main() -> None:
    """解析命令行参数并执行对应任务，错误向命令行调用方传播。"""
    parser = argparse.ArgumentParser(description="打印全部实体类别 F1 并保存 JSON/CSV 报告")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", help="train.py 产生的 best_adapter 目录")
    source.add_argument("--predictions", help="已有预测 JSONL，仅重新计分，无需模型依赖")
    parser.add_argument("--data", help="SFT 测试 JSONL，生成时默认为 data/processed/test.jsonl")
    parser.add_argument("--dev-data", help="可选的完整验证 JSONL，用于补充排除重合文本的报告")
    parser.add_argument("--output-dir", required=True, help="报告目录")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="模型运行设备")
    parser.add_argument("--wandb-run-id", help="可选：将最终测试聚合指标写回原训练 run")
    parser.add_argument("--wandb-project", default="tcm-ner")
    parser.add_argument("--wandb-entity", help="原训练 run 所属团队")
    args = parser.parse_args()
    records: Sequence[Mapping[str, Any]]
    overlap_texts = None
    if args.predictions:
        records = read_jsonl(args.predictions)
        if args.data:
            samples = read_jsonl(args.data)
            validate_samples(samples)
            validate_samples(records)
            by_id = {r["id"]: r for r in records}
            if set(by_id) != {s["id"] for s in samples}:
                raise ValueError("预测文件与测试文件 ID 集合不一致，不能漏评或多评样本")
            records = [
                {
                    **s,
                    "raw_output": by_id[s["id"]]["raw_output"],
                    "hit_generation_limit": by_id[s["id"]].get("hit_generation_limit", False),
                }
                for s in samples
            ]
    else:
        from model_runtime import generate_records, load_adapter

        samples = read_jsonl(args.data or "data/processed/test.jsonl")
        validate_samples(samples)
        if not samples:
            raise ValueError("测试集不能为空")
        model, tokenizer, metadata = load_adapter(args.adapter, args.device)
        records = generate_records(
            model,
            tokenizer,
            cast(list[Sample], samples),
            metadata["max_length"],
            metadata["generation"]["max_new_tokens"],
            progress=True,
        )
        overlap_file = Path(args.adapter) / "validation_texts.json"
        if overlap_file.exists():
            overlap_texts = set(json.loads(overlap_file.read_text(encoding="utf-8")))
    if args.dev_data:
        dev = read_jsonl(args.dev_data)
        validate_samples(dev)
        overlap_texts = {s["input"] for s in dev}
    if not records:
        raise ValueError("待评估预测不能为空")
    report = score_predictions(records)
    write_jsonl(Path(args.output_dir) / "test_predictions.jsonl", records)
    save_report(report, args.output_dir)
    print_report(report)
    if overlap_texts is not None:
        filtered = [r for r in records if r["input"] not in overlap_texts]
        extra = score_predictions(filtered)
        extra["excluded_overlap_samples"] = len(records) - len(filtered)
        save_report(extra, args.output_dir, "test_without_dev_overlap")
        print(
            f"排除验证原文重合 {len(records) - len(filtered)} 条后的 Micro-F1：{extra['micro']['f1']:.4f}"
        )
    if args.wandb_run_id:
        import wandb

        # 本地报告已落盘；恢复原训练 run，仅提交聚合指标，不上传预测原文。
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id=args.wandb_run_id,
            resume="must",
            job_type="test",
        ) as run:
            run.summary.update(
                {
                    k.replace("test_", "test/", 1): v
                    for k, v in flatten_metrics(report, "test").items()
                }
            )


if __name__ == "__main__":
    main()
