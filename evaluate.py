"""完整测试集生成及实体评估，也可对保存的预测离线重新计分。"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from ner_types import MetricsReport, Prediction, Sample
from ner_utils import (
    flatten_metrics,
    parse_prediction,
    print_report,
    read_jsonl,
    save_report,
    score_predictions,
    validate_samples,
    write_json,
    write_jsonl,
)


def write_final_report(path: Path, report: MetricsReport, test_data: str, source: str) -> None:
    """生成可阅读的测试集报告。"""
    lines = [
        "# 命名实体识别测试报告",
        "",
        f"- 测试数据：`{test_data}`",
        f"- 预测来源：`{source}`",
        f"- 纳入测试样本：{report['sample_count']} 条",
        "- 定位规则：按实体文本在原文中的首次精确匹配计算字符位置。",
        f"- 计分规则：{report['scoring_rule']}",
        f"- Macro-F1：{report['macro_f1']:.2%}",
        f"- JSON 解析失败：{report['parse_failure_count']} 条",
        f"- 非法实体条目：{report['invalid_item_count']} 个",
        f"- 生成达到长度上限的比例：{report['generation_limit_rate']:.2%}",
        "",
        "| 类别 | 精确率 | 召回率 | F1 | 标注实体数 | 命中 | 误报 | 漏报 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [*report["per_type"].items(), ("总体（Micro）", report["micro"])]:
        lines.append(
            f"| {name} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} "
            f"| {row['support']} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


BOUNDARY_GUIDANCE = (
    "边界规则：实体须连续且逐字复制原文。复合治则按能独立表达的施治原则判断边界。"
    "方剂保留属于名称的修饰语，排除用量、给药动作和连接标点。"
    "药品剂型属于原文完整药名时保留。按上下文区分诊断、症状和治疗。"
)
WEAK_TYPES = {"中医治则", "方剂"}
RECHECK_GUIDANCE = (
    "本轮专项检查仅输出中医治则和方剂，其他类别不输出。"
    "留意施治原则、处方名、方剂简称和列药前的方名。"
)


def merge_recheck(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """把补查中通过原文定位校验的目标实体加入首轮结果。"""
    text = first["input"]
    added = [entity for entity in second["predicted_entities"] if entity["type"] in WEAK_TYPES]
    if not added:
        return dict(first)
    try:
        items = json.loads(first["raw_output"])
    except ValueError:
        items = []
    if not isinstance(items, list):
        items = []
    raw = json.dumps([*items, *added], ensure_ascii=False)
    parsed = parse_prediction(text, raw)
    return {
        **first,
        "raw_output": raw,
        "predicted_entities": parsed["entities"],
        "parse_error": parsed["parse_error"],
        "item_errors": parsed["errors"],
        "hit_generation_limit": first["hit_generation_limit"] or second["hit_generation_limit"],
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数并执行对应任务，错误向命令行调用方传播。"""
    parser = argparse.ArgumentParser(description="打印全部实体类别 F1 并保存 JSON/CSV 报告")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter", help="train.py 产生的 best_adapter 目录")
    source.add_argument("--predictions", help="已有预测 JSONL，仅重新计分，无需模型依赖")
    parser.add_argument("--data", help="SFT 测试 JSONL，生成时默认为 data/processed/test.jsonl")
    parser.add_argument("--output-dir", required=True, help="报告目录")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="模型运行设备")
    parser.add_argument(
        "--tensorboard-log-dir", type=Path, help="可选：写入测试集聚合指标的 TensorBoard 目录"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="测试集生成批大小")
    parser.add_argument("--final-report", action="store_true", help="生成中文测试集报告")
    parser.add_argument("--validation-experiment", choices=["baseline", "boundary", "recheck"])
    args = parser.parse_args()
    if args.validation_experiment and (not args.adapter or not args.data or args.final_report):
        parser.error("验证实验需要 --adapter 和 --data，并应单独运行")
    if args.final_report and args.predictions and not args.data:
        parser.error("离线生成最终报告需要 --data 指定测试集")
    return args


def main() -> None:
    """运行验证实验或测试集评估。"""
    args = parse_args()
    records: Sequence[Mapping[str, Any]]
    if args.predictions:
        records = read_jsonl(args.predictions)
        if args.data:
            samples = read_jsonl(args.data)
            validate_samples(samples)
            validate_samples(records)
            by_id = {r["id"]: r for r in records}
            if set(by_id) != {s["id"] for s in samples}:
                raise ValueError("预测文件与测试文件 ID 集合不一致，不能漏评或多评样本")
            for sample in samples:
                if by_id[sample["id"]]["input"] != sample["input"]:
                    raise ValueError(f"预测文件与测试文件的原文不一致：{sample['id']}")
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

        def generate(guidance: str) -> list[Prediction]:
            return generate_records(
                model,
                tokenizer,
                cast(list[Sample], samples),
                metadata["max_length"],
                metadata["generation"]["max_new_tokens"],
                progress=True,
                batch_size=args.batch_size,
                instruction=metadata["instruction"],
                guidance=guidance,
            )

        start = perf_counter()
        records = generate(BOUNDARY_GUIDANCE if args.validation_experiment == "boundary" else "")
        first_seconds = perf_counter() - start
        if args.validation_experiment:
            output_dir = Path(args.output_dir)
            if args.validation_experiment == "recheck":
                start = perf_counter()
                second = generate(RECHECK_GUIDANCE)
                second_seconds = perf_counter() - start
                write_jsonl(output_dir / "recheck_predictions.jsonl", second)
                write_jsonl(output_dir / "first_pass_predictions.jsonl", records)
                records = [merge_recheck(x, y) for x, y in zip(records, second, strict=True)]
            else:
                second_seconds = 0.0
            report = score_predictions(records)
            write_jsonl(output_dir / "validation_predictions.jsonl", records)
            save_report(report, output_dir, "validation")
            write_json(
                output_dir / "experiment.json",
                {
                    "方案": args.validation_experiment,
                    "数据": args.data,
                    "适配器": args.adapter,
                    "首轮推理秒数": first_seconds,
                    "补查推理秒数": second_seconds,
                },
            )
            print_report(report)
            return
    if not records:
        raise ValueError("待评估预测不能为空")
    report = score_predictions(records)
    write_jsonl(Path(args.output_dir) / "test_predictions.jsonl", records)
    save_report(report, args.output_dir)
    print_report(report)
    if args.final_report:
        write_final_report(
            Path(args.output_dir) / "测试集报告.md",
            report,
            args.data or "data/processed/test.jsonl",
            args.adapter or args.predictions,
        )
    if args.tensorboard_log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        with SummaryWriter(log_dir=str(args.tensorboard_log_dir)) as writer:
            for key, value in flatten_metrics(report, "test").items():
                writer.add_scalar(key.replace("test_", "test/", 1), value, global_step=0)


if __name__ == "__main__":
    main()
