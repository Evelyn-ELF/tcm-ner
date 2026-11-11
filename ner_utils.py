"""不依赖深度学习框架的数据、提示模板和完整实体计分逻辑。"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ner_types import Entity, EntityMetrics, MetricsReport, ParsedPrediction, Sample

LABELS = (
    "中药",
    "方剂",
    "临床表现",
    "中医证候",
    "中医诊断",
    "西医诊断",
    "中医治则",
    "中医治疗",
    "西医治疗",
    "其他治疗",
)
INSTRUCTION = (
    "识别文本中的中医药实体，仅输出JSON数组。允许类别："
    + "、".join(LABELS)
    + "。每个实体包含text、type、start、end；位置从0开始，end不包含在实体内。无实体时输出[]。"
)
TEMPLATE_VERSION = "tcm-ner-plain-v1"


def write_json(path: str | Path, value: object) -> None:
    """将可 JSON 序列化的值写为 UTF-8 文件，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, records: Iterable[object]) -> None:
    """逐行写入 JSON 对象，覆盖目标文件；序列化及文件错误向调用方传播。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 对象列表，跳过空行；非法 JSON 或非对象行抛出 ValueError。

    返回值尚未校验业务字段，使用前应调用 validate_samples。
    """
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {number} 行应为对象")
            records.append(record)
    return records


def entity_key(item: Entity) -> tuple[str, int, int]:
    """返回用于严格匹配和去重的类别、起点、终点。"""
    return item["type"], item["start"], item["end"]


def valid_entity(text: str, item: object) -> bool:
    """检查类别、字符边界和原文切片；布尔值不能用作整数位置。"""
    return (
        isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and item["type"] in LABELS
        and type(item.get("start")) is int
        and type(item.get("end")) is int
        and 0 <= item["start"] < item["end"] <= len(text)
        and isinstance(item.get("text"), str)
        and text[item["start"] : item["end"]] == item["text"]
    )


def validate_samples(samples: Sequence[Mapping[str, Any]]) -> None:
    """校验带标注样本的唯一 ID、原文与实体列表；错误时抛出 ValueError。

    接受空集合；调用方根据训练或评估场景另行检查非空约束。
    """
    seen = set()
    for sample in samples:
        sid = sample.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"样本 ID 缺失或重复：{sid!r}")
        seen.add(sid)
        text, entities = sample.get("input"), sample.get("output")
        if not isinstance(text, str) or not isinstance(entities, list):
            raise ValueError(f"{sid} 的 input/output 格式错误")
        if any(not valid_entity(text, item) for item in entities):
            raise ValueError(f"{sid} 的真实实体类别、位置或原文切片错误")
        keys = [entity_key(item) for item in entities]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{sid} 有重复真实实体")


def read_bio(path: str | Path, split: str) -> list[Sample]:
    """将字符级 BIO 文件转换为样本列表，保留空格及原始字符位置。

    split 用于生成稳定的顺序 ID；非法标签或 BIO 衔接抛出 ValueError。
    """
    samples: list[Sample] = []
    chars: list[str] = []
    tags: list[str] = []
    path = Path(path)

    def flush() -> None:
        if not chars:
            return
        text = "".join(chars)
        entities: list[Entity] = []
        start, kind = None, None
        for index, tag in enumerate(tags + ["O"]):
            if start is not None and not tag.startswith("I-"):
                entities.append(
                    {
                        "text": text[start:index],
                        "type": cast(str, kind),
                        "start": start,
                        "end": index,
                    }
                )
                start, kind = None, None
            if tag.startswith("B-"):
                start, kind = index, tag[2:]
        samples.append(
            {
                "id": f"{split}-{len(samples) + 1:06d}",
                "instruction": INSTRUCTION,
                "input": text,
                "output": entities,
            }
        )
        chars.clear()
        tags.clear()

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.rstrip("\r\n")
            if line == "":
                flush()
                continue
            match = re.fullmatch(r"(.)[ \t]+(O|[BI]-.+)", line)
            if not match:
                raise ValueError(f"{path} 第 {line_number} 行格式错误：{line!r}")
            char, tag = match.groups()
            if tag != "O" and tag[2:] not in LABELS:
                raise ValueError(f"{path} 第 {line_number} 行未知标签：{tag}")
            if tag.startswith("I-") and (not tags or tags[-1] not in ("B-" + tag[2:], tag)):
                raise ValueError(f"{path} 第 {line_number} 行非法 BIO 衔接：{tag}")
            chars.append(char)
            tags.append(tag)
    flush()
    validate_samples(samples)
    return samples


def build_prompt(sample: Mapping[str, Any]) -> str:
    """仅使用原文和固定指令构建提示；指令版本不符时抛出 ValueError。"""
    instruction = sample.get("instruction", INSTRUCTION)
    if instruction != INSTRUCTION:
        raise ValueError("样本 instruction 与当前模板不一致，请重新生成数据")
    return f"任务：{instruction}\n文本：{sample['input']}\n答案：\n"


def parse_prediction(text: str, raw: str) -> ParsedPrediction:
    """解析模型输出并校验实体；JSON 错误返回空实体及 parse_error。

    合法实体去重，非法条目按类别累计，用于额外计算假阳性。
    """
    result: ParsedPrediction = {
        "entities": [],
        "parse_error": None,
        "invalid_by_type": {},
        "unknown_type_fp": 0,
        "errors": [],
    }
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("顶层应为 JSON 数组")
    except (ValueError, TypeError, RecursionError) as exc:
        result["parse_error"] = str(exc)
        return result
    seen: set[tuple[str, int, int]] = set()
    invalid: Counter[str] = Counter()
    for index, item in enumerate(items):
        if not valid_entity(text, item):
            kind = item.get("type") if isinstance(item, dict) else None
            if isinstance(kind, str) and kind in LABELS:
                invalid[kind] += 1
            else:
                result["unknown_type_fp"] += 1
            result["errors"].append({"index": index, "reason": "实体类别、位置或原文切片错误"})
            continue
        key = entity_key(cast(Entity, item))
        if key not in seen:
            seen.add(key)
            result["entities"].append(
                cast(Entity, {k: item[k] for k in ("text", "type", "start", "end")})
            )
    result["entities"].sort(key=lambda e: (e["start"], e["end"], e["type"]))
    result["invalid_by_type"] = dict(invalid)
    return result


def _metrics(tp: int, fp: int, fn: int) -> EntityMetrics:
    """根据实体计数计算指标，零分母按零处理。"""
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "support": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_predictions(records: Sequence[Mapping[str, Any]]) -> MetricsReport:
    """对含 output 和 raw_output 的记录计算严格实体指标。

    标签或 ID 错误抛出 ValueError；解析失败按空预测计分，空集合返回零指标。
    """
    validate_samples(records)
    counts: dict[str, Counter[str]] = {kind: Counter() for kind in LABELS}
    failures = unknown = invalid = capped = 0
    for record in records:
        parsed = parse_prediction(record["input"], record["raw_output"])
        failures += parsed["parse_error"] is not None
        unknown += parsed["unknown_type_fp"]
        invalid += sum(parsed["invalid_by_type"].values()) + parsed["unknown_type_fp"]
        capped += bool(record.get("hit_generation_limit", False))
        gold = {entity_key(e) for e in record["output"]}
        pred = {entity_key(e) for e in parsed["entities"]}
        for key in gold & pred:
            counts[key[0]]["tp"] += 1
        for key in pred - gold:
            counts[key[0]]["fp"] += 1
        for key in gold - pred:
            counts[key[0]]["fn"] += 1
        for kind, count in parsed["invalid_by_type"].items():
            counts[kind]["fp"] += count
    per_type = {kind: _metrics(c["tp"], c["fp"], c["fn"]) for kind, c in counts.items()}
    tp = sum(c["tp"] for c in counts.values())
    fp = sum(c["fp"] for c in counts.values()) + unknown
    fn = sum(c["fn"] for c in counts.values())
    n = len(records)
    return {
        "sample_count": n,
        "per_type": per_type,
        "micro": _metrics(tp, fp, fn),
        "macro_f1": sum(row["f1"] for row in per_type.values()) / len(LABELS),
        "parse_failure_rate": failures / n if n else 0.0,
        "parse_failure_count": failures,
        "invalid_item_count": invalid,
        "unknown_type_fp": unknown,
        "generation_limit_rate": capped / n if n else 0.0,
        "scoring_rule": "完整实体严格匹配；非法数组条目额外计FP，未知类别纳入总体FP",
    }


def flatten_metrics(report: MetricsReport, prefix: str = "eval") -> dict[str, float | int]:
    """将嵌套计分报告转换为 Trainer 和 TensorBoard 使用的标量指标。"""
    values = {
        f"{prefix}_entity_micro_f1": report["micro"]["f1"],
        f"{prefix}_entity_macro_f1": report["macro_f1"],
    }
    for name in (
        "parse_failure_rate",
        "invalid_item_count",
        "unknown_type_fp",
        "generation_limit_rate",
    ):
        values[f"{prefix}_{name}"] = cast(Any, report)[name]
    for kind, row in report["per_type"].items():
        for name in ("precision", "recall", "f1", "support"):
            values[f"{prefix}_entity_{name}/{kind}"] = cast(Any, row)[name]
    return values


def save_report(report: MetricsReport, output_dir: str | Path, prefix: str = "test") -> None:
    """保存 JSON 完整报告和逐类 CSV，覆盖同前缀文件。"""
    output_dir = Path(output_dir)
    write_json(output_dir / f"{prefix}_metrics.json", report)
    with (output_dir / f"{prefix}_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "类别",
                "precision",
                "recall",
                "f1",
                "support",
                "tp",
                "fp",
                "fn",
            ],
        )
        writer.writeheader()
        for kind, row in report["per_type"].items():
            writer.writerow({"类别": kind, **row})
        writer.writerow({"类别": "Micro", **report["micro"]})


def print_report(report: MetricsReport) -> None:
    """打印全部类别、Micro/Macro 指标与生成异常统计。"""
    print("类别\tPrecision\tRecall\tF1\tsupport\tTP\tFP\tFN")
    for kind, row in [*report["per_type"].items(), ("Micro", report["micro"])]:
        print(
            f"{kind}\t{row['precision']:.4f}\t{row['recall']:.4f}\t{row['f1']:.4f}"
            f"\t{row['support']}\t{row['tp']}\t{row['fp']}\t{row['fn']}"
        )
    print(
        f"Macro-F1：{report['macro_f1']:.4f}；JSON解析失败率：{report['parse_failure_rate']:.4f}；"
        f"非法条目：{report['invalid_item_count']}；未知类别FP：{report['unknown_type_fp']}；"
        f"生成上限命中率：{report['generation_limit_rate']:.4f}"
    )
