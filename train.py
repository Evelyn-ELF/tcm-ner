"""中医药实体抽取的单卡 QLoRA/LoRA SFT 训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ner_types import EncodedSample, Sample

if TYPE_CHECKING:
    import torch

    from ner_trainer import EntityTrainer

from model_runtime import (
    add_lora,
    encode_sample,
    load_base_model,
    save_runtime_metadata,
)
from ner_utils import read_jsonl, validate_samples, write_json


def load_config(path: str | Path) -> dict[str, Any]:
    """读取训练 YAML，校验顶层配置结构；不合法时抛出 ValueError。"""
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("配置必须是 YAML 对象")
    required = {"model", "data", "training", "lora", "generation", "tensorboard"}
    if set(config) != required:
        raise ValueError(f"配置顶层字段必须为 {sorted(required)}")
    for key in required:
        if not isinstance(config[key], dict):
            raise ValueError(f"配置 {key} 必须为对象")
    return config


def _environment() -> dict[str, Any]:
    import torch

    names = ["torch", "transformers", "peft", "accelerate", "tensorboard", "PyYAML"]
    if torch.cuda.is_available():
        names.append("bitsandbytes")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


def _validate_training_config(config: dict[str, Any]) -> None:
    training = config["training"]
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "logging_steps",
    ):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f"training.{key} 必须为正整数")
    for section, key in (("training", "eval_batch_size"), ("generation", "batch_size")):
        value = config[section].get(key, 1)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{section}.{key} 必须为正整数")
    if training["epochs"] <= 0 or config["generation"]["max_new_tokens"] <= 0:
        raise ValueError("训练轮数及生成长度必须为正数")
    if not isinstance(config["tensorboard"].get("enabled"), bool):
        raise ValueError("tensorboard.enabled 应为布尔值")


def _create_run_directory(config: dict[str, Any], run_name: str | None) -> tuple[Path, str]:
    training = config["training"]
    run_name = run_name or f"tcm-ner-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    if run_name in (".", "..") or Path(run_name).name != run_name:
        raise ValueError("run_name 必须为单个目录名称")
    directory = Path(training["output_root"]) / run_name
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    return directory, run_name


def _load_training_samples(
    config: dict[str, Any], directory: Path, train_limit: int | None, eval_limit: int | None
) -> tuple[Path, dict[str, list[Sample]]]:
    processed = Path(config["data"]["processed_dir"])
    samples = {split: read_jsonl(processed / f"{split}.jsonl") for split in ("train", "dev")}
    for split, rows in samples.items():
        validate_samples(rows)
        if not rows:
            raise ValueError(f"{split} 数据为空")
    if {s["input"] for s in samples["train"]} & {s["input"] for s in samples["dev"]}:
        raise ValueError("训练和验证集存在相同原文，请重新运行 prepare_data.py")
    for split, limit in (("train", train_limit), ("dev", eval_limit)):
        if limit is not None:
            if limit <= 0:
                raise ValueError("样本限制必须大于 0")
            samples[split] = samples[split][:limit]
    audit_path = processed / "data_audit.json"
    if audit_path.exists():
        shutil.copyfile(audit_path, directory / "data_audit.json")
    return processed, cast(dict[str, list[Sample]], samples)


def _encode_training_samples(
    config: dict[str, Any],
    samples: dict[str, list[Sample]],
    tokenizer: Any,
    eos_id: int,
    model: Any,
) -> tuple[dict[str, list[EncodedSample]], dict[str, dict[str, int]]]:
    training = config["training"]
    features = {
        split: [encode_sample(row, tokenizer, eos_id, training["max_length"]) for row in rows]
        for split, rows in samples.items()
    }
    max_new = config["generation"]["max_new_tokens"]
    model_limit = getattr(model.config, "max_position_embeddings", training["max_length"])
    lengths = {}
    for split, encoded in features.items():
        prompt_lengths = [row["labels"].count(-100) for row in encoded]
        answer_lengths = [len(row["labels"]) - row["labels"].count(-100) for row in encoded]
        if max(answer_lengths) > max_new:
            raise ValueError(
                f"{split} 参考答案最长 {max(answer_lengths)} token 超过生成预算 {max_new}"
            )
        if max(prompt_lengths) + max_new > min(training["max_length"], model_limit):
            raise ValueError(f"{split} 提示+生成预算超出上下文，请调整 max_length/max_new_tokens")
        if max(len(row["input_ids"]) for row in encoded) > model_limit:
            raise ValueError(f"{split} 样本超过模型上下文 {model_limit}")
        lengths[split] = {
            "samples": len(encoded),
            "max_prompt": max(prompt_lengths),
            "max_answer": max(answer_lengths),
            "max_total": max(len(row["input_ids"]) for row in encoded),
        }
    return features, lengths


def _save_environment(
    config: dict[str, Any],
    directory: Path,
    processed: Path,
    model: Any,
    lengths: dict[str, dict[str, int]],
    train_limit: int | None,
    eval_limit: int | None,
) -> None:
    training = config["training"]
    environment = _environment()
    data_hashes = {
        split: hashlib.sha256((processed / f"{split}.jsonl").read_bytes()).hexdigest()
        for split in ("train", "dev", "test")
        if (processed / f"{split}.jsonl").exists()
    }
    metadata = {
        "config": config,
        "environment": environment,
        "data_sha256": data_hashes,
        "token_lengths": lengths,
        "train_limit": train_limit,
        "eval_limit": eval_limit,
        "effective_batch_size": training["batch_size"] * training["gradient_accumulation_steps"],
        "model_revision": getattr(model.config, "_commit_hash", None),
    }
    write_json(directory / "environment.json", metadata)
    print(f"训练目录：{directory.resolve()}；长度统计：{json.dumps(lengths, ensure_ascii=False)}")


def _create_trainer(
    config: dict[str, Any],
    directory: Path,
    run_name: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    dtype: torch.dtype,
    features: dict[str, list[EncodedSample]],
    samples: dict[str, list[Sample]],
) -> EntityTrainer:
    import torch
    from transformers import DataCollatorForSeq2Seq, TrainingArguments

    from ner_trainer import EntityTrainer

    training = config["training"]
    max_new = config["generation"]["max_new_tokens"]
    args = TrainingArguments(
        output_dir=str(directory),
        run_name=run_name,
        group_by_length=True,
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training.get("eval_batch_size", 1),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["epochs"],
        max_steps=training["max_steps"],
        warmup_ratio=training["warmup_ratio"],
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        weight_decay=0.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=training["logging_steps"],
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_entity_micro_f1",
        greater_is_better=True,
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        use_cpu=device.type == "cpu",
        seed=training["seed"],
        data_seed=training["seed"],
        logging_dir=str(directory / "tensorboard"),
        report_to="tensorboard" if config["tensorboard"]["enabled"] else "none",
        label_names=["labels"],
        prediction_loss_only=True,
        dataloader_pin_memory=device.type == "cuda",
    )
    trainer = EntityTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=features["train"],
        eval_dataset=features["dev"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        raw_eval_samples=samples["dev"],
        ner_max_length=training["max_length"],
        ner_max_new_tokens=max_new,
        ner_generation_batch_size=config["generation"].get("batch_size", 1),
    )
    return trainer


def _save_training_result(
    config: dict[str, Any],
    directory: Path,
    trainer: EntityTrainer,
    tokenizer: Any,
    samples: dict[str, list[Sample]],
    eos_id: int,
    result: Any,
    train_limit: int | None,
    eval_limit: int | None,
) -> None:
    training = config["training"]
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError("未产生最佳 checkpoint，请检查验证策略")
    adapter = directory / "best_adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    save_runtime_metadata(adapter, config, trainer.model, eos_id)
    # 保留验证原文用于最终重合补充评估，不含标签。
    write_json(
        adapter / "validation_texts.json",
        sorted({row["input"] for row in samples["dev"]}),
    )
    trainer.save_state()
    summary = {
        "best_validation_f1": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_adapter": str(adapter.resolve()),
        "global_step": trainer.state.global_step,
        "training_metrics": result.metrics,
        "tensorboard_log_dir": str((directory / "tensorboard").resolve())
        if config["tensorboard"]["enabled"]
        else None,
        "smoke_test": train_limit is not None
        or eval_limit is not None
        or training["max_steps"] > 0,
    }
    write_json(directory / "summary.json", summary)
    print(f"最佳验证实体 Micro-F1：{trainer.state.best_metric:.4f}；adapter：{adapter.resolve()}")


def run_training(
    config: dict[str, Any],
    run_name: str | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
) -> Path:
    """运行单设备训练，保存最佳适配器并返回独立实验目录。

    config 为完整训练配置；样本限制仅用于贯通检查，必须为正数。
    输入、长度或设备不合法时抛出 ValueError，目录已存在时抛出 FileExistsError。
    训练异常继续向调用方传播。
    """
    from transformers import set_seed

    _validate_training_config(config)
    directory, run_name = _create_run_directory(config, run_name)
    processed, samples = _load_training_samples(config, directory, train_limit, eval_limit)
    training = config["training"]
    set_seed(training["seed"])
    model, tokenizer, device, dtype = load_base_model(
        config["model"], training["device"], training=True
    )
    eos = model.config.eos_token_id
    eos_id = eos[0] if isinstance(eos, list) else eos
    features, lengths = _encode_training_samples(config, samples, tokenizer, eos_id, model)
    model = add_lora(model, config)
    _save_environment(config, directory, processed, model, lengths, train_limit, eval_limit)
    trainer = _create_trainer(
        config, directory, run_name, model, tokenizer, device, dtype, features, samples
    )
    result = trainer.train()
    _save_training_result(
        config,
        directory,
        trainer,
        tokenizer,
        samples,
        eos_id,
        result,
        train_limit,
        eval_limit,
    )
    return directory


def main() -> None:
    """解析命令行参数并执行对应任务，错误向命令行调用方传播。"""
    parser = argparse.ArgumentParser(description="单卡 Qwen2.5-7B QLoRA/LoRA 中医药 NER 微调")
    parser.add_argument("--config", default="configs/train.yaml", help="训练配置路径")
    parser.add_argument("--run-name", help="实验名称，不能复用已有输出目录")
    parser.add_argument("--train-limit", type=int, help="仅取前 N 条训练数据，供贯通检查")
    parser.add_argument("--eval-limit", type=int, help="仅取前 N 条验证数据，供贯通检查")
    parser.add_argument("--batch-size", type=int, help="覆盖单卡训练批大小")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="覆盖梯度累积步数")
    parser.add_argument("--generation-batch-size", type=int, help="覆盖验证生成批大小")
    parser.add_argument("--max-steps", type=int, help="覆盖优化器步数，正数用于贯通检查")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        help="启用或关闭 TensorBoard 日志",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.generation_batch_size is not None:
        config["generation"]["batch_size"] = args.generation_batch_size
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.tensorboard is not None:
        config["tensorboard"]["enabled"] = args.tensorboard
    run_training(config, args.run_name, args.train_limit, args.eval_limit)


if __name__ == "__main__":
    main()
