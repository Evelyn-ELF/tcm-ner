"""训练及推理共用的编码、模型加载和生成；重依赖在需要时导入。"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ner_types import EncodedSample, InputSample, Prediction, Sample

if TYPE_CHECKING:
    import torch

from ner_utils import (
    INSTRUCTION,
    LABELS,
    TEMPLATE_VERSION,
    build_prompt,
    parse_prediction,
    write_json,
)


def encode_sample(sample: Sample, tokenizer: Any, eos_id: int, max_length: int) -> EncodedSample:
    """编码提示、答案和 EOS，仅监督答案区；超出 max_length 时抛出 ValueError。"""
    prompt_ids = tokenizer(build_prompt(sample), add_special_tokens=False)["input_ids"]
    answer = json.dumps(sample["output"], ensure_ascii=False, separators=(",", ":"))
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"] + [eos_id]
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError(
            f"超长样本 {sample['id']}：需要 {len(ids)} token，配置仅 {max_length}，禁止截断"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def resolve_device(device: str) -> torch.device:
    """解析 cpu/cuda 单设备；多进程、不可用设备或多张可见 GPU 抛出 ValueError。"""
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("本项目支持单进程单设备，请勿使用 torchrun/多进程启动")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("没有可用 CUDA GPU；完整微调需 NVIDIA GPU，CPU 仅用于微型 LoRA 测试")
        if torch.cuda.device_count() != 1:
            raise ValueError("请通过 CUDA_VISIBLE_DEVICES 仅暴露一张 GPU")
        return torch.device("cuda:0")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError("device 仅支持 cuda 或 cpu")


def compute_dtype(device: torch.device, precision: str = "auto") -> torch.dtype:
    """根据设备选择计算精度；不支持的精度与设备组合抛出 ValueError。"""
    import torch

    if precision == "auto":
        return (
            torch.float32
            if device.type == "cpu"
            else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        )
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision not in mapping:
        raise ValueError("precision 应为 auto/bf16/fp16/fp32")
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU 测试仅支持 fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("当前 GPU 不支持 BF16")
    return mapping[precision]


def load_base_model(
    model_config: dict[str, Any], device_name: str, training: bool = False
) -> tuple[Any, Any, torch.device, torch.dtype]:
    """加载基座与 tokenizer，返回模型、tokenizer、设备和精度。

    配置需含 name/method；QLoRA 仅支持 CUDA，模型文件错误由底层库传播。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = resolve_device(device_name)
    dtype = compute_dtype(device, model_config.get("precision", "auto"))
    method = model_config["method"]
    if method not in ("qlora", "lora"):
        raise ValueError("微调方式仅支持 qlora 或 lora")
    if method == "qlora" and device.type != "cuda":
        raise ValueError("QLoRA 需要 CUDA；微型 CPU 测试请设置 method: lora")
    kwargs = {"revision": model_config.get("revision"), "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], **kwargs)
    loading = {"torch_dtype": dtype, "attn_implementation": "sdpa", **kwargs}
    if method == "qlora":
        loading["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        loading["device_map"] = {"": 0}
    model: Any = AutoModelForCausalLM.from_pretrained(model_config["name"], **loading)
    if method != "qlora":
        model.to(device)
    eos = model.config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    if not eos_ids or any(type(e) is not int for e in eos_ids):
        raise ValueError("模型配置缺少合法 EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(eos_ids[0])
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = eos_ids
    model.config.use_cache = not training
    return model, tokenizer, device, dtype


def add_lora(model: Any, config: dict[str, Any]) -> Any:
    """按 model、training 和 lora 配置注入适配器；无可训练参数时抛出 ValueError。"""
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if config["model"]["method"] == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["training"]["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
        ),
    )
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("没有可训练参数，LoRA 注入失败")
    model.print_trainable_parameters()
    return model


def generate_records(
    model: Any,
    tokenizer: Any,
    samples: Sequence[InputSample],
    max_length: int,
    max_new_tokens: int,
    progress: bool = False,
    batch_size: int = 1,
) -> list[Prediction]:
    """分批确定性生成并保留原始输出，不将真实标签放入提示。

    提示与生成预算超出上下文时抛出 ValueError；结束或异常时恢复模型模式。
    """
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("生成批大小必须为正整数")
    device = next(model.parameters()).device
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    context_limit = min(max_length, getattr(model.config, "max_position_embeddings", max_length))
    was_training = model.training
    padding_side = tokenizer.padding_side
    records: list[Prediction] = []
    model.eval()
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                prompts = tokenizer(
                    [build_prompt(sample) for sample in batch],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                for sample, length in zip(
                    batch, prompts["attention_mask"].sum(dim=1).tolist(), strict=True
                ):
                    if length + max_new_tokens > context_limit:
                        raise ValueError(
                            f"{sample['id']} 的提示+生成预算 {length + max_new_tokens} "
                            f"超过上下文 {context_limit}，请调整长度配置"
                        )
                width = prompts["input_ids"].shape[1]
                outputs = model.generate(
                    input_ids=prompts["input_ids"].to(device),
                    attention_mask=prompts["attention_mask"].to(device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )[:, width:].tolist()
                for sample, output in zip(batch, outputs, strict=True):
                    # 已结束的样本会补齐到批内最长输出，只保留首个结束符之前的内容。
                    end = next((i for i, token in enumerate(output) if token in eos_ids), None)
                    raw = tokenizer.decode(output[:end], skip_special_tokens=True)
                    parsed = parse_prediction(sample["input"], raw)
                    records.append(
                        {
                            **sample,
                            "raw_output": raw,
                            "predicted_entities": parsed["entities"],
                            "parse_error": parsed["parse_error"],
                            "item_errors": parsed["errors"],
                            "hit_generation_limit": end is None and len(output) >= max_new_tokens,
                        }
                    )
                if progress and (start // 50 != len(records) // 50 or len(records) == len(samples)):
                    print(f"已生成 {len(records)}/{len(samples)} 条", flush=True)
    finally:
        tokenizer.padding_side = padding_side
        model.train(was_training)
    return records


def save_runtime_metadata(
    path: str | Path, config: dict[str, Any], model: Any, eos_id: int
) -> None:
    """保存模板、标签、模型版本和生成配置，供适配器重载时校验。"""
    base_name = config["model"]["name"]
    if Path(base_name).exists():
        base_name = str(Path(base_name).resolve())
    model_config = {**config["model"], "name": base_name}
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved:
        model_config["revision"] = resolved
    write_json(
        Path(path) / "runtime.json",
        {
            "template_version": TEMPLATE_VERSION,
            "instruction": INSTRUCTION,
            "labels": list(LABELS),
            "model": model_config,
            "generation": config["generation"],
            "max_length": config["training"]["max_length"],
            "eos_id": eos_id,
        },
    )


def load_adapter(adapter_path: str | Path, device_name: str) -> tuple[Any, Any, dict[str, Any]]:
    """加载适配器及其基座，返回模型、tokenizer 和运行元数据。

    模板或标签版本不一致时抛出 ValueError；模型默认进入评估模式。
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    path = Path(adapter_path)
    meta = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
    if (
        meta["template_version"] != TEMPLATE_VERSION
        or meta["instruction"] != INSTRUCTION
        or meta["labels"] != list(LABELS)
    ):
        raise ValueError("adapter 的模板或类别版本不一致")
    model, _, _, _ = load_base_model(meta["model"], device_name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = PeftModel.from_pretrained(model, path, is_trainable=False)
    model.eval()
    return model, tokenizer, meta
