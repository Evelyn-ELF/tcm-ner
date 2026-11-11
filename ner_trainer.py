"""在 Trainer 验证阶段返回自主生成的实体 F1，供最佳 checkpoint 选择。"""

from pathlib import Path
from typing import Any, cast

from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput

from model_runtime import generate_records
from ner_types import Sample
from ner_utils import (
    flatten_metrics,
    print_report,
    save_report,
    score_predictions,
    write_jsonl,
)


class EntityTrainer(Trainer):
    """在标准验证 loss 之外计算生成式实体指标，用于最佳模型选择。"""

    def __init__(
        self,
        *args: Any,
        raw_eval_samples: list[Sample],
        ner_max_length: int,
        ner_max_new_tokens: int,
        ner_generation_batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        """接收 Trainer 参数及完整验证样本、上下文和生成预算。"""
        super().__init__(*args, **kwargs)
        self.raw_eval_samples = raw_eval_samples
        self.ner_max_length = ner_max_length
        self.ner_max_new_tokens = ner_max_new_tokens
        self.ner_generation_batch_size = ner_generation_batch_size

    def evaluation_loop(
        self,
        dataloader: Any,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """计算 loss 和实体 F1，保存预测报告后返回扩展的 Trainer 验证结果。"""
        # 先计算 loss，随后在 super.evaluate() 日志/回调/模型选择之前补充生成指标。
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=True,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        records = generate_records(
            self.accelerator.unwrap_model(self.model),
            self.processing_class,
            self.raw_eval_samples,
            self.ner_max_length,
            self.ner_max_new_tokens,
            progress=True,
            batch_size=self.ner_generation_batch_size,
        )
        report = score_predictions(records)
        directory = (
            Path(cast(str, self.args.output_dir)) / "validation" / f"step-{self.state.global_step}"
        )
        write_jsonl(directory / f"{metric_key_prefix}_predictions.jsonl", records)
        save_report(report, directory, metric_key_prefix)
        print_report(report)
        # 父类正常返回时 metrics 已初始化；兼容其可选类型声明。
        if output.metrics is None:
            raise RuntimeError("Trainer 验证结果缺少指标字典")
        output.metrics.update(flatten_metrics(report, metric_key_prefix))
        return output
