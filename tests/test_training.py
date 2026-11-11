"""真实随机微型 Qwen2 的 CPU 贯通验证，不下载模型、不代表 7B 训练效果。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from model_runtime import generate_records
from ner_utils import INSTRUCTION, build_prompt, read_jsonl
from prepare_data import prepare_data
from train import run_training

ROOT = Path(__file__).resolve().parents[1]


def create_tiny_model(path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    corpus = [
        INSTRUCTION,
        build_prompt({"input": "口苦头痛心悸"}),
        '[{"text":"口苦","type":"临床表现","start":0,"end":2}]',
        "[]",
    ]
    backend.train_from_iterator(
        corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["[EOS]", "[UNK]"])
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="[EOS]",
        pad_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            eos_token_id=0,
            pad_token_id=0,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(path)
    return model, tokenizer


class TrainingTests(unittest.TestCase):
    def test_generation_never_conditions_on_gold_and_restores_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, tokenizer = create_tiny_model(tmp)
            model.train()
            samples = [
                {"id": "a", "input": "口苦", "output": []},
                {
                    "id": "b",
                    "input": "口苦",
                    "output": [{"text": "口苦", "type": "临床表现", "start": 0, "end": 2}],
                },
            ]
            records = generate_records(model, tokenizer, samples, 2048, 8)
            self.assertEqual(records[0]["raw_output"], records[1]["raw_output"])
            self.assertTrue(model.training)
            with self.assertRaisesRegex(ValueError, "上下文"):
                generate_records(model, tokenizer, samples, 10, 8)
            self.assertTrue(model.training)

    def test_train_best_adapter_reload_and_cli_evaluate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "tiny"
            create_tiny_model(model_path)
            data = root / "data"
            data.mkdir()
            (data / "medical.train").write_text("口 B-临床表现\n苦 I-临床表现\n", encoding="utf-8")
            (data / "medical.dev").write_text("头 B-临床表现\n痛 I-临床表现\n", encoding="utf-8")
            (data / "medical.test").write_text("心 B-临床表现\n悸 I-临床表现\n", encoding="utf-8")
            processed = data / "processed"
            prepare_data(data, processed)
            config = yaml.safe_load((ROOT / "configs/train.yaml").read_text())
            config["model"].update(
                name=str(model_path), method="lora", precision="fp32", revision=None
            )
            config["data"]["processed_dir"] = str(processed)
            config["training"].update(
                device="cpu",
                output_root=str(root / "outputs"),
                max_steps=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
            )
            config["generation"]["max_new_tokens"] = 128
            config["tensorboard"]["enabled"] = True
            run_dir = run_training(config, run_name="贯通测试")
            adapter = run_dir / "best_adapter"
            self.assertTrue((adapter / "adapter_model.safetensors").exists())
            self.assertTrue((adapter / "runtime.json").exists())
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["best_validation_f1"], 0.0)
            self.assertIsNotNone(summary["best_checkpoint"])
            self.assertEqual(len(list(run_dir.glob("validation/step-*/eval_metrics.json"))), 1)
            self.assertTrue(list((run_dir / "tensorboard").glob("events.out.tfevents.*")))
            eval_dir = root / "evaluation"
            env = {
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
            command = [
                sys.executable,
                str(ROOT / "evaluate.py"),
                "--adapter",
                str(adapter),
                "--data",
                str(processed / "test.jsonl"),
                "--output-dir",
                str(eval_dir),
                "--tensorboard-log-dir",
                summary["tensorboard_log_dir"],
                "--device",
                "cpu",
            ]
            result = subprocess.run(
                command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            events = EventAccumulator(summary["tensorboard_log_dir"]).Reload()
            self.assertIn("train/loss", events.Tags()["scalars"])
            self.assertIn("eval/entity_micro_f1", events.Tags()["scalars"])
            self.assertIn("test/entity_micro_f1", events.Tags()["scalars"])
            report = json.loads((eval_dir / "test_metrics.json").read_text())
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(len(report["per_type"]), 10)
            records = read_jsonl(eval_dir / "test_predictions.jsonl")
            self.assertEqual(records[0]["id"], "test-000001")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "predict.py"),
                    "--adapter",
                    str(adapter),
                    "--text",
                    "口苦",
                    "--device",
                    "cpu",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("entities", json.loads(result.stdout))


if __name__ == "__main__":
    unittest.main()
