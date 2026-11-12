# 中医药命名实体识别

使用 Qwen2.5-7B 基座模型进行生成式实体抽取，默认采用 QLoRA 微调。模型输出实体文字、类别和字符位置，评估要求类别与完整边界同时匹配。

目前已验证数据处理、计分和微型模型的 CPU 训练、保存及重载流程。**完整 7B CUDA/QLoRA 训练、显存占用和实际识别效果尚未验证。**

| 你想做什么 | 阅读入口 |
|---|---|
| 运行训练、评估或预测 | 本文 |
| 理解数据协议、模型流程和计分规则 | [技术方案](docs/中医药命名实体识别技术方案.md) |
| 修改代码、运行检查、启用提交钩子 | [代码规范](docs/代码规范.md) |
| 清洗数据、复核标注并重建数据集 | [数据清洗与标注复核手册](docs/数据清洗与标注复核手册.md) |
| 查看已完成的验证及其限制 | [开发验证记录](docs/开发验证记录.md) |

## 1. 安装环境

正式训练面向 Linux、Python 3.12 和单张 NVIDIA CUDA GPU。以下命令均在项目根目录执行；CPU 模式用于微型模型测试。

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# CUDA 12.4 安装示例，需与目标机器驱动兼容。
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

[训练配置](configs/train.yaml) 默认从 `/mnt/Qwen2.5-7B` 加载模型。按实际位置修改 `model.name`，也可填写 `Qwen/Qwen2.5-7B` 以下载模型。

默认配置使用 QLoRA；普通 LoRA 可设置 `model.method: lora`。显存是否足够，需要在目标 GPU 上通过短训练确认。

## 2. 处理数据

准备 `data/medical.train`、`data/medical.dev`、`data/medical.test`：每行一个字符和 BIO 标签，空行分隔样本。

```bash
python prepare_data.py
```

结果写入 `data/processed/`：三份 JSONL、`labels.json` 和 `data_audit.json`。程序保留原文字符，从训练集中去除重复记录、隔离标注冲突，并排除与验证或测试原文重合的样本。测试集中与验证集原文重合的样本也会排除。随后对三个集合应用经复核的标注修订和样本排除清单。

正式数据的样本数和排除记录见[数据审计报告](data/processed/data_audit.json)，标注口径和复核流程见[清洗手册](docs/数据清洗与标注复核手册.md)。

## 3. 训练

默认训练批大小为 16，梯度累积步数为 1，单卡有效批大小为 16。每轮验证先用 `training.eval_batch_size` 分批计算损失，再用 `generation.batch_size` 分批生成预测，验证损失和验证生成的默认批大小分别为 4 和 16。生成上限由 `generation.max_new_tokens` 控制，默认每条 1536 个 token。

默认使用 TensorBoard 在本地记录实验。先进行小规模检查，再运行完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --run-name smoke-qwen25 \
  --train-limit 32 --eval-limit 4 --max-steps 2

CUDA_VISIBLE_DEVICES=0 python train.py --run-name qwen25-qlora-01

CUDA_VISIBLE_DEVICES=0 python train.py --run-name qwen25-qlora-b8-a2-g8 \
  --batch-size 8 --gradient-accumulation-steps 2 --generation-batch-size 8
```

每次实验使用独立目录 `outputs/<run-name>/`，不允许覆盖已有目录。小规模检查只使用部分数据，不能作为正式评估结果。

超参数集中在 [configs/train.yaml](configs/train.yaml)，用 `--config` 指定其他配置。默认每个 epoch 生成验证集答案，按实体 Micro-F1 选择最佳 checkpoint；训练结束后保存到 `best_adapter/`。

训练日志位于 `outputs/<run-name>/tensorboard/`。在项目根目录运行以下命令查看：

```bash
tensorboard --logdir outputs
```

使用 `--no-tensorboard` 关闭事件日志，或在配置中设置 `tensorboard.enabled: false`。

## 4. 评估测试集

训练结束后，用最佳 adapter 运行最终测试：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --adapter outputs/qwen25-qlora-01/best_adapter \
  --data data/processed/test.jsonl \
  --batch-size 4 \
  --output-dir outputs/qwen25-qlora-01
```

测试集生成默认每批 4 条，可通过 `--batch-size` 调整。命令打印全部 10 类的 Precision、Recall、F1 和实体计数，保存原始预测及 JSON/CSV 报告。adapter 内保存了实际验证原文，程序据此额外生成排除验证重合文本后的测试报告；也可用 `--dev-data data/processed/dev.jsonl` 覆盖这份清单。

评估默认仅保存本地报告。添加 `--tensorboard-log-dir outputs/<run-name>/tensorboard` 可将聚合指标写入对应实验。目录也可从 `summary.json` 的 `tensorboard_log_dir` 字段获取。

重新计算已保存预测的指标，不需要加载模型：

```bash
python evaluate.py \
  --predictions outputs/qwen25-qlora-01/test_predictions.jsonl \
  --data data/processed/test.jsonl \
  --dev-data data/processed/dev.jsonl \
  --output-dir outputs/qwen25-qlora-01/recomputed
```

指定 `--data` 后，以该文件的真实标注为准。两份文件须具有完全相同的 ID 集合，且每个 ID 的原文须逐字一致。原文或样本范围变更后，请重新生成预测。未指定 `--data` 时，采用预测文件自带的原文和标注。

## 5. 预测新文本

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --adapter outputs/qwen25-qlora-01/best_adapter \
  --text '患者口苦，予当归、丹参治疗。'
```

输出包含 `entities`、`raw_output`、`parse_error`、`item_errors` 和 `hit_generation_limit`，以及输入原文 `text`。实体位置使用 Python 字符索引 `[start, end)`。输入过长会报错；无效 JSON 会保留错误信息，不自动修补。

adapter 仍依赖原始基座。使用本地基座时，`best_adapter/runtime.json` 保存绝对路径，迁移机器后，应确保该路径可用，或更新 `runtime.json` 中的 `model.name`，使其指向同一基座权重。

## 6. 查看结果

| 路径（相对于实验目录） | 内容 |
|---|---|
| `config.json`、`environment.json` | 实际配置、运行环境、数据哈希和长度统计 |
| `data_audit.json` | 数据审计副本，源文件存在时复制 |
| `checkpoint-*/` | 训练 checkpoint |
| `validation/step-*/` | 每次验证的原始预测和报告 |
| `best_adapter/` | 最佳 LoRA 权重、tokenizer 和运行元数据 |
| `trainer_state.json`、`summary.json` | 训练状态、最佳验证指标及 TensorBoard 日志目录 |
| `test_predictions.jsonl`、`test_metrics.{json,csv}` | 单独运行评估后生成的预测与主报告 |
| `test_without_dev_overlap_metrics.{json,csv}` | 有验证原文来源时生成的补充报告 |

`summary.json` 的 `global_step` 是训练结束步数，最佳模型位置看 `best_checkpoint`。评估和数据转换会覆盖同名输出文件；需要保留旧结果时使用新目录。

开发检查命令统一维护在[代码规范](docs/代码规范.md)。
