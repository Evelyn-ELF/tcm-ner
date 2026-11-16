# 中医药命名实体识别

使用 Qwen2.5-7B 做生成式实体识别，覆盖中药、方剂、临床表现、中医证候、中医诊断、西医诊断、中医治则、中医治疗、西医治疗和其他治疗。项目已完成单卡 CUDA QLoRA 训练及验证、测试评估。

**结项模型为第 5 次训练的第 1032 步适配器，使用原提示。** 复核版验证集 Micro-F1 为 **83.66%**，测试集 Micro-F1 为 **77.49%**。评估数据、逐类指标和历史结果见[结项报告](docs/结项报告.md)。

## 环境与模型

在项目根目录执行。正式训练和第 5 次适配器的推理需要 NVIDIA CUDA GPU。以下 PyTorch 安装命令适用于兼容 CUDA 12.4 的驱动。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

将训练机保存的 `best_adapter/` 完整目录放到 `outputs/qwen25-qlora-05-exam-results/best_adapter/`。Git 不包含模型产物。适配器需要训练时的原始 Qwen2.5-7B 基座。若基座迁移，修改适配器 `runtime.json` 中的 `model.name`，指向同一份权重。

## 预测与评估

```bash
python predict.py --device cuda \
  --adapter outputs/qwen25-qlora-05-exam-results/best_adapter \
  --text '患者口苦，予当归、丹参治疗。'

python evaluate.py --device cuda \
  --adapter outputs/qwen25-qlora-05-exam-results/best_adapter \
  --data data/processed-exam-results/test.jsonl \
  --batch-size 32 --final-report --output-dir outputs/结项测试复评
```

预测结果包含实体文本、类别和原文字符区间 `[start, end)`。评估按类别和完整跨度严格匹配，输出预测 JSONL、指标 JSON/CSV 和中文测试报告。

已有预测时可在 CPU 上离线复算。预测与标注必须具有相同样本 ID 和原文，标注可以是修订版。

```bash
python evaluate.py \
  --predictions outputs/qwen25-qlora-05-exam-results/validation/step-1032/eval_predictions.jsonl \
  --data data/processed-exam-results/dev-clinical-reviewed.jsonl \
  --output-dir outputs/结项验证复算
```

## 复现第 5 次训练

```bash
python train.py --config data/processed-exam-results/train-config.json \
  --run-name qwen25-qlora-05-repro --generation-batch-size 32
```

训练使用配置目录的 `dev.jsonl` 按 Micro-F1 选出最佳 checkpoint，结项时再离线使用 `dev-clinical-reviewed.jsonl` 复算。每次训练须用新的实验名，结果保存在 `outputs/<run-name>/`，其中 `summary.json` 记录最佳步数，`best_adapter/` 保存可重载的适配器。

如需从字符级 BIO 原始文件重建基础数据，执行 `python prepare_data.py`，生成 `data/processed/`。第 5 次的补标与重采样数据直接使用上述实验目录。

- [数据目录索引](data/README.md)
- [技术说明](docs/中医药命名实体识别技术方案.md)
- [数据与标注手册](docs/数据清洗与标注复核手册.md)
- [代码规范与检查命令](docs/代码规范.md)
- [历史档案索引](docs/历史档案/README.md)
