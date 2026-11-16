# 数据目录

| 位置 | 用途 |
|---|---|
| `medical.train`、`medical.dev`、`medical.test` | 字符级 BIO 源数据 |
| `annotation_revisions.json`、`split_exclusions.json` | 基础标注裁定和样本隔离记录 |
| `processed/` | `python prepare_data.py` 生成的基础数据，供默认训练配置使用 |
| `processed-exam-results/` | 第 5 次训练和结项评估使用的数据 |
| `archive/processed-weak-resample/` | 第 4 次重采样实验，仅供追溯 |
| `archive/processed-classic-50/` | 第 6 次古籍补样实验 |
| `archive/validation-reviews/` | 验证集草案和裁定记录 |

第 5 次训练配置读取 `processed-exam-results/` 下的 `train.jsonl`、`dev.jsonl` 和 `test.jsonl`。其中训练集 5490 条，验证集 655 条，测试集 656 条。结项验证复算使用同目录的 `dev-clinical-reviewed.jsonl`，样本 ID 和原文与训练时的验证集一致。第 5 次补标记录见同目录的 `confirmed_additions.json`。

基础数据可从 BIO 源文件重建。第 5 次数据包含后续补标和重采样，复现时直接使用保存的快照。历史训练配置已指向归档后的路径。从项目根目录运行命令。

[结项报告](../docs/结项报告.md) · [标注与清洗手册](../docs/数据清洗与标注复核手册.md)
