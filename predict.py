"""加载已保存的 adapter，对一段新文本抽取实体。"""

import argparse
import json


def main() -> None:
    """解析命令行参数并执行对应任务，错误向命令行调用方传播。"""
    parser = argparse.ArgumentParser(description="中医药命名实体抽取")
    parser.add_argument("--adapter", required=True, help="best_adapter 目录")
    parser.add_argument("--text", required=True, help="待识别原文")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if not args.text:
        parser.error("文本不能为空")
    from model_runtime import generate_records, load_adapter

    model, tokenizer, metadata = load_adapter(args.adapter, args.device)
    record = generate_records(
        model,
        tokenizer,
        [{"id": "predict-1", "input": args.text}],
        metadata["max_length"],
        metadata["generation"]["max_new_tokens"],
    )[0]
    print(
        json.dumps(
            {
                "text": args.text,
                "entities": record["predicted_entities"],
                "raw_output": record["raw_output"],
                "parse_error": record["parse_error"],
                "item_errors": record["item_errors"],
                "hit_generation_limit": record["hit_generation_limit"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
