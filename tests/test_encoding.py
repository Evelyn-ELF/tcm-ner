import unittest

from tokenizers import Tokenizer, models
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from model_runtime import encode_sample
from ner_utils import INSTRUCTION, build_prompt


class CharacterTokenizer:
    pad_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 10 for c in text]}


class EncodingTests(unittest.TestCase):
    def test_only_answer_and_eos_are_supervised(self):
        sample = {"id": "1", "input": "口苦", "output": [], "instruction": INSTRUCTION}
        encoded = encode_sample(sample, CharacterTokenizer(), 1, 2048)
        prefix = [ord(c) + 10 for c in build_prompt(sample)]
        self.assertEqual(encoded["input_ids"][: len(prefix)], prefix)
        self.assertEqual(encoded["labels"][: len(prefix)], [-100] * len(prefix))
        self.assertEqual(encoded["labels"][len(prefix) :], [101, 103, 1])
        self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))

    def test_long_samples_fail_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, "超长.*1"):
            encode_sample({"id": "1", "input": "口苦", "output": []}, CharacterTokenizer(), 1, 10)

    def test_padding_does_not_mask_real_eos(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]")
            ),
            pad_token="[EOS]",
            eos_token="[EOS]",
            unk_token="[UNK]",
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=None, padding=True, label_pad_token_id=-100
        )
        batch = collator(
            [
                {
                    "input_ids": [10, 20, 1],
                    "attention_mask": [1, 1, 1],
                    "labels": [-100, 20, 1],
                },
                {"input_ids": [10, 1], "attention_mask": [1, 1], "labels": [-100, 1]},
            ]
        )
        self.assertEqual(batch["labels"].tolist(), [[-100, 20, 1], [-100, 1, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["input_ids"].tolist(), [[10, 20, 1], [10, 1, 1]])


if __name__ == "__main__":
    unittest.main()
