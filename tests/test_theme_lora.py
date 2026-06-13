# -*- coding: utf-8 -*-
import unittest

import torch

from build_theme_dataset import build_theme_vocab, encode_records, ThemedPoemRecord
from model import CharGPT
from poetry_format import format_structured_poem


class ThemeLoRATests(unittest.TestCase):
    def test_format_structured_poem_with_theme(self) -> None:
        text = format_structured_poem("七绝", "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。", theme="战争")
        self.assertTrue(text.startswith("【七绝】【战争】"))
        self.assertTrue(text.endswith("@"))

    def test_encode_records_emits_theme_labels(self) -> None:
        vocab_chars = sorted(set("【七绝】【战争】秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山|/@\n"))
        vocab_stoi = {ch: idx for idx, ch in enumerate(vocab_chars)}
        records = [
            ThemedPoemRecord(
                genre="七绝",
                theme="战争",
                poem_text="秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
            )
        ]
        theme_vocab = build_theme_vocab(records)
        full_text, token_ids, tone_ids, rhyme_ids, theme_ids = encode_records(
            records,
            vocab_stoi=vocab_stoi,
            theme_stoi=theme_vocab["stoi"],
            theme_pad_id=int(theme_vocab["pad_id"]),
        )
        self.assertEqual(len(full_text), len(token_ids))
        self.assertEqual(len(token_ids), len(tone_ids))
        self.assertEqual(len(token_ids), len(rhyme_ids))
        self.assertEqual(len(token_ids), len(theme_ids))
        self.assertTrue(torch.all(theme_ids == 0))

    def test_model_supports_theme_and_lora(self) -> None:
        model = CharGPT(
            vocab_size=64,
            block_size=16,
            d_model=32,
            n_head=4,
            n_layer=2,
            d_ff=64,
            dropout=0.0,
            use_prosody=True,
            num_tones=4,
            num_rhymes=15,
            use_aux_loss=True,
            aux_loss_weight=0.1,
            use_theme=True,
            num_themes=10,
        )
        model.enable_lora(rank=4, alpha=8.0, dropout=0.0)
        model.freeze_base_for_theme_lora()
        idx = torch.randint(0, 64, (2, 16))
        targets = torch.randint(0, 64, (2, 16))
        tone_ids = torch.randint(0, 4, (2, 16))
        rhyme_ids = torch.randint(0, 15, (2, 16))
        tone_targets = torch.randint(0, 4, (2, 16))
        rhyme_targets = torch.randint(0, 15, (2, 16))
        theme_ids = torch.randint(0, 10, (2, 16))
        logits, loss = model(
            idx,
            targets=targets,
            tone_ids=tone_ids,
            rhyme_ids=rhyme_ids,
            tone_targets=tone_targets,
            rhyme_targets=rhyme_targets,
            theme_ids=theme_ids,
        )
        self.assertEqual(tuple(logits.shape), (2, 16, 64))
        self.assertIsNotNone(loss)
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(any("lora_" in name for name in trainable))
        self.assertTrue(any("theme_emb" in name for name in trainable))


if __name__ == "__main__":
    unittest.main()
