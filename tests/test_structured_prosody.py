# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import torch

from dataset import PoetryBlockDataset
from model import CharGPT
from prosody import RHYME_UNK, TONE_PUNCT, build_label_sequences


class StructuredProsodyTests(unittest.TestCase):
    def test_build_label_sequences_marks_prefix_and_structure_as_special(self) -> None:
        text = "【七绝】春眠不觉晓|处处闻啼鸟/@\n"
        tone_ids, rhyme_ids = build_label_sequences(text)
        self.assertEqual(len(text), len(tone_ids))
        self.assertEqual(len(text), len(rhyme_ids))
        self.assertEqual(tone_ids[0], TONE_PUNCT)
        self.assertEqual(tone_ids[1], TONE_PUNCT)
        self.assertEqual(tone_ids[2], TONE_PUNCT)
        self.assertEqual(tone_ids[3], TONE_PUNCT)
        self.assertEqual(rhyme_ids[0], RHYME_UNK)
        self.assertEqual(tone_ids[text.index("|")], TONE_PUNCT)
        self.assertEqual(rhyme_ids[text.index("@")], RHYME_UNK)

    def test_dataset_returns_prosody_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = torch.arange(32, dtype=torch.long)
            tone = torch.arange(32, dtype=torch.long) % 4
            rhyme = torch.arange(32, dtype=torch.long) % 15
            data_path = root / "data.pt"
            tone_path = root / "tone.pt"
            rhyme_path = root / "rhyme.pt"
            torch.save(data, data_path)
            torch.save(tone, tone_path)
            torch.save(rhyme, rhyme_path)

            ds = PoetryBlockDataset(
                str(data_path),
                block_size=8,
                num_samples=4,
                sample_random=False,
                tone_path=str(tone_path),
                rhyme_path=str(rhyme_path),
            )
            x, y, tone_x, tone_y, rhyme_x, rhyme_y = ds[0]
            self.assertEqual(tuple(x.shape), (8,))
            self.assertTrue(torch.equal(y, data[1:9]))
            self.assertTrue(torch.equal(tone_x, tone[:8]))
            self.assertTrue(torch.equal(tone_y, tone[1:9]))
            self.assertTrue(torch.equal(rhyme_x, rhyme[:8]))
            self.assertTrue(torch.equal(rhyme_y, rhyme[1:9]))

    def test_model_forward_supports_aux_loss(self) -> None:
        model = CharGPT(
            vocab_size=32,
            block_size=8,
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
        )
        idx = torch.randint(0, 32, (2, 8))
        targets = torch.randint(0, 32, (2, 8))
        tone_ids = torch.randint(0, 4, (2, 8))
        rhyme_ids = torch.randint(0, 15, (2, 8))
        tone_targets = torch.randint(0, 4, (2, 8))
        rhyme_targets = torch.randint(0, 15, (2, 8))

        logits, loss = model(
            idx,
            targets=targets,
            tone_ids=tone_ids,
            rhyme_ids=rhyme_ids,
            tone_targets=tone_targets,
            rhyme_targets=rhyme_targets,
        )
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertIsNotNone(loss)
        self.assertGreater(float(loss.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
