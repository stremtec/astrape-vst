from __future__ import annotations

import unittest

import torch

from astrape.fsq import (
    DEFAULT_LEVELS,
    indices_to_codes,
    indices_to_level_indices,
    masked_fsq_cross_entropy,
)


def _product(levels: tuple[int, ...]) -> int:
    product = 1
    for level in levels:
        product *= level
    return product


class FSQHelperTests(unittest.TestCase):
    def test_invalid_token_indices_are_rejected(self):
        codebook_size = _product(DEFAULT_LEVELS)
        for token_index in (-1, codebook_size):
            with self.subTest(token_index=token_index):
                with self.assertRaisesRegex(ValueError, "FSQ token indices"):
                    indices_to_level_indices(
                        torch.tensor([token_index], dtype=torch.long),
                        DEFAULT_LEVELS,
                    )

        with self.assertRaisesRegex(ValueError, "FSQ token indices"):
            indices_to_codes(
                torch.tensor([[0, codebook_size]], dtype=torch.long),
                DEFAULT_LEVELS,
            )

    def test_all_false_mask_returns_zero_loss_and_metrics(self):
        levels = (2, 3)
        logits = (
            torch.randn(2, 2, 4, dtype=torch.float64),
            torch.randn(2, 3, 4, dtype=torch.float64),
        )
        token_indices = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        loss, accuracy, exact = masked_fsq_cross_entropy(
            logits, token_indices, mask, levels
        )

        for value in (loss, accuracy, exact):
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.device, logits[0].device)
            self.assertTrue(torch.isfinite(value))
            self.assertEqual(value.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
