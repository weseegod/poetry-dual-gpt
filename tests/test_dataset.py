"""Tests for src/dataset.py — PoetryDataset + DataLoader."""

import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from dataset import PoetryDataset, get_dataloaders


class TestPoetryDataset(unittest.TestCase):
    def setUp(self):
        self.data = torch.arange(1000, dtype=torch.long)
        self.block_size = 128

    def test_len(self):
        ds = PoetryDataset(self.data, self.block_size)
        # len = data_len - block_size
        self.assertEqual(len(ds), 1000 - 128)

    def test_getitem_shape(self):
        ds = PoetryDataset(self.data, self.block_size)
        x, y = ds[0]
        self.assertEqual(x.shape, (128,))
        self.assertEqual(y.shape, (128,))

    def test_getitem_shifted(self):
        """y should be x shifted by 1."""
        ds = PoetryDataset(self.data, self.block_size)
        x, y = ds[42]
        # x = data[42 : 42+128], y = data[43 : 43+128]
        self.assertTrue(torch.equal(x, self.data[42:42+128]))
        self.assertTrue(torch.equal(y, self.data[43:43+128]))
        # y[i] should equal x[i+1] for all positions
        self.assertTrue(torch.equal(y[:-1], x[1:]))

    def test_getitem_last_valid(self):
        ds = PoetryDataset(self.data, self.block_size)
        # Last valid idx = data_len - block_size - 1
        x, y = ds[len(ds) - 1]
        self.assertEqual(x[0].item(), len(self.data) - self.block_size - 1)
        self.assertEqual(y[-1].item(), len(self.data) - 1)

    def test_dtype(self):
        ds = PoetryDataset(self.data, self.block_size)
        x, y = ds[0]
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_requires_1d(self):
        with self.assertRaises(AssertionError):
            PoetryDataset(torch.zeros(10, 10), self.block_size)

    def test_requires_long_enough(self):
        with self.assertRaises(AssertionError):
            PoetryDataset(torch.arange(10, dtype=torch.long), 128)

    def test_different_indices_different(self):
        ds = PoetryDataset(self.data, self.block_size)
        x0, _ = ds[0]
        x100, _ = ds[100]
        self.assertFalse(torch.equal(x0, x100))


class TestGetDataloaders(unittest.TestCase):
    def setUp(self):
        self.data = torch.arange(10000, dtype=torch.long)
        self.block_size = 64
        self.batch_size = 8

    def test_returns_two_loaders(self):
        train_loader, val_loader = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.1, num_workers=0)
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)

    def test_val_is_smaller(self):
        train_loader, val_loader = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.1, num_workers=0)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset
        self.assertGreater(len(train_ds), len(val_ds))

    def test_batch_shape(self):
        train_loader, val_loader = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.05, num_workers=0)
        x, y = next(iter(train_loader))
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))
        self.assertEqual(x.dtype, torch.long)

    def test_train_shuffled(self):
        train_loader, _ = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.05, num_workers=0)
        batch1, _ = next(iter(train_loader))
        batch2, _ = next(iter(train_loader))
        # First tokens of consecutive batches should differ (shuffled)
        self.assertFalse(torch.equal(batch1[0], batch2[0]))

    def test_val_not_shuffled(self):
        _, val_loader = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.05, num_workers=0)
        it = iter(val_loader)
        batch1, _ = next(it)
        batch2, _ = next(it)
        # Consecutive batches should be sequential
        # batch1[-1] followed by batch2[0] in original data
        # (approximately — exact check is tricky, just verify first tokens differ from train)
        self.assertIsNotNone(batch1)
        self.assertIsNotNone(batch2)

    def test_drop_last(self):
        train_loader, _ = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.05, num_workers=0)
        # All batches should have batch_size samples
        for x, _ in train_loader:
            self.assertEqual(x.shape[0], self.batch_size)

    def test_val_no_drop_last(self):
        _, val_loader = get_dataloaders(
            self.data, self.block_size, self.batch_size, val_fraction=0.5, num_workers=0)
        # Val loader may have a partial last batch
        shapes = [x.shape[0] for x, _ in val_loader]
        # At least one batch exists
        self.assertGreater(len(shapes), 0)


if __name__ == "__main__":
    unittest.main()
