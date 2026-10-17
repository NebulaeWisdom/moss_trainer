"""CPU contracts for deterministic bucket plans and sample-weighted accumulation."""

import hashlib
import json
import random
import unittest
from collections import Counter

from moss_trainer.launch import configure_environment

configure_environment()  # Before torch/model imports: runtime caches stay in temp/.

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from moss_trainer.data import BucketBatchSampler
from moss_trainer.train import _sample_weighted_loss


class RecordingDataset:
    def __init__(self, bucket_keys):
        self.bucket_keys = bucket_keys
        self.requests = []

    def __len__(self):
        return len(self.bucket_keys)

    def __getitem__(self, request):
        self.requests.append(request)
        return request


def reference_batches(bucket_keys, batch_size, seed, epoch):
    """The original uncached sampler's ordering, including its stable seed encoding."""
    encoded = json.dumps((seed, epoch, "batches"), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big"))
    buckets = {}
    for index, key in enumerate(bucket_keys):
        buckets.setdefault(key, []).append(index)
    batches = []
    for key in sorted(buckets):
        indices = buckets[key].copy()
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batches.append(indices[start : start + batch_size])
    rng.shuffle(batches)
    return [[(index, epoch) for index in batch] for batch in batches]


class BucketBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        # Interleaved, unsorted buckets containing 5, 4, 2, and 1 records.
        self.keys = [
            (1024, 1024), (768, 1344), (1536, 640), (1024, 1024),
            (768, 1344), (640, 1536), (1024, 1024), (1536, 640),
            (768, 1344), (1024, 1024), (768, 1344), (1024, 1024),
        ]
        self.dataset = RecordingDataset(self.keys)

    def test_order_matches_original_sampler_with_mixed_buckets(self):
        for seed in (0, 42, 137):
            for epoch in (0, 3):
                for batch_size in (1, 2, 3, 4):
                    with self.subTest(seed=seed, epoch=epoch, batch_size=batch_size):
                        sampler = BucketBatchSampler(self.dataset, batch_size, seed)
                        sampler.set_epoch(epoch)
                        expected = reference_batches(self.keys, batch_size, seed, epoch)
                        self.assertEqual(list(sampler), expected)
                        self.assertEqual(list(sampler), expected)

    def test_all_records_once_with_every_bucket_tail(self):
        bucket_counts = Counter(self.keys)
        for batch_size in (2, 3, 5):
            with self.subTest(batch_size=batch_size):
                sampler = BucketBatchSampler(self.dataset, batch_size, 42)
                sampler.set_epoch(2)
                batches = list(sampler)
                self.assertEqual(len(batches), sampler.total_batches)
                self.assertEqual(sorted(index for batch in batches for index, _ in batch),
                                 list(range(len(self.keys))))
                sizes = {key: [] for key in bucket_counts}
                for batch in batches:
                    key = self.keys[batch[0][0]]
                    self.assertEqual({self.keys[index] for index, _ in batch}, {key})
                    self.assertEqual({epoch for _, epoch in batch}, {2})
                    sizes[key].append(len(batch))
                for key, count in bucket_counts.items():
                    full, tail = divmod(count, batch_size)
                    expected = [batch_size] * full + ([tail] if tail else [])
                    self.assertEqual(sorted(sizes[key]), sorted(expected))

    def test_sample_counts_match_full_partial_and_resumed_windows(self):
        sampler = BucketBatchSampler(self.dataset, 3, 42)
        for epoch in (0, 4):
            expected = reference_batches(self.keys, 3, 42, epoch)
            for resume_batch in (0, 2, len(expected) - 1, len(expected)):
                with self.subTest(epoch=epoch, resume_batch=resume_batch):
                    sampler.set_epoch(epoch, start_batch=resume_batch)
                    for start in range(resume_batch, len(expected) + 1):
                        for count in range(len(expected) - start + 1):
                            self.assertEqual(sampler.sample_count(start, count),
                                             sum(len(batch) for batch in expected[start : start + count]))
                    self.assertEqual(len(sampler), len(expected) - resume_batch)
                    self.assertEqual(list(sampler), expected[resume_batch:])
        self.assertEqual(self.dataset.requests, [])

    def test_resume_never_loads_skipped_records_or_preloads_counted_window(self):
        sampler = BucketBatchSampler(self.dataset, 3, 42)
        sampler.set_epoch(3, start_batch=2)
        expected = reference_batches(self.keys, 3, 42, 3)[2:]
        self.assertEqual(sampler.sample_count(2, len(expected)), sum(map(len, expected)))
        self.assertEqual(self.dataset.requests, [])
        loader = DataLoader(self.dataset, batch_sampler=sampler, collate_fn=list, num_workers=0,
                            generator=torch.Generator(device="cpu").manual_seed(42))
        self.assertEqual(list(loader), expected)
        self.assertEqual(self.dataset.requests, [request for batch in expected for request in batch])

    def test_changing_epoch_refreshes_cached_order_and_counts(self):
        sampler = BucketBatchSampler(self.dataset, 3, 42)
        for epoch in (0, 1, 5, 0):
            with self.subTest(epoch=epoch):
                sampler.set_epoch(epoch)
                expected = reference_batches(self.keys, 3, 42, epoch)
                for start, batch in enumerate(expected):
                    self.assertEqual(sampler.sample_count(start, 1), len(batch))
                self.assertEqual(list(sampler), expected)

    def test_started_iterator_retains_captured_epoch_and_resume_position(self):
        sampler = BucketBatchSampler(self.dataset, 3, 42)
        sampler.set_epoch(2, start_batch=1)
        expected_old = reference_batches(self.keys, 3, 42, 2)
        old_iterator = iter(sampler)
        self.assertEqual(next(old_iterator), expected_old[1])
        sampler.set_epoch(6, start_batch=2)
        expected_new = reference_batches(self.keys, 3, 42, 6)
        self.assertEqual(sampler.sample_count(2, 1), len(expected_new[2]))
        self.assertEqual(list(old_iterator), expected_old[2:])
        self.assertEqual(list(sampler), expected_new[2:])

    def test_iterator_captures_epoch_on_first_next_not_construction(self):
        sampler = BucketBatchSampler(self.dataset, 3, 42)
        sampler.set_epoch(2, start_batch=1)
        iterator = iter(sampler)
        sampler.set_epoch(6, start_batch=2)
        self.assertEqual(list(iterator), reference_batches(self.keys, 3, 42, 6)[2:])


def regression_batches(sizes):
    batches = []
    for batch_index, size in enumerate(sizes):
        # Different spatial lengths also require a sample mean, not an element mean.
        width = batch_index + 2
        values = torch.arange(size * width * 2, dtype=torch.float64).reshape(size, width, 2)
        values = values / (7 + batch_index) + batch_index * 0.3
        targets = torch.arange(size * width, dtype=torch.float64).reshape(size, width) / 11
        batches.append((values, targets))
    return batches


class WeightedAccumulationTests(unittest.TestCase):
    def accumulate(self, batches, *, legacy=False):
        parameter = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
        window_samples = sum(values.shape[0] for values, _ in batches)
        total_loss = None
        for values, targets in batches:
            batch_mean = F.mse_loss((values * parameter).sum(dim=-1), targets)
            if legacy:
                (batch_mean / len(batches)).backward()
                logged_loss = batch_mean.detach()
            else:
                loss = _sample_weighted_loss(batch_mean, values.shape[0], window_samples)
                loss.backward()
                logged_loss = loss.detach()
            if total_loss is None:
                total_loss = logged_loss
            else:
                total_loss.add_(logged_loss)
        loss_value = total_loss.item() / len(batches) if legacy else total_loss.item()
        return loss_value, parameter.grad

    def test_uneven_batches_match_per_sample_mean_loss_and_gradient(self):
        for sizes in ([2, 1], [3, 2, 1]):
            with self.subTest(sizes=sizes):
                batches = regression_batches(sizes)
                actual_loss, actual_gradient = self.accumulate(batches)
                parameter = torch.tensor([0.4, -0.7], dtype=torch.float64, requires_grad=True)
                losses = [F.mse_loss((values * parameter).sum(dim=-1), target)
                          for batch, targets in batches for values, target in zip(batch, targets)]
                expected = torch.stack(losses).mean()
                expected.backward()
                torch.testing.assert_close(actual_loss, expected.item(), rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(actual_gradient, parameter.grad, rtol=1e-12, atol=1e-12)

    def test_batch_size_one_matches_previous_microbatch_mean(self):
        for count in (1, 3, 8):
            with self.subTest(microbatch_count=count):
                batches = regression_batches([1] * count)
                actual_loss, actual_gradient = self.accumulate(batches)
                expected_loss, expected_gradient = self.accumulate(batches, legacy=True)
                torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
