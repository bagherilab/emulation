import unittest

from permutation.metrics import SequentialMetric, BatchMetric


class SequentialMetricTests(unittest.TestCase):
    def test_empty_defaults(self) -> None:
        testSeqMetric = SequentialMetric("test")

        self.assertFalse(testSeqMetric.values)
        self.assertFalse(testSeqMetric.nums)
        self.assertEqual(testSeqMetric.name, "test")

    def test_update(self) -> None:
        testSeqMetric = SequentialMetric("test")
        testSeqMetric.update(1.0, 1)

        expected_values = [1.0]
        expected_nums = [1]

        self.assertEqual(testSeqMetric.values, expected_values)
        self.assertEqual(testSeqMetric.nums, expected_nums)

    def test_multiple_updates(self) -> None:
        testSeqMetric = SequentialMetric("test")
        testSeqMetric.update(1.0, 1)
        testSeqMetric.update(2.0, 2)

        expected_values = [1.0, 2.0]
        expected_nums = [1, 2]

        self.assertEqual(testSeqMetric.values, expected_values)
        self.assertEqual(testSeqMetric.nums, expected_nums)

    def test_mutability(self) -> None:
        """test for intializing empty lists correctly"""
        _ = SequentialMetric("foo")
        _.update(1.0, 1)
        testSeqMetric = SequentialMetric("test")

        self.assertFalse(testSeqMetric.values)
        self.assertFalse(testSeqMetric.nums)

    def test_batchupdate(self) -> None:
        testSeqMetric = SequentialMetric("test")
        testSeqMetric.batchupdate([1.0, 2.0, 3.0], [1, 2, 3])

        expected_values = [1.0, 2.0, 3.0]
        expected_nums = [1, 2, 3]

        self.assertEqual(testSeqMetric.values, expected_values)
        self.assertEqual(testSeqMetric.nums, expected_nums)

    def test_backorder(self) -> None:
        """test for ensuring sequential n"""

        def add_nonsequential_args():
            testSeqMetric = SequentialMetric("test")
            testSeqMetric.update(2.0, 2)
            testSeqMetric.update(1.0, 1)

        self.assertRaises(ValueError, add_nonsequential_args)


class BatchMetricTests(unittest.TestCase):
    def test_empty_defaults(self) -> None:
        testBatchMetric = BatchMetric("test")

        self.assertFalse(testBatchMetric.values)
        self.assertEqual(testBatchMetric.name, "test")
        self.assertEqual(testBatchMetric.total, 0.0)
        self.assertEqual(testBatchMetric.average, 0.0)
        self.assertEqual(testBatchMetric.length, 0)

    def test_update(self) -> None:
        testBatchMetric = BatchMetric("test")
        testBatchMetric.update(1.0)

        self.assertEqual(testBatchMetric.values, [1.0])
        self.assertEqual(testBatchMetric.total, 1.0)
        self.assertEqual(testBatchMetric.average, 1.0)
        self.assertEqual(testBatchMetric.length, 1)

    def test_multiple_updates(self) -> None:
        testBatchMetric = BatchMetric("test")
        testBatchMetric.update(1.0)
        testBatchMetric.update(2.0)

        self.assertEqual(testBatchMetric.values, [1.0, 2.0])
        self.assertEqual(testBatchMetric.total, 3.0)
        self.assertEqual(testBatchMetric.average, 1.5)
        self.assertEqual(testBatchMetric.length, 2)

    def test_mutability(self) -> None:
        """test for intializing empty lists correctly"""
        _ = BatchMetric("foo")
        _.update(1.0)
        testBatchMetric = BatchMetric("test")

        self.assertFalse(testBatchMetric.values)

    def test_batchupdate(self) -> None:
        testBatchMetric = BatchMetric("test")
        testBatchMetric.batchupdate([1.0, 2.0, 3.0])

        self.assertEqual(testBatchMetric.values, [1.0, 2.0, 3.0])
        self.assertEqual(testBatchMetric.total, 6.0)
        self.assertEqual(testBatchMetric.average, 2.0)
        self.assertEqual(testBatchMetric.length, 3)
