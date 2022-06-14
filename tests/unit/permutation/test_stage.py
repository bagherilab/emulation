import unittest

from permutation.stage import Stage


class StageTestSuite(unittest.TestCase):
    def test_stage_defaults(self) -> None:
        self.assertEqual(Stage.TRAIN.value, "train")
        self.assertEqual(Stage.TEST.value, "test")
        self.assertEqual(Stage.VAL.value, "val")
        self.assertEqual(Stage.PERM.value, "perm")
