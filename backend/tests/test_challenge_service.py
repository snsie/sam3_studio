from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.challenge_service import ChallengeService


class ChallengeServiceTests(unittest.TestCase):
    def test_extract_multi_class_masks_merges_regions_with_same_class_value(
        self,
    ) -> None:
        label_levels = np.array([0, 42, 85, 128, 170, 212, 255], dtype=np.uint8)
        label_map = np.zeros((24, 24), dtype=np.uint8)
        label_map[2:6, 3:8] = label_levels[2]
        label_map[12:16, 14:20] = label_levels[2]
        label_map[7:15, 8:12] = label_levels[6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = Path(tmp_dir) / "labels_00000.jpg"
            Image.fromarray(label_map).save(mask_path, format="JPEG", quality=100)

            service = ChallengeService.__new__(ChallengeService)
            masks = service._extract_multi_class_masks(mask_path)

        self.assertEqual(len(masks), 2)
        self.assertEqual(int(masks[0].sum()), 44)
        self.assertEqual(int(masks[1].sum()), 32)
        self.assertTrue(bool(masks[0][3, 4]))
        self.assertTrue(bool(masks[0][13, 15]))
        self.assertFalse(bool(masks[0][9, 9]))
        self.assertTrue(bool(masks[1][9, 9]))


if __name__ == "__main__":
    unittest.main()
