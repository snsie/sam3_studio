from __future__ import annotations

import unittest

import numpy as np

from app.services.bbbc038_data import (
    ImageType,
    classify_image_type,
    masks_to_label_map,
    rle_decode,
    rle_encode,
    suppress_duplicate_masks,
)
from app.services.nuclei_metrics import (
    DSB_IOU_THRESHOLDS,
    aggregate_metrics,
    classify_matches,
    compute_image_metrics,
    iou_matrix,
)


def square(shape: tuple[int, int], top: int, left: int, size: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


class RunLengthTests(unittest.TestCase):
    def test_decode_runs_top_to_bottom_then_left_to_right(self) -> None:
        # Pixel 1 is (row 0, col 0); pixel 4 starts column 1 of a 3-row image.
        mask = rle_decode("1 2 4 1", height=3, width=2)
        expected = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    def test_encode_round_trips(self) -> None:
        rng = np.random.default_rng(0)
        mask = rng.random((17, 23)) > 0.6
        mask[0, 0] = True
        mask[-1, -1] = True
        np.testing.assert_array_equal(rle_decode(rle_encode(mask), 17, 23), mask)


class DsbMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shape = (40, 40)
        self.gt = masks_to_label_map(
            [square(self.shape, 2, 2, 8), square(self.shape, 20, 20, 10)], self.shape
        )

    def test_perfect_prediction_scores_one(self) -> None:
        metrics = compute_image_metrics(self.gt, self.gt.copy())
        self.assertEqual(metrics.dsb_score, 1.0)
        self.assertEqual(metrics.f1_at_50, 1.0)
        self.assertEqual(metrics.pixel_dice, 1.0)
        self.assertEqual(metrics.count_error, 0)

    def test_missing_object_counts_as_false_negative_at_every_threshold(self) -> None:
        prediction = masks_to_label_map([square(self.shape, 2, 2, 8)], self.shape)
        metrics = compute_image_metrics(self.gt, prediction)
        # TP=1, FP=0, FN=1 at every threshold -> 1/2.
        self.assertAlmostEqual(metrics.dsb_score, 0.5)
        self.assertAlmostEqual(metrics.f1_at_50, 2 / 3)
        self.assertEqual(metrics.count_error, -1)
        self.assertEqual(metrics.abs_count_error, 1)

    def test_partial_overlap_only_counts_below_its_iou(self) -> None:
        # 8x8 GT square vs the same square shifted by 1 column: IoU = 56 / 72 = 0.777...
        gt = masks_to_label_map([square(self.shape, 2, 2, 8)], self.shape)
        prediction = masks_to_label_map([square(self.shape, 2, 3, 8)], self.shape)
        self.assertAlmostEqual(float(iou_matrix(gt, prediction)[0, 0]), 56 / 72)
        metrics = compute_image_metrics(gt, prediction)
        passing = sum(threshold < 56 / 72 for threshold in DSB_IOU_THRESHOLDS)
        self.assertAlmostEqual(metrics.dsb_score, passing / len(DSB_IOU_THRESHOLDS))
        self.assertEqual(metrics.f1_at_50, 1.0)

    def test_empty_prediction_scores_zero(self) -> None:
        metrics = compute_image_metrics(self.gt, np.zeros(self.shape, dtype=np.int32))
        self.assertEqual(metrics.dsb_score, 0.0)
        self.assertEqual(metrics.instance_mean_iou, 0.0)

    def test_classify_matches_labels_tp_fp_fn(self) -> None:
        prediction = masks_to_label_map(
            [square(self.shape, 2, 2, 8), square(self.shape, 30, 2, 5)], self.shape
        )
        matches = classify_matches(self.gt, prediction)
        self.assertEqual(matches.true_positive_pred, [1])
        self.assertEqual(matches.false_positive_pred, [2])
        self.assertEqual(matches.false_negative_gt, [2])
        self.assertEqual(matches.matched_gt_for_pred, {1: 1})

    def test_aggregate_averages_over_images(self) -> None:
        perfect = compute_image_metrics(self.gt, self.gt).to_dict()
        empty = compute_image_metrics(self.gt, np.zeros(self.shape, dtype=np.int32)).to_dict()
        summary = aggregate_metrics([perfect, empty])
        self.assertEqual(summary["image_count"], 2)
        self.assertAlmostEqual(summary["dsb_score"], 0.5)
        self.assertEqual(summary["gt_count_total"], 4)
        self.assertEqual(len(summary["ap_per_threshold"]), len(DSB_IOU_THRESHOLDS))


class LabelMapTests(unittest.TestCase):
    def test_higher_scores_win_overlapping_pixels(self) -> None:
        shape = (10, 10)
        low = square(shape, 0, 0, 6)
        high = square(shape, 3, 3, 6)
        labels = masks_to_label_map([high, low], shape, scores=[0.9, 0.2])
        self.assertEqual(int(labels.max()), 2)
        self.assertEqual(labels[4, 4], labels[8, 8])
        self.assertNotEqual(labels[0, 0], labels[4, 4])

    def test_fully_hidden_instances_are_dropped(self) -> None:
        shape = (10, 10)
        labels = masks_to_label_map(
            [square(shape, 0, 0, 8), square(shape, 2, 2, 3)], shape, scores=[0.9, 0.1]
        )
        self.assertEqual(int(labels.max()), 1)

    def test_suppress_duplicate_masks_keeps_highest_score(self) -> None:
        shape = (20, 20)
        masks = [square(shape, 0, 0, 10), square(shape, 0, 1, 10), square(shape, 12, 12, 5)]
        kept, scores = suppress_duplicate_masks(masks, [0.6, 0.8, 0.7])
        self.assertEqual(scores, [0.8, 0.7])
        np.testing.assert_array_equal(kept[0], masks[1])


class ImageTypeTests(unittest.TestCase):
    def test_classifies_by_color_statistics(self) -> None:
        dark = np.full((8, 8, 3), 10, dtype=np.uint8)
        bright = np.full((8, 8, 3), 220, dtype=np.uint8)
        stained = np.zeros((8, 8, 3), dtype=np.uint8)
        stained[...] = (200, 120, 190)
        self.assertEqual(classify_image_type(dark), ImageType.FLUORESCENT)
        self.assertEqual(classify_image_type(bright), ImageType.BRIGHTFIELD)
        self.assertEqual(classify_image_type(stained), ImageType.HISTOLOGY)


if __name__ == "__main__":
    unittest.main()
