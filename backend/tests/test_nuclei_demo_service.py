from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.models.schemas import (
    InteractionMode,
    JobStatus,
    JobStatusResponse,
    MaskCandidate,
    NucleiModelKey,
    SegmentationRequest,
    SegmentationResult,
)
from app.services.nuclei_demo_service import (
    NucleiDemoService,
    NucleiModelUnavailableError,
    decode_mask_data_url,
    rgba_to_data_url,
)
from app.services.sam3_service import SAM3Service

IMAGE_ID = "a" * 64
SHAPE = (32, 48)


def square(top: int, left: int, size: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=bool)
    mask[top : top + size, left : left + size] = True
    return mask


def mask_data_url(mask: np.ndarray) -> str:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = (33, 211, 197, 255)
    return rgba_to_data_url(rgba)


class FakeQueue:
    """Stands in for InferenceQueue and returns fixed text-grounding candidates."""

    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = masks
        self.requests: list[SegmentationRequest] = []

    async def submit_and_wait(self, payload: SegmentationRequest, *, timeout_seconds: float):
        self.requests.append(payload)
        now = datetime.now(timezone.utc)
        candidates = [
            MaskCandidate(
                candidate_id=f"mask-{index}",
                score=0.9 - index * 0.1,
                width=SHAPE[1],
                height=SHAPE[0],
                area=int(mask.sum()),
                image_data_url=mask_data_url(mask),
                color_rgb=[33, 211, 197],
            )
            for index, mask in enumerate(self.masks)
        ]
        return JobStatusResponse(
            job_id="job-1",
            session_id=payload.session_id,
            status=JobStatus.COMPLETE,
            progress=1.0,
            message="done",
            created_at=now,
            updated_at=now,
            result=SegmentationResult(
                session_id=payload.session_id,
                mode=InteractionMode.TEXT,
                source="text_grounding",
                text_prompt_supported=True,
                supports_point_refinement=False,
                supports_box_refinement=True,
                candidates=candidates,
            ),
        )


class NucleiDemoServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        dataset_dir = root / "bbbc038"
        self.test_dir = dataset_dir / "raw" / "stage1_test"
        self.models_dir = root / "models"
        image_dir = self.test_dir / IMAGE_ID / "images"
        masks_dir = self.test_dir / IMAGE_ID / "masks"
        image_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        self.models_dir.mkdir()

        rgba = np.zeros((*SHAPE, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        Image.fromarray(rgba).save(image_dir / f"{IMAGE_ID}.png")
        for index, mask in enumerate([square(2, 2, 8), square(16, 20, 10)]):
            Image.fromarray(mask.astype(np.uint8) * 255).save(masks_dir / f"{index:04d}.png")

        (dataset_dir / "split.json").write_text(
            json.dumps({"test": [IMAGE_ID], "image_types": {IMAGE_ID: "fluorescent"}})
        )
        (self.models_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "prompt": "blobs",
                    "models": {"sam3_zero_shot": {"aggregate": {"dsb_score": 0.25}}},
                    "per_image": {
                        IMAGE_ID: {
                            "image_type": "fluorescent",
                            "gt_count": 2,
                            "models": {"sam3_zero_shot": {"dsb_score": 0.5}},
                        }
                    },
                }
            )
        )

        self.settings = Settings(uploads_dir=root / "uploads")
        self.settings.uploads_dir.mkdir()
        # One correct nucleus, one false positive, one missed nucleus.
        self.queue = FakeQueue([square(2, 2, 8), square(0, 40, 6)])
        self.service = NucleiDemoService(
            self.settings,
            SAM3Service(self.settings),
            self.queue,
            models_dir=self.models_dir,
            test_dir=self.test_dir,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_list_images_reports_type_and_ground_truth_count(self) -> None:
        response = self.service.list_images()
        self.assertEqual(len(response.images), 1)
        item = response.images[0]
        self.assertEqual(item.image_type, "fluorescent")
        self.assertEqual(item.gt_count, 2)
        self.assertEqual((item.height, item.width), SHAPE)
        self.assertEqual(response.image_type_counts["fluorescent"], 1)
        availability = {option.model_key: option.available for option in response.models}
        self.assertTrue(availability[NucleiModelKey.ZERO_SHOT])
        self.assertFalse(availability[NucleiModelKey.TRAINED])

    def test_metrics_summary_omits_per_image_records(self) -> None:
        summary = self.service.get_metrics_summary()
        self.assertNotIn("per_image", summary)
        image_metrics = self.service.get_image_metrics(IMAGE_ID)
        self.assertEqual(image_metrics["models"]["sam3_zero_shot"]["dsb_score"], 0.5)

    def test_predict_scores_live_masks_against_ground_truth(self) -> None:
        response = asyncio.run(self.service.predict(IMAGE_ID, NucleiModelKey.ZERO_SHOT))

        self.assertEqual(self.queue.requests[0].text_prompt, "blobs")
        self.assertEqual(response.counts, {"tp": 1, "fp": 1, "fn": 1, "pred": 2, "gt": 2})
        self.assertAlmostEqual(response.metrics["dsb_score"], 1 / 3)
        statuses = sorted(instance.status for instance in response.predictions)
        self.assertEqual(statuses, ["fp", "tp"])
        self.assertEqual(
            sorted(instance.status for instance in response.ground_truth), ["fn", "matched"]
        )
        comparison = decode_mask_data_url(response.overlays.comparison)
        self.assertEqual(comparison.shape, SHAPE)
        self.assertTrue(comparison[20, 24])  # the missed nucleus is drawn

    def test_predict_rejects_models_without_checkpoints(self) -> None:
        with self.assertRaises(NucleiModelUnavailableError):
            asyncio.run(self.service.predict(IMAGE_ID, NucleiModelKey.TRAINED))

    def test_unknown_image_raises_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.service.get_image_path("missing")


if __name__ == "__main__":
    unittest.main()
