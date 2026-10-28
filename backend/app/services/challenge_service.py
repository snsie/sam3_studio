from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.models.schemas import (
    ChallengeExample,
    ChallengeExampleMask,
    ChallengeExamplesResponse,
    ChallengeImageItem,
    ChallengeImageListResponse,
    ChallengeSummary,
    ChallengeType,
    UploadResponse,
)
from app.services.sam3_service import SAM3Service

IMAGE_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

MASK_COLORS: dict[ChallengeType, list[tuple[str, tuple[int, int, int]]]] = {
    ChallengeType.SINGLE_CLASS: [("Foreground", (45, 226, 207))],
    ChallengeType.MULTI_CLASS: [
        ("Channel 0", (45, 226, 207)),
        ("Channel 1", (245, 158, 11)),
        ("Channel 2", (251, 113, 133)),
        ("Channel 3", (96, 165, 250)),
        ("Channel 4", (52, 211, 153)),
        ("Channel 5", (244, 114, 182)),
        ("Channel 6", (163, 230, 53)),
        ("Channel 7", (192, 132, 252)),
        ("Channel 8", (248, 113, 113)),
        ("Channel 9", (250, 204, 21)),
    ],
}
MULTI_CLASS_LABEL_VALUE_COUNT = 6
MIN_MULTI_CLASS_MASK_AREA = 8

PAIRING_PREFIX_PATTERN = re.compile(
    r"^(?:images|image|masks|mask|labels|label)[_-]*", re.IGNORECASE
)
CHANNEL_SUFFIX_PATTERN = re.compile(r"[_-]channel(?P<channel>\d+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GroundTruthMaskRecord:
    mask_id: str
    label: str
    path: Path
    color_rgb: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ChallengeImageRecord:
    image_id: str
    index: int
    image_path: Path
    masks: tuple[GroundTruthMaskRecord, ...]

    @property
    def name(self) -> str:
        return self.image_path.name


@dataclass(frozen=True, slots=True)
class ChallengeDefinition:
    challenge_id: ChallengeType
    title: str
    description: str
    image_dir: Path
    mask_dir: Path


@dataclass(frozen=True, slots=True)
class ChallengeDatasetCacheEntry:
    signature: tuple[tuple[str, int], ...]
    records: tuple[ChallengeImageRecord, ...]


class ChallengeService:
    def __init__(self, settings: Settings) -> None:
        datasets_dir = settings.challenge_datasets_dir
        self._definitions: dict[ChallengeType, ChallengeDefinition] = {
            ChallengeType.SINGLE_CLASS: ChallengeDefinition(
                challenge_id=ChallengeType.SINGLE_CLASS,
                title="BBBC038 Nuclei Segmentation",
                description=(
                    "Broad Bioimage Benchmark Collection nuclei images, adapted "
                    "here as a single foreground-vs-background mask challenge."
                ),
                image_dir=datasets_dir / "single_class" / "images",
                mask_dir=datasets_dir / "single_class" / "masks",
            ),
            ChallengeType.MULTI_CLASS: ChallengeDefinition(
                challenge_id=ChallengeType.MULTI_CLASS,
                title="CoNIC Colon Nuclei Classification",
                description=(
                    "CoNIC colon histology patches, adapted here as a six-class "
                    "mask-label challenge."
                ),
                image_dir=datasets_dir / "multi_class" / "images",
                mask_dir=datasets_dir / "multi_class" / "labels",
            ),
        }
        self._image_records_cache: dict[ChallengeType, ChallengeDatasetCacheEntry] = {}

    def list_challenges(self) -> list[ChallengeSummary]:
        return [self._build_summary(challenge_id) for challenge_id in ChallengeType]

    def list_images(self, challenge_id: ChallengeType) -> ChallengeImageListResponse:
        records = self.get_image_records(challenge_id)
        return ChallengeImageListResponse(
            challenge=self._build_summary(challenge_id),
            images=[
                ChallengeImageItem(
                    image_id=record.image_id,
                    name=record.name,
                    index=record.index,
                    mask_count=len(record.masks),
                )
                for record in records
            ],
        )

    def get_examples(
        self, challenge_id: ChallengeType, limit: int = 4
    ) -> ChallengeExamplesResponse:
        examples: list[ChallengeExample] = []
        for record in self.get_image_records(challenge_id)[:limit]:
            examples.append(
                ChallengeExample(
                    image_id=record.image_id,
                    title=f"Example {record.index}",
                    image_url=self.build_image_url(challenge_id, record.image_id),
                    overlay_data_url=self._build_overlay_data_url(record),
                    masks=[
                        ChallengeExampleMask(
                            mask_id=mask.mask_id,
                            label=mask.label,
                            color_rgb=list(mask.color_rgb),
                            image_url=self.build_ground_truth_mask_url(
                                challenge_id,
                                record.image_id,
                                mask.mask_id,
                            ),
                        )
                        for mask in record.masks
                    ],
                )
            )

        return ChallengeExamplesResponse(
            challenge=self._build_summary(challenge_id),
            examples=examples,
        )

    def create_session_for_image(
        self,
        challenge_id: ChallengeType,
        image_id: str,
        sam3_service: SAM3Service,
    ) -> UploadResponse:
        record = self.get_image_record(challenge_id, image_id)
        suffix = record.image_path.suffix.lower()
        content_type = IMAGE_CONTENT_TYPES.get(suffix)
        if not content_type:
            raise ValueError(
                f"Unsupported challenge image type: {record.image_path.name}"
            )
        # Challenge images are fixed assets, so key the session to the file itself: revisiting
        # an image then reuses its session and the worker's cached embeddings instead of
        # writing another copy of the image to disk. The stat fields invalidate the key if the
        # dataset file is replaced.
        stat = record.image_path.stat()
        source_key = (
            f"{challenge_id.value}:{record.image_id}:{stat.st_mtime_ns}:{stat.st_size}"
        )
        session = sam3_service.create_session(
            filename=record.image_path.name,
            content_type=content_type,
            content=record.image_path.read_bytes(),
            challenge_id=challenge_id,
            source_key=source_key,
        )
        return session.model_copy(
            update={
                "ground_truth_overlay_data_url": self._build_overlay_data_url(record)
            }
        )

    def get_image_record(
        self, challenge_id: ChallengeType, image_id: str
    ) -> ChallengeImageRecord:
        for record in self.get_image_records(challenge_id):
            if record.image_id == image_id:
                return record
        raise FileNotFoundError(f"Challenge image not found: {image_id}")

    def get_image_records(
        self, challenge_id: ChallengeType
    ) -> list[ChallengeImageRecord]:
        definition = self._get_definition(challenge_id)
        signature = self._build_dataset_signature(definition)
        cached = self._image_records_cache.get(challenge_id)
        if cached and cached.signature == signature:
            return list(cached.records)

        records = self._load_image_records(challenge_id)
        self._image_records_cache[challenge_id] = ChallengeDatasetCacheEntry(
            signature=signature,
            records=records,
        )
        return list(records)

    def get_image_path(self, challenge_id: ChallengeType, image_id: str) -> Path:
        return self.get_image_record(challenge_id, image_id).image_path

    def get_image_shape(
        self, challenge_id: ChallengeType, image_id: str
    ) -> tuple[int, int]:
        """Returns the challenge image's (height, width), the grid masks are scored on."""
        with Image.open(self.get_image_path(challenge_id, image_id)) as image:
            return image.height, image.width

    def get_mask_path(
        self, challenge_id: ChallengeType, image_id: str, mask_id: str
    ) -> Path:
        record = self.get_image_record(challenge_id, image_id)
        for mask in record.masks:
            if mask.mask_id == mask_id:
                return mask.path
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_id}")

    def load_ground_truth_masks(
        self,
        challenge_id: ChallengeType,
        image_id: str,
    ) -> list[np.ndarray]:
        record = self.get_image_record(challenge_id, image_id)
        if (
            challenge_id == ChallengeType.MULTI_CLASS
            and len(record.masks) == 1
            and record.masks[0].mask_id == "label-map"
        ):
            return self._extract_multi_class_masks(record.masks[0].path)
        return [self._read_mask(mask.path) for mask in record.masks]

    def build_image_url(self, challenge_id: ChallengeType, image_id: str) -> str:
        return f"/api/challenges/{challenge_id}/images/{image_id}"

    def build_ground_truth_mask_url(
        self, challenge_id: ChallengeType, image_id: str, mask_id: str
    ) -> str:
        return f"/api/challenges/{challenge_id}/ground-truth/{image_id}/{mask_id}"

    def _build_summary(self, challenge_id: ChallengeType) -> ChallengeSummary:
        definition = self._get_definition(challenge_id)
        image_count = len(self.get_image_records(challenge_id))
        return ChallengeSummary(
            challenge_id=challenge_id,
            title=definition.title,
            description=definition.description,
            image_count=image_count,
            example_count=min(4, image_count),
        )

    def _get_definition(self, challenge_id: ChallengeType) -> ChallengeDefinition:
        try:
            return self._definitions[challenge_id]
        except KeyError as exc:
            raise ValueError(f"Unsupported challenge_id: {challenge_id}") from exc

    def _load_image_records(
        self, challenge_id: ChallengeType
    ) -> tuple[ChallengeImageRecord, ...]:
        definition = self._get_definition(challenge_id)
        if not definition.image_dir.exists() or not definition.mask_dir.exists():
            return tuple()

        image_paths = sorted(
            self._iter_supported_files(definition.image_dir), key=lambda path: path.name
        )
        if challenge_id == ChallengeType.SINGLE_CLASS:
            return tuple(self._pair_single_class(definition, image_paths))
        return tuple(self._pair_multi_class(definition, image_paths))

    def _build_dataset_signature(
        self, definition: ChallengeDefinition
    ) -> tuple[tuple[str, int], ...]:
        return (
            ("images", self._get_directory_mtime(definition.image_dir)),
            ("masks", self._get_directory_mtime(definition.mask_dir)),
        )

    def _get_directory_mtime(self, directory: Path) -> int:
        if not directory.exists():
            return -1
        return directory.stat().st_mtime_ns

    def _pair_single_class(
        self,
        definition: ChallengeDefinition,
        image_paths: Iterable[Path],
    ) -> list[ChallengeImageRecord]:
        masks_by_key = {
            self._normalize_pairing_key(mask_path.stem): mask_path
            for mask_path in self._iter_supported_files(definition.mask_dir)
        }

        records: list[ChallengeImageRecord] = []
        label, color = MASK_COLORS[ChallengeType.SINGLE_CLASS][0]
        for index, image_path in enumerate(image_paths, start=1):
            key = self._normalize_pairing_key(image_path.stem)
            mask_path = masks_by_key.get(key)
            if mask_path is None:
                continue
            records.append(
                ChallengeImageRecord(
                    image_id=key,
                    index=index,
                    image_path=image_path,
                    masks=(
                        GroundTruthMaskRecord(
                            mask_id="foreground",
                            label=label,
                            path=mask_path,
                            color_rgb=color,
                        ),
                    ),
                )
            )
        return records

    def _pair_multi_class(
        self,
        definition: ChallengeDefinition,
        image_paths: Iterable[Path],
    ) -> list[ChallengeImageRecord]:
        mask_paths = list(self._iter_supported_files(definition.mask_dir))
        if not any(
            self._extract_channel_id(mask_path.stem) for mask_path in mask_paths
        ):
            return self._pair_multi_class_label_maps(image_paths, mask_paths)

        grouped_masks: dict[str, list[GroundTruthMaskRecord]] = {}
        palette = MASK_COLORS[ChallengeType.MULTI_CLASS]

        for mask_path in mask_paths:
            channel_id = self._extract_channel_id(mask_path.stem)
            if channel_id is None:
                continue
            key = self._normalize_pairing_key(mask_path.stem)
            channel_number = int(channel_id.replace("channel", ""))
            palette_index = channel_number % len(palette)
            label, color = palette[palette_index]
            grouped_masks.setdefault(key, []).append(
                GroundTruthMaskRecord(
                    mask_id=channel_id,
                    label=label,
                    path=mask_path,
                    color_rgb=color,
                )
            )

        records: list[ChallengeImageRecord] = []
        for index, image_path in enumerate(image_paths, start=1):
            key = self._normalize_pairing_key(image_path.stem)
            masks = grouped_masks.get(key)
            if not masks:
                continue
            ordered_masks = tuple(
                sorted(
                    masks,
                    key=lambda mask: int(mask.mask_id.replace("channel", "")),
                )
            )
            records.append(
                ChallengeImageRecord(
                    image_id=key,
                    index=index,
                    image_path=image_path,
                    masks=ordered_masks,
                )
            )
        return records

    def _pair_multi_class_label_maps(
        self,
        image_paths: Iterable[Path],
        mask_paths: Iterable[Path],
    ) -> list[ChallengeImageRecord]:
        masks_by_key = {
            self._normalize_pairing_key(mask_path.stem): mask_path
            for mask_path in mask_paths
        }
        _, color = MASK_COLORS[ChallengeType.MULTI_CLASS][0]

        records: list[ChallengeImageRecord] = []
        for index, image_path in enumerate(image_paths, start=1):
            key = self._normalize_pairing_key(image_path.stem)
            mask_path = masks_by_key.get(key)
            if mask_path is None:
                continue
            records.append(
                ChallengeImageRecord(
                    image_id=key,
                    index=index,
                    image_path=image_path,
                    masks=(
                        GroundTruthMaskRecord(
                            mask_id="label-map",
                            label="Instance Labels",
                            path=mask_path,
                            color_rgb=color,
                        ),
                    ),
                )
            )
        return records

    def _build_overlay_data_url(self, record: ChallengeImageRecord) -> str:
        if len(record.masks) == 1 and record.masks[0].mask_id == "label-map":
            class_masks = self._extract_multi_class_masks(record.masks[0].path)
            if not class_masks:
                first_mask = self._read_mask(record.masks[0].path)
                class_masks = [first_mask]
            overlay = np.zeros(
                (class_masks[0].shape[0], class_masks[0].shape[1], 4),
                dtype=np.uint8,
            )
            palette = [color for _, color in MASK_COLORS[ChallengeType.MULTI_CLASS]]
            for index, mask in enumerate(class_masks):
                color = palette[index % len(palette)]
                overlay[mask] = (*color, 255)
            buffer = BytesIO()
            Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}"

        first_mask = self._read_mask(record.masks[0].path)
        overlay = np.zeros(
            (first_mask.shape[0], first_mask.shape[1], 4), dtype=np.uint8
        )
        for mask_record in record.masks:
            mask = self._read_mask(mask_record.path)
            overlay[mask] = (*mask_record.color_rgb, 255)

        buffer = BytesIO()
        Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _iter_supported_files(self, directory: Path) -> Iterable[Path]:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_CONTENT_TYPES:
                yield path

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        return np.asarray(Image.open(mask_path).convert("L")) > 0

    def _extract_multi_class_masks(self, mask_path: Path) -> list[np.ndarray]:
        # The normalized workshop dataset keeps only the label channel, and those class
        # IDs were exported as lossy JPEGs. Quantizing back onto the original 0..6 label
        # levels preserves one mask per class so the same class shares one mask ID.
        label_map = np.asarray(Image.open(mask_path).convert("L"))
        quantized = np.rint(
            label_map.astype(np.float32) * MULTI_CLASS_LABEL_VALUE_COUNT / 255.0
        ).astype(np.uint8)
        masks: list[np.ndarray] = []
        for label_value in range(1, MULTI_CLASS_LABEL_VALUE_COUNT + 1):
            mask = quantized == label_value
            if int(mask.sum()) < MIN_MULTI_CLASS_MASK_AREA:
                continue
            masks.append(mask)
        return masks

    def _normalize_pairing_key(self, stem: str) -> str:
        normalized = PAIRING_PREFIX_PATTERN.sub("", stem.lower())
        normalized = CHANNEL_SUFFIX_PATTERN.sub("", normalized)
        return normalized

    def _extract_channel_id(self, stem: str) -> str | None:
        match = CHANNEL_SUFFIX_PATTERN.search(stem)
        if match is None:
            return None
        return f"channel{match.group('channel')}"
