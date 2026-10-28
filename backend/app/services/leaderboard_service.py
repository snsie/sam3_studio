from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.models.schemas import ChallengeScoreSummary, ChallengeType


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DuplicateSubmissionError(ValueError):
    pass


class LeaderboardService:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = Lock()
        self._initialize()

    def add_entry(self, summary: ChallengeScoreSummary) -> ChallengeScoreSummary:
        normalized_name = summary.user_name.strip().casefold()
        payload = summary.model_dump()
        submitted_at = summary.submitted_at.isoformat()

        with self._lock, closing(
            sqlite3.connect(self._db_path, check_same_thread=False)
        ) as connection, connection:
            try:
                connection.execute(
                    """
                    INSERT INTO leaderboard_entries (
                        challenge_id,
                        user_name,
                        normalized_name,
                        dataset_size,
                        images_labeled,
                        percent_completed,
                        mean_iou,
                        mean_dice,
                        pixel_accuracy,
                        labeled_mean_iou,
                        labeled_mean_dice,
                        labeled_pixel_accuracy,
                        overall_score,
                        submitted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["challenge_id"],
                        payload["user_name"],
                        normalized_name,
                        payload["dataset_size"],
                        payload["images_labeled"],
                        payload["percent_completed"],
                        payload["mean_iou"],
                        payload["mean_dice"],
                        payload["pixel_accuracy"],
                        payload["labeled_mean_iou"],
                        payload["labeled_mean_dice"],
                        payload["labeled_pixel_accuracy"],
                        payload["overall_score"],
                        submitted_at,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateSubmissionError(
                    f"A submission already exists for '{summary.user_name}' in {summary.challenge_id}."
                ) from exc

        return self.get_entry(summary.challenge_id, summary.user_name)

    def list_entries(
        self, challenge_id: ChallengeType | None = None
    ) -> list[ChallengeScoreSummary]:
        query = """
            SELECT
                challenge_id,
                user_name,
                dataset_size,
                images_labeled,
                percent_completed,
                mean_iou,
                mean_dice,
                pixel_accuracy,
                labeled_mean_iou,
                labeled_mean_dice,
                labeled_pixel_accuracy,
                overall_score,
                submitted_at
            FROM leaderboard_entries
        """
        params: tuple[str, ...] = ()
        if challenge_id is not None:
            query += " WHERE challenge_id = ?"
            params = (challenge_id,)
        query += " ORDER BY overall_score DESC, submitted_at ASC, user_name ASC"

        with self._lock, closing(
            sqlite3.connect(self._db_path, check_same_thread=False)
        ) as connection, connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, params).fetchall()

        return [self._row_to_summary(row, index + 1) for index, row in enumerate(rows)]

    def get_entry(
        self, challenge_id: ChallengeType, user_name: str
    ) -> ChallengeScoreSummary:
        entries = self.list_entries(challenge_id)
        normalized_name = user_name.strip().casefold()
        for entry in entries:
            if entry.user_name.strip().casefold() == normalized_name:
                return entry
        raise ValueError("Leaderboard entry not found.")

    def _initialize(self) -> None:
        with self._lock, closing(
            sqlite3.connect(self._db_path, check_same_thread=False)
        ) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS leaderboard_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    challenge_id TEXT NOT NULL,
                    user_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    images_labeled INTEGER NOT NULL,
                    percent_completed REAL NOT NULL,
                    mean_iou REAL NOT NULL,
                    mean_dice REAL NOT NULL,
                    pixel_accuracy REAL NOT NULL,
                    labeled_mean_iou REAL NOT NULL,
                    labeled_mean_dice REAL NOT NULL,
                    labeled_pixel_accuracy REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    submitted_at TEXT NOT NULL,
                    UNIQUE(challenge_id, normalized_name)
                )
                """)

    def _row_to_summary(self, row: sqlite3.Row, rank: int) -> ChallengeScoreSummary:
        return ChallengeScoreSummary(
            challenge_id=ChallengeType(row["challenge_id"]),
            user_name=row["user_name"],
            dataset_size=row["dataset_size"],
            images_labeled=row["images_labeled"],
            percent_completed=row["percent_completed"],
            mean_iou=row["mean_iou"],
            mean_dice=row["mean_dice"],
            pixel_accuracy=row["pixel_accuracy"],
            labeled_mean_iou=row["labeled_mean_iou"],
            labeled_mean_dice=row["labeled_mean_dice"],
            labeled_pixel_accuracy=row["labeled_pixel_accuracy"],
            overall_score=row["overall_score"],
            submitted_at=datetime.fromisoformat(row["submitted_at"]),
            rank=rank,
            warnings=[],
        )
