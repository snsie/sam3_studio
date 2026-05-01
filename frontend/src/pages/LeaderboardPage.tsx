import { useEffect, useState } from "react";

import { listLeaderboardEntries } from "../api/client";
import type { LeaderboardHeaderState } from "../headerContext";
import type { ChallengeId, ChallengeScoreSummary } from "../types";

type SortKey =
  | "rank"
  | "user_name"
  | "images_labeled"
  | "percent_completed"
  | "labeled_mean_iou"
  | "labeled_mean_dice"
  | "overall_score"
  | "submitted_at";

type SortDirection = "asc" | "desc";

interface SortState {
  key: SortKey;
  direction: SortDirection;
}

const DEFAULT_SORT_STATE: SortState = {
  key: "rank",
  direction: "asc",
};

const INITIAL_SORT_DIRECTIONS: Record<SortKey, SortDirection> = {
  rank: "asc",
  user_name: "asc",
  images_labeled: "desc",
  percent_completed: "desc",
  labeled_mean_iou: "desc",
  labeled_mean_dice: "desc",
  overall_score: "desc",
  submitted_at: "desc",
};

const LEADERBOARD_COLUMNS: Array<{ key: SortKey; label: string }> = [
  { key: "rank", label: "Rank" },
  { key: "user_name", label: "Name" },
  { key: "images_labeled", label: "Images Labeled" },
  { key: "percent_completed", label: "Completed" },
  { key: "labeled_mean_iou", label: "Mean IoU (Labeled)" },
  { key: "labeled_mean_dice", label: "Mean Dice (Labeled)" },
  { key: "overall_score", label: "Overall Score" },
  { key: "submitted_at", label: "Submitted" },
];

interface LeaderboardPageProps {
  isActive: boolean;
  onHeaderStateChange?: (state: LeaderboardHeaderState | null) => void;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function compareNumber(left: number, right: number, direction: SortDirection) {
  return direction === "asc" ? left - right : right - left;
}

function compareNullableNumber(
  left: number | null,
  right: number | null,
  direction: SortDirection,
) {
  if (left === null && right === null) {
    return 0;
  }

  if (left === null) {
    return 1;
  }

  if (right === null) {
    return -1;
  }

  return compareNumber(left, right, direction);
}

function compareText(left: string, right: string, direction: SortDirection) {
  const result = left.localeCompare(right, undefined, { sensitivity: "base", numeric: true });
  return direction === "asc" ? result : -result;
}

function compareDate(left: string, right: string, direction: SortDirection) {
  return compareNumber(Date.parse(left), Date.parse(right), direction);
}

function compareEntries(
  left: ChallengeScoreSummary,
  right: ChallengeScoreSummary,
  sortState: SortState,
) {
  let result = 0;

  switch (sortState.key) {
    case "rank":
      result = compareNullableNumber(left.rank, right.rank, sortState.direction);
      break;
    case "user_name":
      result = compareText(left.user_name, right.user_name, sortState.direction);
      break;
    case "images_labeled":
      result = compareNumber(left.images_labeled, right.images_labeled, sortState.direction);
      break;
    case "percent_completed":
      result = compareNumber(left.percent_completed, right.percent_completed, sortState.direction);
      break;
    case "labeled_mean_iou":
      result = compareNumber(left.labeled_mean_iou, right.labeled_mean_iou, sortState.direction);
      break;
    case "labeled_mean_dice":
      result = compareNumber(left.labeled_mean_dice, right.labeled_mean_dice, sortState.direction);
      break;
    case "overall_score":
      result = compareNumber(left.overall_score, right.overall_score, sortState.direction);
      break;
    case "submitted_at":
      result = compareDate(left.submitted_at, right.submitted_at, sortState.direction);
      break;
  }

  if (result !== 0) {
    return result;
  }

  const rankTieBreak = compareNullableNumber(left.rank, right.rank, "asc");
  if (rankTieBreak !== 0) {
    return rankTieBreak;
  }

  return compareText(left.user_name, right.user_name, "asc");
}

export function LeaderboardPage({ isActive, onHeaderStateChange }: LeaderboardPageProps) {
  const [activeChallengeId, setActiveChallengeId] = useState<ChallengeId>("single-class");
  const [entries, setEntries] = useState<ChallengeScoreSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sortState, setSortState] = useState<SortState>(DEFAULT_SORT_STATE);
  const topEntry = entries[0] ?? null;
  const sortedEntries = [...entries].sort((left, right) => compareEntries(left, right, sortState));

  const handleSortChange = (key: SortKey) => {
    setSortState((current) => {
      if (current.key === key) {
        return {
          key,
          direction: current.direction === "asc" ? "desc" : "asc",
        };
      }

      return {
        key,
        direction: INITIAL_SORT_DIRECTIONS[key],
      };
    });
  };

  const getAriaSort = (key: SortKey) => {
    if (sortState.key !== key) {
      return "none";
    }

    return sortState.direction === "asc" ? "ascending" : "descending";
  };

  useEffect(() => {
    if (!isActive) {
      return;
    }

    let isCurrent = true;
    const loadEntries = async () => {
      try {
        setIsLoading(true);
        const response = await listLeaderboardEntries(activeChallengeId);
        if (!isCurrent) {
          return;
        }
        setEntries(response.entries);
        setErrorMessage(null);
      } catch (error) {
        if (!isCurrent) {
          return;
        }
        setErrorMessage(error instanceof Error ? error.message : "Failed to load leaderboard.");
      } finally {
        if (isCurrent) {
          setIsLoading(false);
        }
      }
    };

    void loadEntries();
    return () => {
      isCurrent = false;
    };
  }, [activeChallengeId, isActive]);

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    onHeaderStateChange({
      activeChallengeId,
      isLoading,
      errorMessage,
      topEntry,
      onChallengeChange: setActiveChallengeId,
    });
  }, [activeChallengeId, errorMessage, isLoading, onHeaderStateChange, topEntry]);

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    return () => {
      onHeaderStateChange(null);
    };
  }, [onHeaderStateChange]);

  return (
    <section className="page-content-stack">
      {isLoading ? <div className="panel-block">Loading leaderboard...</div> : null}
      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

      {!isLoading && !errorMessage ? (
        <section className="panel-block leaderboard-table-card">
          {entries.length === 0 ? (
            <p className="field-note">No submissions have been recorded for this challenge yet.</p>
          ) : (
            <div className="leaderboard-table-wrapper">
              <table className="leaderboard-table">
                <thead>
                  <tr>
                    {LEADERBOARD_COLUMNS.map((column) => {
                      const isActive = sortState.key === column.key;
                      return (
                        <th key={column.key} scope="col" aria-sort={getAriaSort(column.key)}>
                          <button
                            type="button"
                            className={`leaderboard-sort-button${isActive ? " is-active" : ""}${isActive && sortState.direction === "desc" ? " is-desc" : ""
                              }`}
                            onClick={() => handleSortChange(column.key)}
                          >
                            <span>{column.label}</span>
                          </button>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {sortedEntries.map((entry) => (
                    <tr key={`${entry.challenge_id}-${entry.user_name}`}>
                      <td>{entry.rank ?? "-"}</td>
                      <td>{entry.user_name}</td>
                      <td>{`${entry.images_labeled} / ${entry.dataset_size}`}</td>
                      <td>{formatPercent(entry.percent_completed)}</td>
                      <td>{formatPercent(entry.labeled_mean_iou)}</td>
                      <td>{formatPercent(entry.labeled_mean_dice)}</td>
                      <td>{entry.overall_score.toFixed(1)}</td>
                      <td>{new Date(entry.submitted_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      ) : null}
    </section>
  );
}