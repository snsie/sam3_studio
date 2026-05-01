import { useEffect, useState } from "react";

import { listLeaderboardEntries } from "../api/client";
import type { LeaderboardHeaderState } from "../headerContext";
import type { ChallengeId, ChallengeScoreSummary } from "../types";

interface LeaderboardPageProps {
  onHeaderStateChange?: (state: LeaderboardHeaderState | null) => void;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function LeaderboardPage({ onHeaderStateChange }: LeaderboardPageProps) {
  const [activeChallengeId, setActiveChallengeId] = useState<ChallengeId>("single-class");
  const [entries, setEntries] = useState<ChallengeScoreSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const topEntry = entries[0] ?? null;

  useEffect(() => {
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
  }, [activeChallengeId]);

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
                    <th>Rank</th>
                    <th>Name</th>
                    <th>Images Labeled</th>
                    <th>Completed</th>
                    <th>Mean IoU (Labeled)</th>
                    <th>Mean Dice (Labeled)</th>
                    <th>Overall Score</th>
                    <th>Submitted</th>
                  </tr>
                </thead>
                <tbody>
                  {entries.map((entry) => (
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