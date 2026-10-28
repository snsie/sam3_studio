import type { ChallengeScoreSummary } from "../types";

interface ScorePanelProps {
  userName: string;
  storedImageCount: number;
  datasetSize: number;
  disabled: boolean;
  isSubmitting: boolean;
  errorMessage: string | null;
  result: ChallengeScoreSummary | null;
  onUserNameChange: (value: string) => void;
  onSubmit: () => void;
}

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function ScorePanel({
  userName,
  disabled,
  isSubmitting,
  errorMessage,
  result,
  onUserNameChange,
  onSubmit,
}: ScorePanelProps) {
  return (
    <section className="panel-block score-panel-card">
      <div className="section-heading">
        {/* <span className="section-kicker">Scoring</span> */}
        <h2>Score Dataset</h2>
      </div>
      {/* <label className="field-label" htmlFor="challenge-user-name">
        Your name
      </label> */}
      <div className="score-panel-row">
        <input
          id="challenge-user-name"
          className="text-field"
          type="text"
          value={userName}
          placeholder="Enter your leaderboard name"
          onChange={(event) => onUserNameChange(event.target.value)}
        />
        <button
          type="button"
          className="primary-button score-submit-button"
          disabled={disabled}
          onClick={onSubmit}
        >
          {isSubmitting ? "Scoring..." : "Score Dataset"}
        </button>
      </div>
      {/* <p className="field-note">{`${storedImageCount} stored / ${datasetSize} total images`}</p> */}
      {errorMessage ? <div className="error-banner inline-error">{errorMessage}</div> : null}
      {result ? (
        <div className="score-summary-card">
          <div className="score-summary-topline">
            <strong>{result.user_name}</strong>
            <span>{`Rank ${result.rank ?? "-"}`}</span>
          </div>
          <div className="score-summary-grid">
            <span>{`Completion ${formatPercent(result.percent_completed)}`}</span>
            <span>{`Mean IoU (Labeled) ${formatPercent(result.labeled_mean_iou)}`}</span>
            <span>{`Mean Dice (Labeled) ${formatPercent(result.labeled_mean_dice)}`}</span>
            <span>{`Pixel Accuracy ${formatPercent(result.pixel_accuracy)}`}</span>
            <span>{`Overall ${result.overall_score.toFixed(1)}`}</span>
          </div>
          {result.warnings.length > 0 ? (
            <div className="warnings-card compact-warnings-card">
              {result.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}