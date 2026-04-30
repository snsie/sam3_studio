import { getMaskDisplayNumber } from "../maskLabels";
import type { SegmentationResult } from "../types";

interface ResultsPanelProps {
  result: SegmentationResult | null;
  selectedCandidateId: string | null;
  overlayOpacity: number;
  onSelectCandidate: (candidateId: string) => void;
  onOverlayOpacityChange: (value: number) => void;
  onShowAllMasks: () => void;
}

export function ResultsPanel({
  result,
  selectedCandidateId,
  overlayOpacity,
  onSelectCandidate,
  onOverlayOpacityChange,
  onShowAllMasks,
}: ResultsPanelProps) {
  const hasMask = Boolean(result && result.candidates.length > 0);

  return (
    <aside className="results-panel">
      <div className="results-header">
        <span className="section-kicker">Outputs</span>
        <h2>Mask Candidates</h2>
      </div>
      {!result ? (
        <div className="results-empty">
          <p>No segmentation results yet.</p>
          <span>
            When SAM3 returns multiple masks, you can inspect and switch between
            them here.
          </span>
        </div>
      ) : (
        <>
          <div className="results-meta">
            <span className="results-count">{result.candidates.length} candidates</span>
          </div>
          <div className="results-overlay-controls">
            <label className="field-label" htmlFor="results-opacity-range">
              Overlay opacity
            </label>
            <input
              id="results-opacity-range"
              className="range-input"
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={overlayOpacity}
              disabled={!hasMask}
              onChange={(event) => onOverlayOpacityChange(Number(event.target.value))}
            />
            <div className="range-readout">{Math.round(overlayOpacity * 100)}%</div>
          </div>
          <button
            type="button"
            className={`show-all-button ${selectedCandidateId === null ? "is-active" : ""}`}
            onClick={onShowAllMasks}
          >
            Show All Masks
          </button>
          {result.warnings.length > 0 ? (
            <div className="warnings-card">
              {result.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          <div className="mask-list">
            {result.candidates.map((candidate, index) => {
              const maskNumber = getMaskDisplayNumber(candidate, index);
              const score =
                typeof candidate.score === "number"
                  ? `${(candidate.score * 100).toFixed(1)}%`
                  : "n/a";

              return (
                <button
                  key={candidate.candidate_id}
                  type="button"
                  className={`mask-card ${
                    candidate.candidate_id === selectedCandidateId ? "is-active" : ""
                  }`}
                  onClick={() => onSelectCandidate(candidate.candidate_id)}
                >
                  <div className="mask-card-preview">
                    <img src={candidate.image_data_url} alt={candidate.candidate_id} />
                  </div>
                  <div className="mask-card-copy">
                    <span>{`Mask ${maskNumber}`}</span>
                    <span>Score {score}</span>
                    <span>Area {candidate.area.toLocaleString()} px</span>
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </aside>
  );
}
