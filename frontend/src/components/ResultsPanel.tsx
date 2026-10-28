import { getMaskDisplayNumber } from "../maskLabels";
import type { MaskCandidate, SegmentationResult } from "../types";

interface ResultsPanelProps {
  result: SegmentationResult | null;
  visibleCandidates: MaskCandidate[];
  listCandidates?: MaskCandidate[];
  selectedCandidateId: string | null;
  confidenceThreshold: number;
  overlayOpacity?: number;
  onSelectCandidate: (candidateId: string) => void;
  onConfidenceThresholdChange?: (value: number) => void;
  onOverlayOpacityChange?: (value: number) => void;
  onShowAllMasks: () => void;
  allMasksButtonLabel?: string;
  showConfidenceThresholdControl?: boolean;
  showOverlayOpacityControl?: boolean;
  visibleCandidatesLabel?: string;
  emptyThresholdMessage?: string;
}

export function ResultsPanel({
  result,
  visibleCandidates,
  listCandidates,
  selectedCandidateId,
  confidenceThreshold,
  overlayOpacity,
  onSelectCandidate,
  onConfidenceThresholdChange,
  onOverlayOpacityChange,
  onShowAllMasks,
  allMasksButtonLabel = "Show All Masks",
  showConfidenceThresholdControl = true,
  showOverlayOpacityControl = true,
  visibleCandidatesLabel = "above threshold",
  emptyThresholdMessage = "No masks are above the current confidence threshold.",
}: ResultsPanelProps) {
  const hasMask = visibleCandidates.length > 0;
  const displayedCandidates = listCandidates ?? result?.candidates ?? [];
  // Count what is actually listed: callers may pass a pre-filtered listCandidates, and
  // result is live state while the candidates can come from a deferred one.
  const totalCandidates = displayedCandidates.length;

  const handleCandidateSelect = (candidate: MaskCandidate) => {
    const isAboveThreshold =
      typeof candidate.score === "number" && candidate.score > confidenceThreshold;

    if (!isAboveThreshold) {
      window.alert(
        showConfidenceThresholdControl
          ? "Lower the confidence threshold to enable viewing that mask."
          : "This mask is below the required confidence threshold.",
      );
      return;
    }

    onSelectCandidate(candidate.candidate_id);
  };

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
            <span className="results-count">{totalCandidates} candidates</span>
            <span className="results-count">{visibleCandidates.length} {visibleCandidatesLabel}</span>
          </div>
          {showConfidenceThresholdControl || showOverlayOpacityControl ? (
            <div className="results-overlay-controls">
              {showConfidenceThresholdControl ? (
                <div className="results-slider-group">
                  <label className="field-label" htmlFor="results-confidence-range">
                    Confidence threshold
                  </label>
                  <input
                    id="results-confidence-range"
                    className="range-input"
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={confidenceThreshold}
                    disabled={totalCandidates === 0}
                    onChange={(event) => onConfidenceThresholdChange?.(Number(event.target.value))}
                  />
                  <div className="range-readout">{Math.round(confidenceThreshold * 100)}%</div>
                </div>
              ) : null}
              {showOverlayOpacityControl ? (
                <div className="results-slider-group">
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
                    value={overlayOpacity ?? 1}
                    disabled={!hasMask}
                    onChange={(event) => onOverlayOpacityChange?.(Number(event.target.value))}
                  />
                  <div className="range-readout">{Math.round((overlayOpacity ?? 1) * 100)}%</div>
                </div>
              ) : null}
            </div>
          ) : null}
          <button
            type="button"
            className={`show-all-button ${selectedCandidateId === null ? "is-active" : ""}`}
            onClick={onShowAllMasks}
          >
            {allMasksButtonLabel}
          </button>
          {result.warnings.length > 0 ? (
            <div className="warnings-card">
              {result.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          ) : null}
          {!hasMask && totalCandidates > 0 ? (
            <p className="field-note results-filter-note">
              {emptyThresholdMessage}
            </p>
          ) : null}
          <div className="mask-list">
            {displayedCandidates.map((candidate, index) => {
              const maskNumber = getMaskDisplayNumber(candidate, index);
              const score =
                typeof candidate.score === "number"
                  ? `${(candidate.score * 100).toFixed(1)}%`
                  : "n/a";
              const isAboveThreshold =
                typeof candidate.score === "number" && candidate.score > confidenceThreshold;

              return (
                <button
                  key={candidate.candidate_id}
                  type="button"
                  className={`mask-card ${candidate.candidate_id === selectedCandidateId ? "is-active" : ""
                    } ${!isAboveThreshold ? "is-below-threshold" : ""}`.trim()}
                  aria-disabled={!isAboveThreshold}
                  onClick={() => handleCandidateSelect(candidate)}
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
