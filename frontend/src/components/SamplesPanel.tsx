interface SamplesPanelProps {
  samples: string[];
  currentFilename: string | null;
  disabled: boolean;
  isLoading: boolean;
  errorMessage: string | null;
  onSelectSample: (filename: string) => Promise<void>;
}

export function SamplesPanel({
  samples,
  currentFilename,
  disabled,
  isLoading,
  errorMessage,
  onSelectSample,
}: SamplesPanelProps) {
  return (
    <section className="panel-block">
      <div className="section-heading">
        <span className="section-kicker">Library</span>
        <h2>Samples</h2>
      </div>
      {isLoading ? <p className="field-note">Loading sample images...</p> : null}
      {errorMessage ? <div className="error-banner inline-error">{errorMessage}</div> : null}
      {!isLoading && !errorMessage && samples.length === 0 ? (
        <p className="field-note">No sample images found.</p>
      ) : null}
      {samples.length > 0 ? (
        <div className="sample-list" role="list">
          {samples.map((sample) => {
            const isActive = currentFilename === sample;
            return (
              <button
                key={sample}
                type="button"
                className={`sample-list-button ${isActive ? "is-active" : ""}`.trim()}
                disabled={disabled}
                onClick={async () => {
                  await onSelectSample(sample);
                }}
              >
                <span className="sample-list-title">{sample}</span>
                <span className="sample-list-subtitle">
                  {isActive ? "Loaded on canvas" : "Load sample image"}
                </span>
              </button>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}