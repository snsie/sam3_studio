import { useEffect, useState } from "react";

import type { ChallengeExample } from "../types";

interface ExamplesModalProps {
  challengeTitle: string;
  examples: ChallengeExample[];
  isOpen: boolean;
  onClose: () => void;
}

export function ExamplesModal({
  challengeTitle,
  examples,
  isOpen,
  onClose,
}: ExamplesModalProps) {
  const [areMasksVisible, setAreMasksVisible] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setAreMasksVisible(true);
    }
  }, [isOpen]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="examples-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`${challengeTitle} examples`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="examples-modal-header">
          <div>
            <span className="section-kicker">Examples</span>
            <h2>{challengeTitle}</h2>
          </div>
          <div className="examples-modal-actions">
            <button
              type="button"
              className="secondary-button compact-button"
              aria-pressed={areMasksVisible}
              onClick={() => setAreMasksVisible((current) => !current)}
            >
              {areMasksVisible ? "Hide Masks" : "Show Masks"}
            </button>
            <button type="button" className="secondary-button compact-button" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
        <div className="examples-grid">
          {examples.map((example) => (
            <figure key={example.image_id} className="example-card">
              <div className="example-stage">
                <img src={example.image_url} alt={example.title} className="example-image" />
                {areMasksVisible ? (
                  <img
                    src={example.overlay_data_url}
                    alt={`${example.title} ground truth overlay`}
                    className="example-overlay"
                  />
                ) : null}
              </div>
              <figcaption>{example.title}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </div>
  );
}