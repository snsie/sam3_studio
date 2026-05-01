import type { BrushMode } from "../types";

interface BrushControlsProps {
  brushMode: BrushMode;
  brushRadius: number;
  disabled: boolean;
  onBrushModeChange: (mode: BrushMode) => void;
  onBrushRadiusChange: (value: number) => void;
}

export function BrushControls({
  brushMode,
  brushRadius,
  disabled,
  onBrushModeChange,
  onBrushRadiusChange,
}: BrushControlsProps) {
  return (
    <section className="panel-block brush-controls-card">
      <div className="section-heading">
        <span className="section-kicker">Brush</span>
        <h2>Manual Editing</h2>
      </div>
      <div className="inline-toggle-group">
        <button
          type="button"
          className={`inline-toggle-button${brushMode === "off" ? " is-active" : ""}`}
          disabled={disabled}
          onClick={() => onBrushModeChange("off")}
        >
          Prompt Mode
        </button>
        <button
          type="button"
          className={`inline-toggle-button${brushMode === "draw" ? " is-active" : ""}`}
          disabled={disabled}
          onClick={() => onBrushModeChange("draw")}
        >
          Draw
        </button>
        <button
          type="button"
          className={`inline-toggle-button${brushMode === "erase" ? " is-active" : ""}`}
          disabled={disabled}
          onClick={() => onBrushModeChange("erase")}
        >
          Erase
        </button>
      </div>
      <label className="field-label" htmlFor="brush-radius-range">
        Brush radius
      </label>
      <input
        id="brush-radius-range"
        className="range-input"
        type="range"
        min={2}
        max={48}
        step={1}
        value={brushRadius}
        disabled={disabled}
        onChange={(event) => onBrushRadiusChange(Number(event.target.value))}
      />
      <div className="range-readout">{brushRadius}px</div>
    </section>
  );
}