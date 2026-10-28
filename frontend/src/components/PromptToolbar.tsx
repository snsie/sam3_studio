import type { BrushMode, InteractionMode } from "../types";

const MODE_OPTIONS: Array<{
  mode: InteractionMode;
  title: string;
  detail: string;
}> = [
  { mode: "click", title: "Click", detail: "Instant positive click-to-mask" },
  {
    mode: "positive_point",
    title: "+ Point",
    detail: "Accumulate positive point prompts",
  },
  {
    mode: "negative_point",
    title: "- Point",
    detail: "Exclude regions with negative points",
  },
  { mode: "box", title: "Box", detail: "Drag a positive bounding box" },
];

interface PromptToolbarProps {
  disabled: boolean;
  mode: InteractionMode;
  availableModes?: InteractionMode[];
  brushMode?: BrushMode;
  brushRadius?: number;
  brushDisabled?: boolean;
  onModeChange: (mode: InteractionMode) => void;
  onBrushModeChange?: (mode: BrushMode) => void;
  onBrushRadiusChange?: (value: number) => void;
}

export function PromptToolbar({
  disabled,
  mode,
  availableModes,
  brushMode,
  brushRadius,
  brushDisabled = false,
  onModeChange,
  onBrushModeChange,
  onBrushRadiusChange,
}: PromptToolbarProps) {
  const visibleOptions = availableModes
    ? MODE_OPTIONS.filter((option) => availableModes.includes(option.mode))
    : MODE_OPTIONS;
  const showBrushControls =
    brushMode !== undefined &&
    brushRadius !== undefined &&
    onBrushModeChange !== undefined &&
    onBrushRadiusChange !== undefined;

  return (
    <div className="prompt-toolbar" role="toolbar" aria-label="Interaction modes">
      {visibleOptions.map((option) => (
        <button
          key={option.mode}
          type="button"
          className={`prompt-toolbar-button ${option.mode === mode ? "is-active" : ""}`}
          title={option.detail}
          disabled={disabled}
          onClick={() => onModeChange(option.mode)}
        >
          {option.title}
        </button>
      ))}
      {showBrushControls ? (
        <div className="prompt-toolbar-brush-group">
          <button
            type="button"
            className={`prompt-toolbar-button ${brushMode === "draw" ? "is-active" : ""}`}
            title="Paint pixels into the active mask"
            disabled={brushDisabled}
            onClick={() => onBrushModeChange("draw")}
          >
            Draw Mask
          </button>
          <button
            type="button"
            className={`prompt-toolbar-button ${brushMode === "erase" ? "is-active" : ""}`}
            title="Erase pixels from the active mask"
            disabled={brushDisabled}
            onClick={() => onBrushModeChange("erase")}
          >
            Erase Mask
          </button>
          <label className="prompt-toolbar-brush-size" htmlFor="challenge-brush-size-range">
            <span className="prompt-toolbar-brush-size-value">{brushRadius}px</span>
            <input
              id="challenge-brush-size-range"
              className="range-input prompt-toolbar-range-input"
              type="range"
              min={2}
              max={48}
              step={1}
              value={brushRadius}
              disabled={brushDisabled}
              aria-label="Brush size"
              onChange={(event) => onBrushRadiusChange(Number(event.target.value))}
            />
          </label>
        </div>
      ) : null}
    </div>
  );
}
