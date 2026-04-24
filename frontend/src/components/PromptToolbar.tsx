import type { InteractionMode } from "../types";

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
  onModeChange: (mode: InteractionMode) => void;
}

export function PromptToolbar({
  disabled,
  mode,
  onModeChange,
}: PromptToolbarProps) {
  return (
    <div className="prompt-toolbar" role="toolbar" aria-label="Interaction modes">
      {MODE_OPTIONS.map((option) => (
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
    </div>
  );
}
