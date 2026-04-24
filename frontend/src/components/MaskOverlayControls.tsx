interface MaskOverlayControlsProps {
  disabled: boolean;
  hasMask: boolean;
  maskVisible: boolean;
  overlayOpacity: number;
  onMaskVisibleChange: (visible: boolean) => void;
  onOverlayOpacityChange: (value: number) => void;
}

export function MaskOverlayControls({
  disabled,
  hasMask,
  maskVisible,
  overlayOpacity,
  onMaskVisibleChange,
  onOverlayOpacityChange,
}: MaskOverlayControlsProps) {
  return (
    <section className="panel-block">
      <div className="section-heading">
        <span className="section-kicker">Overlay</span>
        <h2>Mask Controls</h2>
      </div>
      <label className="toggle-row">
        <span>Show active mask</span>
        <input
          type="checkbox"
          checked={maskVisible}
          disabled={disabled || !hasMask}
          onChange={(event) => onMaskVisibleChange(event.target.checked)}
        />
      </label>
      <label className="field-label" htmlFor="opacity-range">
        Overlay opacity
      </label>
      <input
        id="opacity-range"
        className="range-input"
        type="range"
        min={0.1}
        max={1}
        step={0.05}
        value={overlayOpacity}
        disabled={disabled || !hasMask}
        onChange={(event) => onOverlayOpacityChange(Number(event.target.value))}
      />
      <div className="range-readout">{Math.round(overlayOpacity * 100)}%</div>
    </section>
  );
}
