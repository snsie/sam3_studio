import type { EditableMaskLayer } from "../types";

interface MaskIdPanelProps {
  layers: EditableMaskLayer[];
  activeMaskId: string | null;
  disabled: boolean;
  onAddMask: () => void;
  onSelectMask: (maskId: string) => void;
  onRenameMask: (maskId: string, label: string) => void;
  onDeleteMask: (maskId: string) => void;
}

export function MaskIdPanel({
  layers,
  activeMaskId,
  disabled,
  onAddMask,
  onSelectMask,
  onRenameMask,
  onDeleteMask,
}: MaskIdPanelProps) {
  return (
    <section className="panel-block mask-id-panel">
      <div className="section-heading mask-id-panel-heading">
        <div>
          <span className="section-kicker">Mask IDs</span>
          <h2>Active Masks</h2>
        </div>
        <button
          type="button"
          className="secondary-button compact-button"
          disabled={disabled}
          onClick={onAddMask}
        >
          Add New Mask
        </button>
      </div>
      <div className="mask-id-list">
        {layers.map((layer) => {
          const isActive = layer.mask_id === activeMaskId;
          return (
            <div key={layer.mask_id} className={`mask-id-item${isActive ? " is-active" : ""}`}>
              <button
                type="button"
                className="mask-id-select"
                disabled={disabled}
                onClick={() => onSelectMask(layer.mask_id)}
              >
                <span
                  className="mask-color-chip"
                  style={{
                    backgroundColor: `rgb(${layer.color_rgb[0]}, ${layer.color_rgb[1]}, ${layer.color_rgb[2]})`,
                  }}
                />
              </button>
              <input
                className="mask-id-input"
                type="text"
                disabled={disabled}
                value={layer.label}
                onChange={(event) => onRenameMask(layer.mask_id, event.target.value)}
              />
              <button
                type="button"
                className="mask-delete-button"
                disabled={disabled}
                onClick={() => onDeleteMask(layer.mask_id)}
                aria-label={`Delete ${layer.label}`}
              >
                ×
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}