import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getMaskDisplayNumber } from "../maskLabels";
import type { InteractionMode, MaskCandidate, PromptBox, PromptPoint } from "../types";

interface SegmentationCanvasProps {
  imageUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  mode: InteractionMode;
  points: PromptPoint[];
  boxes: PromptBox[];
  masks: MaskCandidate[];
  selectedCandidateId: string | null;
  overlayOpacity: number;
  disabled: boolean;
  uploadDisabled: boolean;
  onSelectMask: (candidateId: string) => void;
  onUploadImage: (file: File) => Promise<void>;
  onAddPoint: (point: PromptPoint) => void;
  onRemovePoint: (pointId: string) => void;
  onSingleClickSegment: (point: PromptPoint) => void;
  onAddBox: (box: PromptBox) => void;
  onRemoveBox: (boxId: string) => void;
}

interface ViewState {
  x: number;
  y: number;
  scale: number;
}

interface ImagePoint {
  x: number;
  y: number;
}

interface BoxDragState {
  kind: "box";
  startImageX: number;
  startImageY: number;
}

interface PanDragState {
  kind: "pan";
  startClientX: number;
  startClientY: number;
  startViewX: number;
  startViewY: number;
}

type DragState = BoxDragState | PanDragState;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getMaskLabelStyles(color: number[]) {
  const [red, green, blue] = color;
  const luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255;
  return {
    backgroundColor: `rgba(${red}, ${green}, ${blue}, 0.58)`,
    color: luminance > 0.58 ? "#061018" : "#f8fbfd",
    borderColor: luminance > 0.58 ? "rgba(6, 16, 24, 0.22)" : "rgba(248, 251, 253, 0.28)",
  };
}

export function SegmentationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  mode,
  points,
  boxes,
  masks,
  selectedCandidateId,
  overlayOpacity,
  disabled,
  uploadDisabled,
  onSelectMask,
  onUploadImage,
  onAddPoint,
  onRemovePoint,
  onSingleClickSegment,
  onAddBox,
  onRemoveBox,
}: SegmentationCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pendingClickRef = useRef<{ clientX: number; clientY: number } | null>(null);
  const [view, setView] = useState<ViewState>({ x: 0, y: 0, scale: 1 });
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [draftBox, setDraftBox] = useState<PromptBox | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [spacePressed, setSpacePressed] = useState(false);

  const dataTransferHasFiles = (dataTransfer: DataTransfer | null) =>
    Array.from(dataTransfer?.types ?? []).includes("Files");

  const fitToView = useCallback(() => {
    const container = containerRef.current;
    if (!container || !imageWidth || !imageHeight) {
      return;
    }

    const bounds = container.getBoundingClientRect();
    const scale = Math.min(bounds.width / imageWidth, bounds.height / imageHeight, 1);
    setView({
      scale,
      x: (bounds.width - imageWidth * scale) / 2,
      y: (bounds.height - imageHeight * scale) / 2,
    });
  }, [imageHeight, imageWidth]);

  useEffect(() => {
    fitToView();
  }, [fitToView, imageUrl]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        setSpacePressed(true);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        setSpacePressed(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  const getImageCoordinates = useCallback(
    (clientX: number, clientY: number): ImagePoint | null => {
      const container = containerRef.current;
      if (!container || !imageWidth || !imageHeight) {
        return null;
      }

      const rect = container.getBoundingClientRect();
      const x = (clientX - rect.left - view.x) / view.scale;
      const y = (clientY - rect.top - view.y) / view.scale;

      return {
        x: clamp(x, 0, imageWidth),
        y: clamp(y, 0, imageHeight),
      };
    },
    [imageHeight, imageWidth, view.scale, view.x, view.y],
  );

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (!imageUrl) {
      return;
    }
    event.preventDefault();

    const container = containerRef.current;
    if (!container) {
      return;
    }

    const rect = container.getBoundingClientRect();
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    const pointX = (localX - view.x) / view.scale;
    const pointY = (localY - view.y) / view.scale;

    const nextScale = clamp(view.scale * (event.deltaY < 0 ? 1.1 : 0.92), 0.08, 12);
    setView({
      scale: nextScale,
      x: localX - pointX * nextScale,
      y: localY - pointY * nextScale,
    });
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!imageUrl || disabled) {
      return;
    }

    const container = containerRef.current;
    if (!container) {
      return;
    }

    if (event.button === 2) {
      pendingClickRef.current = null;
      return;
    }

    const isPanGesture = spacePressed || event.altKey || event.button === 1;
    if (isPanGesture) {
      event.preventDefault();
      container.setPointerCapture(event.pointerId);
      setDragState({
        kind: "pan",
        startClientX: event.clientX,
        startClientY: event.clientY,
        startViewX: view.x,
        startViewY: view.y,
      });
      pendingClickRef.current = null;
      return;
    }

    if (mode === "box") {
      const point = getImageCoordinates(event.clientX, event.clientY);
      if (!point) {
        return;
      }
      container.setPointerCapture(event.pointerId);
      setDragState({
        kind: "box",
        startImageX: point.x,
        startImageY: point.y,
      });
      setDraftBox({
        id: "draft-box",
        x0: point.x,
        y0: point.y,
        x1: point.x,
        y1: point.y,
        label: "positive",
      });
      return;
    }

    pendingClickRef.current = { clientX: event.clientX, clientY: event.clientY };
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!dragState) {
      return;
    }

    if (dragState.kind === "pan") {
      setView((current) => ({
        ...current,
        x: dragState.startViewX + event.clientX - dragState.startClientX,
        y: dragState.startViewY + event.clientY - dragState.startClientY,
      }));
      return;
    }

    const point = getImageCoordinates(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    setDraftBox({
      id: "draft-box",
      x0: Math.min(dragState.startImageX, point.x),
      y0: Math.min(dragState.startImageY, point.y),
      x1: Math.max(dragState.startImageX, point.x),
      y1: Math.max(dragState.startImageY, point.y),
      label: "positive",
    });
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const container = containerRef.current;
    if (container?.hasPointerCapture(event.pointerId)) {
      container.releasePointerCapture(event.pointerId);
    }

    if (event.button === 2) {
      pendingClickRef.current = null;
      return;
    }

    if (dragState?.kind === "pan") {
      setDragState(null);
      pendingClickRef.current = null;
      return;
    }

    if (dragState?.kind === "box") {
      if (draftBox && draftBox.x1 - draftBox.x0 > 4 && draftBox.y1 - draftBox.y0 > 4) {
        onAddBox({ ...draftBox, id: crypto.randomUUID() });
      }
      setDraftBox(null);
      setDragState(null);
      pendingClickRef.current = null;
      return;
    }

    const pendingClick = pendingClickRef.current;
    pendingClickRef.current = null;
    if (!pendingClick || disabled || !imageUrl) {
      return;
    }

    const dx = Math.abs(event.clientX - pendingClick.clientX);
    const dy = Math.abs(event.clientY - pendingClick.clientY);
    if (dx > 4 || dy > 4) {
      return;
    }

    const point = getImageCoordinates(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    if (mode === "positive_point" || mode === "negative_point") {
      onAddPoint({
        id: crypto.randomUUID(),
        x: point.x,
        y: point.y,
        label: mode === "positive_point" ? "positive" : "negative",
        source: mode,
      });
      return;
    }

    if (mode === "click") {
      onSingleClickSegment({
        id: crypto.randomUUID(),
        x: point.x,
        y: point.y,
        label: "positive",
        source: "click",
      });
    }
  };

  const cursorClass = useMemo(() => {
    if (mode === "box") {
      return "cursor-crosshair";
    }
    if (mode === "click" || mode === "positive_point" || mode === "negative_point") {
      return "cursor-target";
    }
    return "cursor-default";
  }, [mode]);

  const interceptCanvasPointerEvent = (event: React.PointerEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };

  const handleCanvasContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    pendingClickRef.current = null;
  };

  const handleCanvasMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button === 1) {
      event.preventDefault();
    }
  };

  const handlePromptContextMenu = (
    event: React.MouseEvent<SVGElement>,
    removePrompt: () => void,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    pendingClickRef.current = null;
    if (disabled) {
      return;
    }
    removePrompt();
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    if (!dataTransferHasFiles(event.dataTransfer)) {
      return;
    }

    event.preventDefault();
    if (!uploadDisabled) {
      setIsDragOver(true);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    if (!dataTransferHasFiles(event.dataTransfer)) {
      return;
    }

    event.preventDefault();
    event.dataTransfer.dropEffect = uploadDisabled ? "none" : "copy";
    if (!uploadDisabled) {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget;
    if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
      return;
    }
    setIsDragOver(false);
  };

  const handleDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    if (!dataTransferHasFiles(event.dataTransfer)) {
      return;
    }

    event.preventDefault();
    setIsDragOver(false);

    if (uploadDisabled) {
      return;
    }

    const file = event.dataTransfer.files?.[0];
    if (!file) {
      return;
    }

    await onUploadImage(file);
  };

  return (
    <section className="viewer-shell">
      <div
        ref={containerRef}
        className={`viewer-stage ${cursorClass} ${isDragOver ? "is-dragging" : ""}`.trim()}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onMouseDown={handleCanvasMouseDown}
        onWheel={handleWheel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onContextMenu={handleCanvasContextMenu}
      >
        {isDragOver ? (
          <div className="viewer-drop-overlay" aria-hidden="true">
            <span className="viewer-drop-overlay-title">Drop image to upload</span>
            <span className="viewer-drop-overlay-subtitle">
              Replace the current canvas image and clear active prompts.
            </span>
          </div>
        ) : null}
        {imageUrl ? (
          <>
            <div className="viewer-stage-hud viewer-stage-hud-top">
              <span className="viewer-scale-badge">{Math.round(view.scale * 100)}%</span>
              <span className="viewer-hint-badge">
                Wheel to zoom. Middle-drag, Space-drag, or Alt-drag to pan. Left-drag draws boxes.
              </span>
            </div>
            <button
              type="button"
              className="viewer-fit-button"
              onPointerDown={interceptCanvasPointerEvent}
              onPointerUp={interceptCanvasPointerEvent}
              onClick={(event) => {
                event.stopPropagation();
                fitToView();
              }}
            >
              Fit View
            </button>
          </>
        ) : null}
        {!imageUrl ? (
          <div className="viewer-empty-state">
            <h3>SAM3 interactive canvas</h3>
            <p>
              Upload an image to place prompts, compare mask candidates, and
              explore interactive segmentation locally.
            </p>
          </div>
        ) : (
          <div
            className="viewer-world"
            style={{
              width: imageWidth,
              height: imageHeight,
              transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
            }}
          >
            <img
              className="viewer-image"
              src={imageUrl}
              alt="Uploaded scene"
              draggable={false}
            />
            {masks.map((mask, index) => {
              const maskNumber = getMaskDisplayNumber(mask, index);
              return (
                <img
                  key={mask.candidate_id}
                  className={`viewer-mask ${selectedCandidateId === mask.candidate_id ? "is-selected" : ""
                    }`}
                  src={mask.image_data_url}
                  alt={`Mask ${maskNumber}`}
                  style={{ opacity: overlayOpacity }}
                  draggable={false}
                />
              );
            })}
            <svg className="viewer-overlay" viewBox={`0 0 ${imageWidth} ${imageHeight}`}>
              {boxes.map((box) => (
                <rect
                  key={box.id}
                  className="prompt-box"
                  x={box.x0}
                  y={box.y0}
                  width={box.x1 - box.x0}
                  height={box.y1 - box.y0}
                  onContextMenu={(event) => {
                    handlePromptContextMenu(event, () => onRemoveBox(box.id));
                  }}
                />
              ))}
              {draftBox ? (
                <rect
                  className="prompt-box draft"
                  x={draftBox.x0}
                  y={draftBox.y0}
                  width={draftBox.x1 - draftBox.x0}
                  height={draftBox.y1 - draftBox.y0}
                />
              ) : null}
              {points.map((point) => (
                <g
                  key={point.id}
                  transform={`translate(${point.x}, ${point.y})`}
                  onContextMenu={(event) => {
                    handlePromptContextMenu(event, () => onRemovePoint(point.id));
                  }}
                >
                  <circle
                    className={`prompt-point ${point.label === "positive" ? "positive" : "negative"}`}
                    r={9}
                  />
                  <line x1={-4} y1={0} x2={4} y2={0} className="prompt-line" />
                  {point.label === "positive" ? (
                    <line x1={0} y1={-4} x2={0} y2={4} className="prompt-line" />
                  ) : null}
                </g>
              ))}
            </svg>
            {masks.map((mask, index) => {
              const maskNumber = getMaskDisplayNumber(mask, index);
              if (!mask.centroid) {
                return null;
              }
              const [centerX, centerY] = mask.centroid;
              return (
                <button
                  key={`${mask.candidate_id}-button`}
                  type="button"
                  className={`mask-label-button ${selectedCandidateId === mask.candidate_id ? "is-active" : ""
                    }`}
                  style={{
                    left: centerX,
                    top: centerY,
                    transform: "translate(-50%, -50%)",
                    ...getMaskLabelStyles(mask.color_rgb),
                  }}
                  onClick={(event) => {
                    event.stopPropagation();
                    onSelectMask(mask.candidate_id);
                  }}
                >
                  {`Mask ${maskNumber}`}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
