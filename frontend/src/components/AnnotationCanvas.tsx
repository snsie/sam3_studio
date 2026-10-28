import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import type {
  BrushMode,
  BrushStrokeSegment,
  EditableMaskLayer,
  InteractionMode,
  MaskCandidate,
  PromptBox,
  PromptPoint,
} from "../types";
import { drawLayersToCanvas } from "../utils/challengeMasks";

interface AnnotationCanvasProps {
  imageUrl: string | null;
  groundTruthOverlayUrl: string | null;
  imageWidth: number;
  imageHeight: number;
  headerActions?: ReactNode;
  interactionMode: InteractionMode;
  brushMode: BrushMode;
  brushRadius: number;
  overlayOpacity: number;
  disabled: boolean;
  points: PromptPoint[];
  boxes: PromptBox[];
  samMasks: MaskCandidate[];
  layers: EditableMaskLayer[];
  onAddPoint: (point: PromptPoint) => void;
  onRemovePoint: (pointId: string) => void;
  onAddBox: (box: PromptBox) => void;
  onRemoveBox: (boxId: string) => void;
  onBrushStroke: (stroke: BrushStrokeSegment) => void;
}

interface ImagePoint {
  x: number;
  y: number;
}

interface BoxDraftState {
  start: ImagePoint;
  current: ImagePoint;
}

interface ViewState {
  x: number;
  y: number;
  scale: number;
}

interface PanDragState {
  kind: "pan";
  startClientX: number;
  startClientY: number;
  startViewX: number;
  startViewY: number;
}

interface BoxDragState {
  kind: "box";
  start: ImagePoint;
}

type DragState = PanDragState | BoxDragState;

interface DisplayFrame {
  left: number;
  top: number;
  width: number;
  height: number;
}

const CLICK_DRAG_THRESHOLD = 4;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function getDisplayFrame(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): DisplayFrame | null {
  if (!containerWidth || !containerHeight || !imageWidth || !imageHeight) {
    return null;
  }

  const containerRatio = containerWidth / containerHeight;
  const imageRatio = imageWidth / imageHeight;

  if (containerRatio > imageRatio) {
    const height = containerHeight;
    const width = height * imageRatio;
    return {
      left: (containerWidth - width) / 2,
      top: 0,
      width,
      height,
    };
  }

  const width = containerWidth;
  const height = width / imageRatio;
  return {
    left: 0,
    top: (containerHeight - height) / 2,
    width,
    height,
  };
}

function promptLabelForMode(mode: InteractionMode) {
  return mode === "negative_point" ? "negative" : "positive";
}

function promptSourceForMode(mode: InteractionMode) {
  if (mode === "negative_point") {
    return "negative_point" as const;
  }
  return "positive_point" as const;
}

export function AnnotationCanvas({
  imageUrl,
  groundTruthOverlayUrl,
  imageWidth,
  imageHeight,
  headerActions,
  interactionMode,
  brushMode,
  brushRadius,
  overlayOpacity,
  disabled,
  points,
  boxes,
  samMasks,
  layers,
  onAddPoint,
  onRemovePoint,
  onAddBox,
  onRemoveBox,
  onBrushStroke,
}: AnnotationCanvasProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const editableCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointerStartRef = useRef<{ clientX: number; clientY: number } | null>(null);
  const isPaintingRef = useRef(false);
  const lastPaintPointRef = useRef<ImagePoint | null>(null);
  const [view, setView] = useState<ViewState>({ x: 0, y: 0, scale: 1 });
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [draftBox, setDraftBox] = useState<BoxDraftState | null>(null);
  const [cursorPoint, setCursorPoint] = useState<ImagePoint | null>(null);
  const [stageSize, setStageSize] = useState({ width: 0, height: 0 });
  const [isGroundTruthVisible, setIsGroundTruthVisible] = useState(false);

  const hasImage = Boolean(imageUrl && imageWidth > 0 && imageHeight > 0);

  useEffect(() => {
    setIsGroundTruthVisible(false);
  }, [groundTruthOverlayUrl, imageUrl]);

  useEffect(() => {
    setView({ x: 0, y: 0, scale: 1 });
    setDragState(null);
    setDraftBox(null);
    setCursorPoint(null);
    pointerStartRef.current = null;
    isPaintingRef.current = false;
    lastPaintPointRef.current = null;
  }, [imageHeight, imageUrl, imageWidth]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) {
      return;
    }

    const updateStageSize = () => {
      const nextWidth = stage.clientWidth;
      const nextHeight = stage.clientHeight;
      setStageSize((current) =>
        current.width === nextWidth && current.height === nextHeight
          ? current
          : { width: nextWidth, height: nextHeight },
      );
    };

    updateStageSize();

    const observer = new ResizeObserver(updateStageSize);
    observer.observe(stage);
    return () => {
      observer.disconnect();
    };
  }, [hasImage]);

  const displayFrame = useMemo(() => {
    const baseFrame = getDisplayFrame(stageSize.width, stageSize.height, imageWidth, imageHeight);
    if (!baseFrame) {
      return null;
    }

    return {
      left: baseFrame.left + view.x,
      top: baseFrame.top + view.y,
      width: baseFrame.width * view.scale,
      height: baseFrame.height * view.scale,
    };
  }, [imageHeight, imageWidth, stageSize.height, stageSize.width, view.scale, view.x, view.y]);

  useLayoutEffect(() => {
    const canvas = editableCanvasRef.current;
    if (!canvas || !displayFrame || !imageWidth || !imageHeight) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const draw = () => {
      drawLayersToCanvas(context, layers, imageWidth, imageHeight, overlayOpacity);
    };

    draw();
    const frameId = window.requestAnimationFrame(draw);

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [displayFrame, imageHeight, imageWidth, layers, overlayOpacity]);

  const getImagePoint = (clientX: number, clientY: number): ImagePoint | null => {
    const stage = stageRef.current;
    if (!stage || !imageWidth || !imageHeight) {
      return null;
    }

    const bounds = stage.getBoundingClientRect();
    const baseFrame = getDisplayFrame(bounds.width, bounds.height, imageWidth, imageHeight);
    if (!baseFrame) {
      return null;
    }

    const frameLeft = baseFrame.left + view.x;
    const frameTop = baseFrame.top + view.y;
    const frameWidth = baseFrame.width * view.scale;
    const frameHeight = baseFrame.height * view.scale;

    const x = ((clientX - bounds.left - frameLeft) / frameWidth) * imageWidth;
    const y = ((clientY - bounds.top - frameTop) / frameHeight) * imageHeight;
    return {
      x: clamp(x, 0, imageWidth - 1),
      y: clamp(y, 0, imageHeight - 1),
    };
  };

  const draftBoxStyle = useMemo(() => {
    if (!draftBox || !imageWidth || !imageHeight) {
      return null;
    }
    const left = (Math.min(draftBox.start.x, draftBox.current.x) / imageWidth) * 100;
    const top = (Math.min(draftBox.start.y, draftBox.current.y) / imageHeight) * 100;
    const width = (Math.abs(draftBox.current.x - draftBox.start.x) / imageWidth) * 100;
    const height = (Math.abs(draftBox.current.y - draftBox.start.y) / imageHeight) * 100;
    return { left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` };
  }, [draftBox, imageHeight, imageWidth]);

  const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (!hasImage) {
      return;
    }

    const stage = stageRef.current;
    if (!stage) {
      return;
    }

    event.preventDefault();

    const bounds = stage.getBoundingClientRect();
    const baseFrame = getDisplayFrame(bounds.width, bounds.height, imageWidth, imageHeight);
    if (!baseFrame) {
      return;
    }

    const localX = event.clientX - bounds.left;
    const localY = event.clientY - bounds.top;
    const frameLeft = baseFrame.left + view.x;
    const frameTop = baseFrame.top + view.y;
    const frameWidth = baseFrame.width * view.scale;
    const frameHeight = baseFrame.height * view.scale;
    const pointX = ((localX - frameLeft) / frameWidth) * imageWidth;
    const pointY = ((localY - frameTop) / frameHeight) * imageHeight;
    const nextScale = clamp(view.scale * (event.deltaY < 0 ? 1.1 : 0.92), 0.08, 12);
    const nextFrameWidth = baseFrame.width * nextScale;
    const nextFrameHeight = baseFrame.height * nextScale;

    setView({
      scale: nextScale,
      x: localX - baseFrame.left - (pointX / imageWidth) * nextFrameWidth,
      y: localY - baseFrame.top - (pointY / imageHeight) * nextFrameHeight,
    });
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!hasImage || disabled) {
      return;
    }

    const stage = stageRef.current;
    if (!stage) {
      return;
    }

    if (event.button === 2) {
      pointerStartRef.current = null;
      return;
    }

    if (event.button === 1) {
      event.preventDefault();
      stage.setPointerCapture(event.pointerId);
      setDragState({
        kind: "pan",
        startClientX: event.clientX,
        startClientY: event.clientY,
        startViewX: view.x,
        startViewY: view.y,
      });
      pointerStartRef.current = null;
      isPaintingRef.current = false;
      setDraftBox(null);
      return;
    }

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    pointerStartRef.current = { clientX: event.clientX, clientY: event.clientY };
    if (brushMode !== "off") {
      isPaintingRef.current = true;
      lastPaintPointRef.current = point;
      onBrushStroke({
        fromX: point.x,
        fromY: point.y,
        toX: point.x,
        toY: point.y,
        draw: brushMode === "draw",
      });
      return;
    }
    if (interactionMode === "box") {
      stage.setPointerCapture(event.pointerId);
      setDragState({ kind: "box", start: point });
      setDraftBox({ start: point, current: point });
      return;
    }

    setDragState(null);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!hasImage) {
      return;
    }

    if (dragState?.kind === "pan") {
      setView((current) => ({
        ...current,
        x: dragState.startViewX + event.clientX - dragState.startClientX,
        y: dragState.startViewY + event.clientY - dragState.startClientY,
      }));
      setCursorPoint(null);
      return;
    }

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }
    setCursorPoint(point);

    if (brushMode !== "off" && isPaintingRef.current) {
      const previousPoint = lastPaintPointRef.current ?? point;
      lastPaintPointRef.current = point;
      onBrushStroke({
        fromX: previousPoint.x,
        fromY: previousPoint.y,
        toX: point.x,
        toY: point.y,
        draw: brushMode === "draw",
      });
      return;
    }
    if (dragState?.kind === "box" && draftBox) {
      setDraftBox({
        start: dragState.start,
        current: point,
      });
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    const stage = stageRef.current;
    if (stage?.hasPointerCapture(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }

    if (!hasImage || disabled) {
      return;
    }

    if (event.button === 2) {
      pointerStartRef.current = null;
      return;
    }

    if (dragState?.kind === "pan") {
      setDragState(null);
      pointerStartRef.current = null;
      return;
    }

    const point = getImagePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    if (brushMode !== "off") {
      isPaintingRef.current = false;
      lastPaintPointRef.current = null;
      pointerStartRef.current = null;
      return;
    }

    if (dragState?.kind === "box" && draftBox) {
      const width = Math.abs(point.x - draftBox.start.x);
      const height = Math.abs(point.y - draftBox.start.y);
      if (width >= 2 && height >= 2) {
        onAddBox({
          id: crypto.randomUUID(),
          x0: Math.min(draftBox.start.x, point.x),
          y0: Math.min(draftBox.start.y, point.y),
          x1: Math.max(draftBox.start.x, point.x),
          y1: Math.max(draftBox.start.y, point.y),
          label: "positive",
        });
      }
      setDraftBox(null);
      setDragState(null);
      pointerStartRef.current = null;
      return;
    }

    const pointerStart = pointerStartRef.current;
    pointerStartRef.current = null;
    const deltaX = pointerStart ? event.clientX - pointerStart.clientX : 0;
    const deltaY = pointerStart ? event.clientY - pointerStart.clientY : 0;
    if (Math.hypot(deltaX, deltaY) > CLICK_DRAG_THRESHOLD) {
      return;
    }

    const pointPayload: PromptPoint = {
      id: crypto.randomUUID(),
      x: point.x,
      y: point.y,
      label: promptLabelForMode(interactionMode),
      source: promptSourceForMode(interactionMode),
    };

    if (interactionMode !== "positive_point" && interactionMode !== "negative_point") {
      return;
    }

    onAddPoint(pointPayload);
  };

  const handlePointerLeave = () => {
    if (dragState?.kind === "pan" || dragState?.kind === "box") {
      return;
    }
    setCursorPoint(null);
    isPaintingRef.current = false;
    lastPaintPointRef.current = null;
  };

  const handlePointerCancel = (event: React.PointerEvent<HTMLDivElement>) => {
    const stage = stageRef.current;
    if (stage?.hasPointerCapture(event.pointerId)) {
      stage.releasePointerCapture(event.pointerId);
    }

    pointerStartRef.current = null;
    isPaintingRef.current = false;
    lastPaintPointRef.current = null;
    setDragState(null);
    setDraftBox(null);
    setCursorPoint(null);
  };

  const handleCanvasContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    pointerStartRef.current = null;
  };

  const handleCanvasMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button === 1) {
      event.preventDefault();
    }
  };

  const interceptCanvasPointerEvent = (event: React.PointerEvent<HTMLElement>) => {
    event.stopPropagation();
  };

  const handleRemovePoint = (
    event: React.MouseEvent<HTMLSpanElement>,
    pointId: string,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    if (disabled) {
      return;
    }
    onRemovePoint(pointId);
  };

  const handleRemoveBox = (
    event: React.MouseEvent<HTMLDivElement>,
    boxId: string,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    if (disabled) {
      return;
    }
    onRemoveBox(boxId);
  };

  return (
    <section className="annotation-canvas-shell panel-block">
      <div className="section-heading compact-heading annotation-canvas-heading">
        <div className="annotation-canvas-title">
          {/* <span className="section-kicker">Canvas</span> */}
          <h2>Annotation Canvas</h2>
        </div>
        {headerActions ? <div className="annotation-canvas-header-actions">{headerActions}</div> : null}
      </div>
      {!hasImage ? (
        <div className="viewer-empty-state challenge-empty-state">
          <h3>Select a challenge image</h3>
          <p>Load an image from the sidebar to start prompting, brushing, and storing masks.</p>
        </div>
      ) : (
        <div
          ref={stageRef}
          className={`annotation-stage${disabled ? " is-disabled" : ""}`}
          onMouseDown={handleCanvasMouseDown}
          onWheel={handleWheel}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onPointerLeave={handlePointerLeave}
          onContextMenu={handleCanvasContextMenu}
        >
          <div className="viewer-stage-hud viewer-stage-hud-top">
            <span className="viewer-scale-badge">{Math.round(view.scale * 100)}%</span>
            <span className="viewer-hint-badge">Wheel to zoom. Middle-drag to pan.</span>
          </div>
          <button
            type="button"
            className="viewer-fit-button"
            onPointerDown={interceptCanvasPointerEvent}
            onPointerUp={interceptCanvasPointerEvent}
            onClick={(event) => {
              event.stopPropagation();
              setView({ x: 0, y: 0, scale: 1 });
            }}
          >
            Reset view
          </button>
          {displayFrame ? (
            <div
              className="annotation-image-frame"
              style={{
                left: displayFrame.left,
                top: displayFrame.top,
                width: displayFrame.width,
                height: displayFrame.height,
              }}
            >
              <img
                src={imageUrl ?? undefined}
                alt="Challenge annotation target"
                className="annotation-stage-image"
                draggable={false}
              />
              {samMasks.map((mask) => (
                <img
                  key={mask.candidate_id}
                  src={mask.image_data_url}
                  alt={mask.candidate_id}
                  className="annotation-sam-overlay"
                  style={{ opacity: overlayOpacity }}
                  draggable={false}
                />
              ))}
              <canvas
                ref={editableCanvasRef}
                width={imageWidth}
                height={imageHeight}
                className="annotation-editable-overlay"
              />
              {isGroundTruthVisible && groundTruthOverlayUrl ? (
                <img
                  src={groundTruthOverlayUrl}
                  alt="Ground truth overlay"
                  className="annotation-ground-truth-overlay"
                  draggable={false}
                />
              ) : null}

              {boxes.map((box) => (
                <div
                  key={box.id}
                  className="annotation-box"
                  onPointerDown={(event) => event.stopPropagation()}
                  onPointerUp={(event) => event.stopPropagation()}
                  onContextMenu={(event) => handleRemoveBox(event, box.id)}
                  style={{
                    left: `${(box.x0 / imageWidth) * 100}%`,
                    top: `${(box.y0 / imageHeight) * 100}%`,
                    width: `${((box.x1 - box.x0) / imageWidth) * 100}%`,
                    height: `${((box.y1 - box.y0) / imageHeight) * 100}%`,
                  }}
                />
              ))}
              {draftBoxStyle ? <div className="annotation-box is-draft" style={draftBoxStyle} /> : null}

              {points.map((point) => (
                <span
                  key={point.id}
                  className={`annotation-point annotation-point-${point.label}`}
                  onPointerDown={(event) => event.stopPropagation()}
                  onPointerUp={(event) => event.stopPropagation()}
                  onContextMenu={(event) => handleRemovePoint(event, point.id)}
                  style={{
                    left: `${(point.x / imageWidth) * 100}%`,
                    top: `${(point.y / imageHeight) * 100}%`,
                  }}
                />
              ))}

              {brushMode !== "off" && cursorPoint ? (
                <span
                  className={`brush-cursor brush-cursor-${brushMode}`}
                  style={{
                    left: `${(cursorPoint.x / imageWidth) * 100}%`,
                    top: `${(cursorPoint.y / imageHeight) * 100}%`,
                    width: `${(brushRadius * 2 * 100) / imageWidth}%`,
                    height: `${(brushRadius * 2 * 100) / imageHeight}%`,
                  }}
                />
              ) : null}

              {groundTruthOverlayUrl ? (
                <button
                  type="button"
                  className={`annotation-ground-truth-toggle${isGroundTruthVisible ? " is-active" : ""
                    }`}
                  aria-pressed={isGroundTruthVisible}
                  onPointerDown={interceptCanvasPointerEvent}
                  onPointerUp={interceptCanvasPointerEvent}
                  onClick={(event) => {
                    event.stopPropagation();
                    setIsGroundTruthVisible((current) => !current);
                  }}
                >
                  {isGroundTruthVisible ? "Hide Ground Truth" : "Peek Ground Truth"}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}