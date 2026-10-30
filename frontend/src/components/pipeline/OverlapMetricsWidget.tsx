import { useMemo, useRef, useState, type PointerEvent } from "react";

import { OVERLAP_COLORS, overlapRgba, rgbCss } from "../../utils/maskRender";
import {
  boundaryDistances,
  confusionCounts,
  diceFromIou,
  diceScore,
  iouScore,
  pixelAccuracy,
  precisionScore,
  recallScore,
} from "../../utils/segMetrics";
import { blobRadiusAt, makeBlob, type Blob } from "../../utils/syntheticCells";
import { PipelineChart } from "./PipelineChart";
import { PixelCanvas } from "./PixelCanvas";

/** Fixed metric grid; the display scales it, so dragging stays cheap. */
const GRID = 128;
const CENTER = GRID / 2;
const OUTLINE_VERTICES = 120;

type ShapeId = "blob" | "circle" | "ellipse" | "gt";

const SHAPES: Array<{ id: ShapeId; label: string }> = [
  { id: "blob", label: "Blob" },
  { id: "circle", label: "Circle" },
  { id: "ellipse", label: "Ellipse" },
  { id: "gt", label: "GT copy" },
];

const BACKGROUND_FACTORS = [1, 2, 4, 8];

const GT_BLOB: Blob = makeBlob({
  cx: CENTER,
  cy: CENTER,
  radius: 27,
  aspect: 1.35,
  angle: 0.5,
  harmonics: [
    [0.16, 2, 0.4],
    [0.1, 3, 2.1],
    [0.05, 5, 4.0],
  ],
  brightness: 1,
});

const PRED_BLOB: Blob = makeBlob({
  cx: 0,
  cy: 0,
  radius: 28,
  aspect: 1.15,
  angle: 1.9,
  harmonics: [
    [0.12, 2, 1.3],
    [0.12, 3, 5.0],
    [0.06, 4, 0.7],
  ],
  brightness: 1,
});

// Fixed high-frequency wobble so "boundary noise" looks the same on every reload.
const NOISE_HARMONICS: Array<[number, number, number]> = [
  [0.5, 7, 0.3],
  [0.3, 11, 2.2],
  [0.25, 17, 4.1],
  [0.15, 23, 1.0],
];

interface PredictionShape {
  shape: ShapeId;
  cx: number;
  cy: number;
  scale: number;
  noise: number;
  /** Pixels added to (dilate) or removed from (erode) the radius. */
  offset: number;
}

function baseRadius(shape: ShapeId, theta: number) {
  switch (shape) {
    case "circle":
      return 28;
    case "ellipse": {
      const a = 36;
      const b = 21;
      const phi = theta - 0.35;
      return (a * b) / Math.hypot(b * Math.cos(phi), a * Math.sin(phi));
    }
    case "gt":
      return blobRadiusAt(GT_BLOB, theta);
    case "blob":
      return blobRadiusAt(PRED_BLOB, theta);
  }
}

function predictionRadius({ shape, scale, noise, offset }: PredictionShape, theta: number) {
  const wobble = NOISE_HARMONICS.reduce((sum, [amplitude, frequency, phase]) => sum + amplitude * Math.sin(frequency * theta + phase), 0);
  return Math.max(0, baseRadius(shape, theta) * scale + offset + noise * 7 * wobble);
}

function rasterizePrediction(prediction: PredictionShape) {
  const mask = new Uint8Array(GRID * GRID);
  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
      const dx = x + 0.5 - prediction.cx;
      const dy = y + 0.5 - prediction.cy;
      if (Math.hypot(dx, dy) < predictionRadius(prediction, Math.atan2(dy, dx))) mask[y * GRID + x] = 1;
    }
  }
  return mask;
}

function outlinePath(radiusAt: (theta: number) => number, cx: number, cy: number) {
  return (
    Array.from({ length: OUTLINE_VERTICES }, (_, index) => {
      const theta = (index / OUTLINE_VERTICES) * 2 * Math.PI;
      const radius = radiusAt(theta);
      return `${index === 0 ? "M" : "L"}${(cx + radius * Math.cos(theta)).toFixed(2)},${(cy + radius * Math.sin(theta)).toFixed(2)}`;
    }).join(" ") + " Z"
  );
}

function formatDistance(value: number) {
  return Number.isFinite(value) ? `${value.toFixed(1)} px` : "∞";
}

const GT_MASK = (() => {
  const mask = new Uint8Array(GRID * GRID);
  for (let y = 0; y < GRID; y += 1) {
    for (let x = 0; x < GRID; x += 1) {
      const dx = x + 0.5 - CENTER;
      const dy = y + 0.5 - CENTER;
      if (Math.hypot(dx, dy) < blobRadiusAt(GT_BLOB, Math.atan2(dy, dx))) mask[y * GRID + x] = 1;
    }
  }
  return mask;
})();
const GT_OUTLINE = outlinePath((theta) => blobRadiusAt(GT_BLOB, theta), CENTER, CENTER);

const INITIAL_PREDICTION: PredictionShape = { shape: "blob", cx: 74, cy: 58, scale: 1, noise: 0, offset: 0 };

export function OverlapMetricsWidget() {
  const [prediction, setPrediction] = useState<PredictionShape>(INITIAL_PREDICTION);
  const [backgroundFactor, setBackgroundFactor] = useState(1);
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);

  const predMask = useMemo(() => rasterizePrediction(prediction), [prediction]);
  const overlayPixels = useMemo(() => overlapRgba(GT_MASK, predMask, undefined, 215), [predMask]);
  const extraBackground = (backgroundFactor * backgroundFactor - 1) * GRID * GRID;
  const counts = useMemo(() => confusionCounts(GT_MASK, predMask, extraBackground), [predMask, extraBackground]);
  const distances = useMemo(() => boundaryDistances(GT_MASK, predMask, GRID, GRID), [predMask]);
  const predOutline = useMemo(
    () => outlinePath((theta) => predictionRadius(prediction, theta), prediction.cx, prediction.cy),
    [prediction],
  );

  const iou = iouScore(counts);
  const dice = diceScore(counts);
  const { tp, fp, fn, tn } = counts;
  const total = tp + fp + fn + tn;

  // The SVG view spans the metric grid plus extra (all-background) image around it.
  const viewSize = GRID * backgroundFactor;
  const viewOrigin = CENTER - viewSize / 2;

  const toGrid = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: viewOrigin + ((event.clientX - bounds.left) / bounds.width) * viewSize,
      y: viewOrigin + ((event.clientY - bounds.top) / bounds.height) * viewSize,
    };
  };

  const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
    const point = toGrid(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { dx: prediction.cx - point.x, dy: prediction.cy - point.y };
  };

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!dragRef.current) {
      return;
    }
    const point = toGrid(event);
    const { dx, dy } = dragRef.current;
    setPrediction((current) => ({
      ...current,
      cx: Math.min(GRID, Math.max(0, point.x + dx)),
      cy: Math.min(GRID, Math.max(0, point.y + dy)),
    }));
  };

  const endDrag = () => {
    dragRef.current = null;
  };

  const metricRows = [
    { name: "IoU (Jaccard)", formula: "TP / (TP + FP + FN)", numbers: `${tp} / (${tp} + ${fp} + ${fn})`, value: iou.toFixed(3) },
    { name: "Dice (F1)", formula: "2TP / (2TP + FP + FN)", numbers: `${2 * tp} / (${2 * tp} + ${fp} + ${fn})`, value: dice.toFixed(3) },
    { name: "Precision", formula: "TP / (TP + FP)", numbers: `${tp} / (${tp} + ${fp})`, value: precisionScore(counts).toFixed(3) },
    { name: "Recall", formula: "TP / (TP + FN)", numbers: `${tp} / (${tp} + ${fn})`, value: recallScore(counts).toFixed(3) },
    {
      name: "Pixel accuracy",
      formula: "(TP + TN) / all pixels",
      numbers: `(${tp} + ${tn.toLocaleString()}) / ${total.toLocaleString()}`,
      value: pixelAccuracy(counts).toFixed(3),
      warn: backgroundFactor > 1,
    },
    {
      name: "HD95",
      formula: "95th pct. boundary distance",
      numbers: `mean boundary distance ${formatDistance(distances.meanDistance)}`,
      value: formatDistance(distances.hd95),
    },
  ];

  const curve = Array.from({ length: 41 }, (_, index) => ({ x: index / 40, y: diceFromIou(index / 40) }));

  return (
    <div className="overlap-widget">
      <div className="overlap-left">
        <div
          className="overlap-stage"
          style={{ ["--inner-size" as string]: `${100 / backgroundFactor}%`, ["--grid-cells" as string]: backgroundFactor }}
        >
          <div className="overlap-stage-inner">
            <PixelCanvas width={GRID} height={GRID} pixels={overlayPixels} label="Overlap of prediction and ground truth" />
          </div>
          <svg
            viewBox={`${viewOrigin} ${viewOrigin} ${viewSize} ${viewSize}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
          >
            <path d={GT_OUTLINE} className="overlap-gt-outline" />
            <path d={predOutline} className="overlap-pred-outline" />
          </svg>
          <span className="overlap-stage-hint">Drag the prediction</span>
        </div>
        <div className="overlap-controls">
          <div className="inline-toggle-group" role="group" aria-label="Prediction shape">
            {SHAPES.map((shape) => (
              <button
                key={shape.id}
                type="button"
                className={`inline-toggle-button${prediction.shape === shape.id ? " is-active" : ""}`}
                onClick={() =>
                  setPrediction((current) => ({
                    ...current,
                    shape: shape.id,
                    ...(shape.id === "gt" ? { cx: CENTER, cy: CENTER, scale: 1, offset: 0, noise: 0 } : {}),
                  }))
                }
              >
                {shape.label}
              </button>
            ))}
          </div>
          <label className="toy-slider">
            <span>
              Scale <b>{`${prediction.scale.toFixed(2)}×`}</b>
            </span>
            <input
              type="range"
              className="range-input"
              min={0.4}
              max={1.6}
              step={0.02}
              value={prediction.scale}
              onChange={(event) => setPrediction((current) => ({ ...current, scale: Number(event.target.value) }))}
            />
          </label>
          <label className="toy-slider">
            <span>
              Erode / dilate <b>{`${prediction.offset > 0 ? "+" : ""}${prediction.offset} px`}</b>
            </span>
            <input
              type="range"
              className="range-input"
              min={-8}
              max={8}
              step={1}
              value={prediction.offset}
              onChange={(event) => setPrediction((current) => ({ ...current, offset: Number(event.target.value) }))}
            />
          </label>
          <label className="toy-slider">
            <span>
              Boundary noise <b>{prediction.noise.toFixed(1)}</b>
            </span>
            <input
              type="range"
              className="range-input"
              min={0}
              max={1}
              step={0.1}
              value={prediction.noise}
              onChange={(event) => setPrediction((current) => ({ ...current, noise: Number(event.target.value) }))}
            />
          </label>
          <div className="overlap-background-toggle">
            <span>Image size</span>
            <div className="inline-toggle-group" role="group" aria-label="Background amount">
              {BACKGROUND_FACTORS.map((factor) => (
                <button
                  key={factor}
                  type="button"
                  className={`inline-toggle-button${factor === backgroundFactor ? " is-active" : ""}`}
                  onClick={() => setBackgroundFactor(factor)}
                >
                  {`${GRID * factor}²`}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="overlap-right">
        <div className="confusion-chips">
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.tp) }}>
            TP <b>{tp.toLocaleString()}</b>
          </span>
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.fp) }}>
            FP <b>{fp.toLocaleString()}</b>
          </span>
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.fn) }}>
            FN <b>{fn.toLocaleString()}</b>
          </span>
          <span className="is-tn">
            TN <b>{tn.toLocaleString()}</b>
          </span>
        </div>
        <table className="metric-table">
          <tbody>
            {metricRows.map((row) => (
              <tr key={row.name} className={row.warn ? "is-warning" : undefined}>
                <th scope="row">
                  {row.name}
                  <small>{row.formula}</small>
                </th>
                <td className="metric-numbers">{row.numbers}</td>
                <td className="metric-value">{row.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <PipelineChart
          title="Dice vs IoU"
          subtitle={`Dice = 2·IoU / (1 + IoU) ≥ IoU · now ${iou.toFixed(2)} → ${dice.toFixed(2)}`}
          series={[
            { key: "dice", label: "Dice", color: "#199e70", points: curve },
            { key: "diag", label: "y = IoU", color: "rgba(148, 169, 181, 0.7)", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], dashed: true },
          ]}
          markers={[{ key: "now", x: iou, y: dice, color: "var(--accent-warm)" }]}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xLabel="IoU"
          height={160}
          square
        />
      </div>
    </div>
  );
}
