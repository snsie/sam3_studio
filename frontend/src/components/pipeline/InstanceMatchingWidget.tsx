import { useMemo, useRef, useState, type PointerEvent } from "react";

import { OVERLAP_COLORS, rgbCss } from "../../utils/maskRender";
import {
  DSB_IOU_THRESHOLDS,
  dsbScore,
  dsbThresholdScore,
  instanceF1,
  instanceIouMatrix,
  matchInstances,
} from "../../utils/segMetrics";
import { createRandom, randomBetween } from "../../utils/seededRandom";
import { blobLabelMap, blobPolygon, makeBlob, randomBlob, scatterBlobs, type Blob } from "../../utils/syntheticCells";

const GRID = 128;

const GT_BLOBS: Blob[] = scatterBlobs(createRandom(505), 5, GRID, GRID, [11, 15], 0.1);
const GT_LABELS = blobLabelMap(GT_BLOBS, GRID, GRID);

/**
 * Predictions for a given jitter seed: four GT nuclei found (with shifted, resized, reshaped
 * outlines), one missed, and one spurious detection. Seed 0 is the rehearsed default.
 */
function makePredictions(seed: number): Blob[] {
  const random = createRandom(9000 + seed);
  const spread = seed === 0 ? 0.6 : 1;
  const found = GT_BLOBS.slice(0, 4).map((gt) => {
    const reshaped = randomBlob(random, gt.cx, gt.cy, gt.radius * randomBetween(random, 0.85, 1.15), 0.1);
    return makeBlob({
      ...reshaped,
      cx: gt.cx + spread * randomBetween(random, -5, 5),
      cy: gt.cy + spread * randomBetween(random, -5, 5),
    });
  });
  // A false positive somewhere away from the GT objects.
  let cx = 0;
  let cy = 0;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    cx = randomBetween(random, 16, GRID - 16);
    cy = randomBetween(random, 16, GRID - 16);
    if (GT_BLOBS.every((gt) => Math.hypot(gt.cx - cx, gt.cy - cy) > gt.radius + 18)) break;
  }
  return [...found, randomBlob(random, cx, cy, 9, 0.15)];
}

function polygonPath(blob: Blob) {
  return blobPolygon(blob, 48).map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ") + " Z";
}

interface InstanceMatchingWidgetProps {
  onOpenNucleiDemo?: () => void;
}

export function InstanceMatchingWidget({ onOpenNucleiDemo }: InstanceMatchingWidgetProps) {
  const [jitterSeed, setJitterSeed] = useState(0);
  const [predictions, setPredictions] = useState<Blob[]>(() => makePredictions(0));
  const [threshold, setThreshold] = useState(0.5);
  const dragRef = useRef<{ index: number; dx: number; dy: number } | null>(null);

  const predLabels = useMemo(() => blobLabelMap(predictions, GRID, GRID), [predictions]);
  const iou = useMemo(() => instanceIouMatrix(GT_LABELS, predLabels), [predLabels]);
  // Label maps are sized by the largest id present, so count predictions explicitly.
  const numPred = predictions.length;
  const paddedIou = useMemo(() => iou.map((row) => Array.from({ length: numPred }, (_, index) => row[index] ?? 0)), [iou, numPred]);
  const match = matchInstances(paddedIou, numPred, threshold);
  const dsb = dsbScore(paddedIou, numPred);
  const bestIouForPred = Array.from({ length: numPred }, (_, predIndex) =>
    paddedIou.reduce((best, row) => Math.max(best, row[predIndex]), 0),
  );
  const matchedGt = new Set(match.predToGt.values());

  const toGrid = (event: PointerEvent<SVGElement>) => {
    const svg = (event.currentTarget as SVGElement).ownerSVGElement ?? (event.currentTarget as SVGSVGElement);
    const bounds = svg.getBoundingClientRect();
    return { x: ((event.clientX - bounds.left) / bounds.width) * GRID, y: ((event.clientY - bounds.top) / bounds.height) * GRID };
  };

  const startDrag = (index: number) => (event: PointerEvent<SVGPathElement>) => {
    const point = toGrid(event);
    (event.currentTarget.ownerSVGElement ?? event.currentTarget).setPointerCapture(event.pointerId);
    dragRef.current = { index, dx: predictions[index].cx - point.x, dy: predictions[index].cy - point.y };
  };

  const handlePointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const point = toGrid(event);
    setPredictions((current) =>
      current.map((blob, index) =>
        index === drag.index
          ? { ...blob, cx: Math.min(GRID, Math.max(0, point.x + drag.dx)), cy: Math.min(GRID, Math.max(0, point.y + drag.dy)) }
          : blob,
      ),
    );
  };

  const endDrag = () => {
    dragRef.current = null;
  };

  const resetTo = (seed: number) => {
    setJitterSeed(seed);
    setPredictions(makePredictions(seed));
  };

  return (
    <div className="instance-widget">
      <div className="instance-left">
        <div className="instance-stage">
          <svg
            viewBox={`0 0 ${GRID} ${GRID}`}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
          >
            <rect width={GRID} height={GRID} className="instance-stage-bg" />
            {GT_BLOBS.map((blob, index) => (
              <path
                key={`gt-${index}`}
                d={polygonPath(blob)}
                className={`instance-gt${matchedGt.has(index) ? " is-matched" : " is-missed"}`}
              />
            ))}
            {predictions.map((blob, index) => {
              const isTp = match.predToGt.has(index);
              const color = rgbCss(isTp ? OVERLAP_COLORS.tp : OVERLAP_COLORS.fp);
              return (
                <path
                  key={`pred-${index}`}
                  d={polygonPath(blob)}
                  className="instance-pred"
                  style={{ fill: rgbCss(isTp ? OVERLAP_COLORS.tp : OVERLAP_COLORS.fp, 0.32), stroke: color }}
                  onPointerDown={startDrag(index)}
                />
              );
            })}
            {predictions.map((blob, index) => (
              <text key={`label-${index}`} x={blob.cx} y={blob.cy + 2.2} className="instance-iou-label" textAnchor="middle">
                {bestIouForPred[index].toFixed(2)}
              </text>
            ))}
            {GT_BLOBS.map((blob, index) =>
              matchedGt.has(index) ? null : (
                <text key={`fn-${index}`} x={blob.cx} y={blob.cy - blob.radius - 3} className="instance-fn-label" textAnchor="middle">
                  missed
                </text>
              ),
            )}
          </svg>
        </div>
        <div className="instance-legend">
          <span>
            <i className="is-gt" /> ground truth
          </span>
          <span>
            <i className="is-tp" /> TP pred
          </span>
          <span>
            <i className="is-fp" /> FP pred
          </span>
          <span>
            <i className="is-fn" /> FN (missed GT)
          </span>
        </div>
      </div>

      <div className="instance-right">
        <label className="toy-slider instance-threshold">
          <span>
            IoU threshold <b>{threshold.toFixed(2)}</b>
          </span>
          <input
            type="range"
            className="range-input"
            min={0.5}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <div className="toy-buttons">
          <button type="button" className="secondary-button" onClick={() => resetTo(jitterSeed + 1)}>
            Jitter predictions
          </button>
          <button type="button" className="secondary-button" onClick={() => resetTo(0)}>
            Reset
          </button>
        </div>

        <div className="confusion-chips">
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.tp) }}>
            TP <b>{match.tp}</b>
          </span>
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.fp) }}>
            FP <b>{match.fp}</b>
          </span>
          <span style={{ ["--chip" as string]: rgbCss(OVERLAP_COLORS.fn) }}>
            FN <b>{match.fn}</b>
          </span>
        </div>

        <table className="metric-table">
          <tbody>
            <tr>
              <th scope="row">
                {`F1 @ ${threshold.toFixed(2)}`}
                <small>2TP / (2TP + FP + FN)</small>
              </th>
              <td className="metric-numbers">{`${2 * match.tp} / (${2 * match.tp} + ${match.fp} + ${match.fn})`}</td>
              <td className="metric-value">{instanceF1(match).toFixed(3)}</td>
            </tr>
            <tr>
              <th scope="row">
                {`DSB @ ${threshold.toFixed(2)}`}
                <small>TP / (TP + FP + FN)</small>
              </th>
              <td className="metric-numbers">{`${match.tp} / (${match.tp} + ${match.fp} + ${match.fn})`}</td>
              <td className="metric-value">{dsbThresholdScore(match).toFixed(3)}</td>
            </tr>
          </tbody>
        </table>

        <div className="dsb-strip">
          <div className="dsb-strip-bars">
            {DSB_IOU_THRESHOLDS.map((value, index) => (
              <button
                key={value}
                type="button"
                className={`dsb-bar${Math.abs(value - threshold) < 1e-6 ? " is-active" : ""}`}
                onClick={() => setThreshold(value)}
                title={`TP / (TP + FP + FN) at IoU > ${value.toFixed(2)}`}
              >
                <span className="dsb-bar-track">
                  <i style={{ height: `${dsb.perThreshold[index] * 100}%` }} />
                </span>
                <span>{value.toFixed(2).slice(1)}</span>
              </button>
            ))}
          </div>
          <div className="dsb-strip-score">
            <span>DSB score = mean over 0.50 : 0.05 : 0.95</span>
            <b>{dsb.score.toFixed(3)}</b>
          </div>
        </div>

        {onOpenNucleiDemo ? (
          <button type="button" className="link-button instance-next" onClick={onOpenNucleiDemo}>
            See real results on the Nuclei Demo tab →
          </button>
        ) : null}
      </div>
    </div>
  );
}
