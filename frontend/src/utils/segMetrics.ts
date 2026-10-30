/**
 * Segmentation metrics on flat row-major masks (1 = foreground, 0 = background).
 *
 * Instance metrics mirror backend/app/services/nuclei_metrics.py so numbers on the
 * Pipeline page line up with the Nuclei Demo page.
 */

export type Mask = Uint8Array;
export type LabelMap = Int32Array;
export type Point = [number, number];

export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

/** Pixel confusion counts. `extraBackground` adds pixels that are background in both masks. */
export function confusionCounts(gt: Mask, pred: Mask, extraBackground = 0): Confusion {
  if (gt.length !== pred.length) {
    throw new Error(`Mask sizes differ: ${gt.length} vs ${pred.length}`);
  }
  let tp = 0;
  let fp = 0;
  let fn = 0;
  for (let index = 0; index < gt.length; index += 1) {
    const g = gt[index] > 0;
    const p = pred[index] > 0;
    if (g && p) tp += 1;
    else if (p) fp += 1;
    else if (g) fn += 1;
  }
  return { tp, fp, fn, tn: gt.length - tp - fp - fn + extraBackground };
}

// Empty-vs-empty counts as a perfect prediction, as in nuclei_metrics.py.
export function iouScore({ tp, fp, fn }: Confusion) {
  const denominator = tp + fp + fn;
  return denominator === 0 ? 1 : tp / denominator;
}

export function diceScore({ tp, fp, fn }: Confusion) {
  const denominator = 2 * tp + fp + fn;
  return denominator === 0 ? 1 : (2 * tp) / denominator;
}

export function precisionScore({ tp, fp }: Confusion) {
  return tp + fp === 0 ? 1 : tp / (tp + fp);
}

export function recallScore({ tp, fn }: Confusion) {
  return tp + fn === 0 ? 1 : tp / (tp + fn);
}

export function pixelAccuracy({ tp, fp, fn, tn }: Confusion) {
  const total = tp + fp + fn + tn;
  return total === 0 ? 1 : (tp + tn) / total;
}

export function diceFromIou(iou: number) {
  return (2 * iou) / (1 + iou);
}

/** Foreground pixels with at least one 4-neighbour outside the mask (or on the image edge). */
export function boundaryPixels(mask: Mask, width: number, height: number): Point[] {
  const points: Point[] = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;
      if (
        x === 0 ||
        y === 0 ||
        x === width - 1 ||
        y === height - 1 ||
        !mask[index - 1] ||
        !mask[index + 1] ||
        !mask[index - width] ||
        !mask[index + width]
      ) {
        points.push([x, y]);
      }
    }
  }
  return points;
}

function nearestDistances(from: Point[], to: Point[]) {
  return from.map(([x, y]) => {
    let best = Infinity;
    for (const [tx, ty] of to) {
      const distance = (tx - x) * (tx - x) + (ty - y) * (ty - y);
      if (distance < best) best = distance;
    }
    return Math.sqrt(best);
  });
}

/** Linear-interpolated percentile (numpy's default), q in [0, 1]. */
export function percentile(values: number[], q: number) {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const position = q * (sorted.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(lower + 1, sorted.length - 1);
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export interface BoundaryDistances {
  /** 95th percentile of all boundary-to-boundary distances, both directions pooled. */
  hd95: number;
  /** Classic Hausdorff distance (the worst boundary point). */
  hausdorff: number;
  /** Mean symmetric boundary distance. */
  meanDistance: number;
}

/** Boundary distances in pixels. Infinity when exactly one mask is empty, 0 when both are. */
export function boundaryDistances(gt: Mask, pred: Mask, width: number, height: number): BoundaryDistances {
  const gtBoundary = boundaryPixels(gt, width, height);
  const predBoundary = boundaryPixels(pred, width, height);
  if (gtBoundary.length === 0 && predBoundary.length === 0) {
    return { hd95: 0, hausdorff: 0, meanDistance: 0 };
  }
  if (gtBoundary.length === 0 || predBoundary.length === 0) {
    return { hd95: Infinity, hausdorff: Infinity, meanDistance: Infinity };
  }
  const distances = [...nearestDistances(gtBoundary, predBoundary), ...nearestDistances(predBoundary, gtBoundary)];
  return {
    hd95: percentile(distances, 0.95),
    hausdorff: Math.max(...distances),
    meanDistance: distances.reduce((sum, value) => sum + value, 0) / distances.length,
  };
}

/** Even-odd fill of a polygon, sampled at pixel centres. */
export function rasterizePolygon(points: Point[], width: number, height: number, target?: Mask, value = 1): Mask {
  const mask = target ?? new Uint8Array(width * height);
  if (points.length < 3) return mask;
  const crossings: number[] = [];
  for (let y = 0; y < height; y += 1) {
    const sampleY = y + 0.5;
    crossings.length = 0;
    for (let index = 0; index < points.length; index += 1) {
      const [x0, y0] = points[index];
      const [x1, y1] = points[(index + 1) % points.length];
      if ((y0 <= sampleY && y1 > sampleY) || (y1 <= sampleY && y0 > sampleY)) {
        crossings.push(x0 + ((sampleY - y0) / (y1 - y0)) * (x1 - x0));
      }
    }
    crossings.sort((left, right) => left - right);
    for (let index = 0; index + 1 < crossings.length; index += 2) {
      const start = Math.max(0, Math.ceil(crossings[index] - 0.5));
      const end = Math.min(width - 1, Math.floor(crossings[index + 1] - 0.5));
      for (let x = start; x <= end; x += 1) mask[y * width + x] = value;
    }
  }
  return mask;
}

/**
 * Kaggle DSB run-length encoding: "start length" pairs, 1-indexed starts, pixels numbered
 * top to bottom then left to right (column-major). Matches bbbc038_data.rle_encode.
 */
export function rleEncode(mask: Mask, width: number, height: number): number[] {
  const runs: number[] = [];
  let runStart = -1;
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const position = x * height + y;
      const on = mask[y * width + x] > 0;
      if (on && runStart < 0) runStart = position;
      if (!on && runStart >= 0) {
        runs.push(runStart + 1, position - runStart);
        runStart = -1;
      }
    }
  }
  if (runStart >= 0) runs.push(runStart + 1, width * height - runStart);
  return runs;
}

/** COCO uncompressed RLE counts: alternating background/foreground run lengths, column-major. */
export function cocoRleCounts(mask: Mask, width: number, height: number): number[] {
  const counts: number[] = [];
  let current = 0;
  let run = 0;
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const value = mask[y * width + x] > 0 ? 1 : 0;
      if (value !== current) {
        counts.push(run);
        current = value;
        run = 0;
      }
      run += 1;
    }
  }
  counts.push(run);
  return counts;
}

export function rleDecode(runs: number[], width: number, height: number): Mask {
  const mask = new Uint8Array(width * height);
  for (let index = 0; index + 1 < runs.length; index += 2) {
    for (let offset = 0; offset < runs[index + 1]; offset += 1) {
      const position = runs[index] - 1 + offset;
      const x = Math.floor(position / height);
      const y = position % height;
      mask[y * width + x] = 1;
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// Instance metrics (DSB 2018)

/** Official DSB 2018 thresholds: 0.50, 0.55, ..., 0.95. */
export const DSB_IOU_THRESHOLDS = Array.from({ length: 10 }, (_, step) => Math.round((0.5 + 0.05 * step) * 100) / 100);

/** IoU[gtIndex][predIndex] for instances 1..N of two non-overlapping label maps. */
export function instanceIouMatrix(gtLabels: LabelMap, predLabels: LabelMap): number[][] {
  let numGt = 0;
  let numPred = 0;
  for (let index = 0; index < gtLabels.length; index += 1) {
    if (gtLabels[index] > numGt) numGt = gtLabels[index];
    if (predLabels[index] > numPred) numPred = predLabels[index];
  }
  const joint = new Float64Array((numGt + 1) * (numPred + 1));
  for (let index = 0; index < gtLabels.length; index += 1) {
    joint[gtLabels[index] * (numPred + 1) + predLabels[index]] += 1;
  }
  const gtAreas = new Float64Array(numGt + 1);
  const predAreas = new Float64Array(numPred + 1);
  for (let g = 0; g <= numGt; g += 1) {
    for (let p = 0; p <= numPred; p += 1) {
      gtAreas[g] += joint[g * (numPred + 1) + p];
      predAreas[p] += joint[g * (numPred + 1) + p];
    }
  }
  const matrix: number[][] = [];
  for (let g = 1; g <= numGt; g += 1) {
    const row: number[] = [];
    for (let p = 1; p <= numPred; p += 1) {
      const intersection = joint[g * (numPred + 1) + p];
      const union = gtAreas[g] + predAreas[p] - intersection;
      row.push(union > 0 ? intersection / union : 0);
    }
    matrix.push(row);
  }
  return matrix;
}

export interface InstanceMatch {
  tp: number;
  fp: number;
  fn: number;
  /** 0-based pred index → 0-based gt index. */
  predToGt: Map<number, number>;
}

/**
 * Matches objects with IoU strictly above `threshold`. For thresholds >= 0.5 each object
 * can overlap at most one other that much, so this is an exact one-to-one matching.
 */
export function matchInstances(iou: number[][], numPred: number, threshold: number): InstanceMatch {
  const predToGt = new Map<number, number>();
  let tp = 0;
  iou.forEach((row, gtIndex) => {
    let matched = false;
    row.forEach((value, predIndex) => {
      if (value > threshold) {
        matched = true;
        predToGt.set(predIndex, gtIndex);
      }
    });
    if (matched) tp += 1;
  });
  return { tp, fp: numPred - predToGt.size, fn: iou.length - tp, predToGt };
}

/** DSB per-threshold score TP / (TP + FP + FN); 1 when both sides are empty. */
export function dsbThresholdScore({ tp, fp, fn }: Pick<InstanceMatch, "tp" | "fp" | "fn">) {
  const denominator = tp + fp + fn;
  return denominator === 0 ? 1 : tp / denominator;
}

export function instanceF1({ tp, fp, fn }: Pick<InstanceMatch, "tp" | "fp" | "fn">) {
  const denominator = 2 * tp + fp + fn;
  return denominator === 0 ? 1 : (2 * tp) / denominator;
}

export function dsbScore(iou: number[][], numPred: number) {
  const perThreshold = DSB_IOU_THRESHOLDS.map((threshold) => dsbThresholdScore(matchInstances(iou, numPred, threshold)));
  return { perThreshold, score: perThreshold.reduce((sum, value) => sum + value, 0) / perThreshold.length };
}
