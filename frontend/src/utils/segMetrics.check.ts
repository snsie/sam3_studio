/**
 * Hand-computed sanity checks for segMetrics.ts. Not imported by the app.
 *
 * Run from frontend/ (esbuild ships with Vite, so no extra dependency):
 *   npx esbuild src/utils/segMetrics.check.ts --bundle --platform=node --outfile=node_modules/.cache/segMetrics.check.mjs \
 *     && node node_modules/.cache/segMetrics.check.mjs
 */
import {
  boundaryDistances,
  cocoRleCounts,
  confusionCounts,
  diceFromIou,
  diceScore,
  dsbScore,
  instanceF1,
  instanceIouMatrix,
  iouScore,
  matchInstances,
  pixelAccuracy,
  precisionScore,
  rasterizePolygon,
  recallScore,
  rleDecode,
  rleEncode,
  type Mask,
} from "./segMetrics";

let failures = 0;

function check(name: string, actual: number | string, expected: number | string, tolerance = 1e-9) {
  const ok =
    typeof actual === "number" && typeof expected === "number"
      ? Math.abs(actual - expected) <= tolerance || (actual === expected)
      : actual === expected;
  if (!ok) failures += 1;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${actual}${ok ? "" : ` (expected ${expected})`}`);
}

const W = 10;
const H = 10;

/** Fills the half-open box [x0, x1) × [y0, y1) of `target` with `value`. */
function fillBox<T extends Uint8Array | Int32Array>(target: T, x0: number, y0: number, x1: number, y1: number, value = 1): T {
  for (let y = y0; y < y1; y += 1) for (let x = x0; x < x1; x += 1) target[y * W + x] = value;
  return target;
}

function box(x0: number, y0: number, x1: number, y1: number): Mask {
  return fillBox(new Uint8Array(W * H), x0, y0, x1, y1);
}

// Identical masks.
const square = box(2, 2, 6, 6);
const same = confusionCounts(square, square);
check("identical: TP", same.tp, 16);
check("identical: IoU", iouScore(same), 1);
check("identical: Dice", diceScore(same), 1);
check("identical: HD95", boundaryDistances(square, square, W, H).hd95, 0);

// Disjoint masks.
const disjoint = confusionCounts(box(0, 0, 3, 3), box(6, 6, 9, 9));
check("disjoint: IoU", iouScore(disjoint), 0);
check("disjoint: Dice", diceScore(disjoint), 0);
check("disjoint: precision", precisionScore(disjoint), 0);

// Half overlap: two 4×4 boxes shifted by 2 columns. TP 8, FP 8, FN 8, TN 76.
const half = confusionCounts(box(2, 2, 6, 6), box(4, 2, 8, 6));
check("half: TP/FP/FN/TN", `${half.tp}/${half.fp}/${half.fn}/${half.tn}`, "8/8/8/76");
check("half: IoU = 8/24", iouScore(half), 1 / 3);
check("half: Dice = 16/32", diceScore(half), 0.5);
check("half: precision", precisionScore(half), 0.5);
check("half: recall", recallScore(half), 0.5);
check("half: accuracy = 84/100", pixelAccuracy(half), 0.84);
check("half + 900 bg px: accuracy", pixelAccuracy(confusionCounts(box(2, 2, 6, 6), box(4, 2, 8, 6), 900)), 0.984);
check("half: Dice = 2·IoU/(1+IoU)", diceScore(half), diceFromIou(iouScore(half)));

// Dice = 2·IoU/(1+IoU) on a few random-ish pairs.
for (const [a, b] of [
  [box(0, 0, 5, 7), box(1, 2, 9, 9)],
  [box(3, 3, 4, 4), box(0, 0, 10, 10)],
  [box(0, 0, 10, 1), box(0, 0, 2, 10)],
] as Array<[Mask, Mask]>) {
  const counts = confusionCounts(a, b);
  check(`identity on ${counts.tp}/${counts.fp}/${counts.fn}`, diceScore(counts), diceFromIou(iouScore(counts)));
  check("Dice >= IoU", Number(diceScore(counts) >= iouScore(counts)), 1);
}

// Empty prediction: recall 0, HD95 infinite.
const empty = new Uint8Array(W * H);
check("empty pred: recall", recallScore(confusionCounts(square, empty)), 0);
check("empty pred: HD95", boundaryDistances(square, empty, W, H).hd95, Infinity);

// Box shifted right by 2 px: every boundary distance is 0 or 2 → Hausdorff 2.
check("shift 2px: Hausdorff", boundaryDistances(box(2, 2, 6, 6), box(4, 2, 8, 6), W, H).hausdorff, 2);

// Polygon rasterisation: an axis-aligned 4×3 rectangle covers exactly 12 pixel centres.
const rectangle = rasterizePolygon([[2, 3], [6, 3], [6, 6], [2, 6]], W, H);
check("polygon area", rectangle.reduce((sum, value) => sum + value, 0), 12);

// RLE (DSB, column-major, 1-indexed). Pixel (x=0, y=1) alone is position 2 → "2 1".
const single = new Uint8Array(W * H);
single[1 * W + 0] = 1;
check("RLE single pixel", rleEncode(single, W, H).join(" "), "2 1");
check("RLE column run", rleEncode(box(1, 0, 2, 3), W, H).join(" "), "11 3");
check("COCO RLE counts", cocoRleCounts(box(1, 0, 2, 3), W, H).join(" "), "10 3 87");
check("RLE round trip", Array.from(rleDecode(rleEncode(rectangle, W, H), W, H)).join(""), Array.from(rectangle).join(""));

// Instance matching. GT: objects 1 (4×4) and 2 (2×2). Pred: 1 = GT1 shifted by one column
// (IoU 12/20 = 0.6), 2 = a spurious object. GT 2 is missed.
const gtLabels = new Int32Array(W * H);
fillBox(gtLabels, 0, 0, 4, 4, 1);
fillBox(gtLabels, 7, 7, 9, 9, 2);
const predLabels = new Int32Array(W * H);
fillBox(predLabels, 1, 0, 5, 4, 1);
fillBox(predLabels, 7, 0, 9, 2, 2);
const iou = instanceIouMatrix(gtLabels, predLabels);
check("instance IoU[0][0]", iou[0][0], 0.6);
const at50 = matchInstances(iou, 2, 0.5);
check("match@0.5 TP/FP/FN", `${at50.tp}/${at50.fp}/${at50.fn}`, "1/1/1");
check("F1@0.5 = 2/4", instanceF1(at50), 0.5);
const at60 = matchInstances(iou, 2, 0.6);
check("match@0.6 is strict (IoU 0.6 not > 0.6)", `${at60.tp}/${at60.fp}/${at60.fn}`, "0/2/2");
// Thresholds 0.50 and 0.55 score 1/3, the other eight score 0.
check("DSB score", dsbScore(iou, 2).score, (2 / 3) / 10);

console.log(failures === 0 ? "\nAll checks passed." : `\n${failures} check(s) failed.`);
if (failures > 0) {
  throw new Error(`${failures} segMetrics check(s) failed`);
}
