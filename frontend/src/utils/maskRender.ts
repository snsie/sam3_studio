/** Turns masks, label maps and float images into RGBA pixels for <canvas>. */
import type { LabelMap, Mask } from "./segMetrics";

type Rgb = [number, number, number];

// Mirrors --tp / --fp (--accent-warm) / --fn (--danger) in styles.css (.pipeline-page).
export const OVERLAP_COLORS = {
  tp: [34, 197, 94] as Rgb,
  fp: [245, 158, 11] as Rgb,
  fn: [251, 113, 133] as Rgb,
};

// Categorical palette for instance ids (stepped for the dark surface).
export const INSTANCE_PALETTE: Rgb[] = [
  [57, 135, 229],
  [217, 89, 38],
  [25, 158, 112],
  [201, 138, 18],
  [155, 93, 229],
  [224, 82, 151],
  [43, 179, 192],
  [143, 191, 58],
];

export function rgbCss([r, g, b]: Rgb, alpha = 1) {
  return alpha === 1 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function instanceColor(id: number): Rgb {
  return INSTANCE_PALETTE[(id - 1) % INSTANCE_PALETTE.length];
}

/** Fluorescence-style tint (dark → blue → near-white), like a DAPI channel. */
export function microscopyRgba(image: Float32Array, out = new Uint8ClampedArray(image.length * 4)) {
  for (let index = 0; index < image.length; index += 1) {
    const value = image[index];
    out[index * 4] = 255 * 0.78 * value ** 1.8;
    out[index * 4 + 1] = 255 * 0.9 * value ** 1.2;
    out[index * 4 + 2] = 255 * Math.min(1, value * 1.25);
    out[index * 4 + 3] = 255;
  }
  return out;
}

export function binaryRgba(mask: Mask, on: Rgb = [237, 246, 251], off: Rgb = [5, 12, 18]) {
  const out = new Uint8ClampedArray(mask.length * 4);
  for (let index = 0; index < mask.length; index += 1) {
    const color = mask[index] ? on : off;
    out.set(color, index * 4);
    out[index * 4 + 3] = 255;
  }
  return out;
}

export function labelRgba(labels: LabelMap, background: Rgb = [5, 12, 18]) {
  const out = new Uint8ClampedArray(labels.length * 4);
  for (let index = 0; index < labels.length; index += 1) {
    out.set(labels[index] > 0 ? instanceColor(labels[index]) : background, index * 4);
    out[index * 4 + 3] = 255;
  }
  return out;
}

/** Probability map in a dark → teal → white ramp (the accent hue). */
export function probabilityRgba(probabilities: Float32Array) {
  const out = new Uint8ClampedArray(probabilities.length * 4);
  for (let index = 0; index < probabilities.length; index += 1) {
    const p = probabilities[index];
    out[index * 4] = 255 * Math.max(0, p - 0.55) * 2.0;
    out[index * 4 + 1] = 20 + 210 * p;
    out[index * 4 + 2] = 30 + 180 * p;
    out[index * 4 + 3] = 255;
  }
  return out;
}

/**
 * TP / FP / FN colouring of two masks. TN pixels show `background` (RGBA) when given,
 * otherwise they are transparent.
 */
export function overlapRgba(gt: Mask, pred: Mask, background?: Uint8ClampedArray, alpha = 255) {
  const out = background ? Uint8ClampedArray.from(background) : new Uint8ClampedArray(gt.length * 4);
  for (let index = 0; index < gt.length; index += 1) {
    const g = gt[index] > 0;
    const p = pred[index] > 0;
    const color = g && p ? OVERLAP_COLORS.tp : p ? OVERLAP_COLORS.fp : g ? OVERLAP_COLORS.fn : null;
    if (color) {
      out.set(color, index * 4);
      out[index * 4 + 3] = alpha;
    }
  }
  return out;
}
