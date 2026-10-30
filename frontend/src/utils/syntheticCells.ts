/** Seeded synthetic "microscopy" scenes: irregular bright blobs on a dark, noisy background. */
import { createRandom, randomBetween, randomNormal, type Random } from "./seededRandom";
import type { LabelMap, Mask, Point } from "./segMetrics";

export interface Blob {
  cx: number;
  cy: number;
  radius: number;
  /** Ratio of major to minor axis (1 = circle). */
  aspect: number;
  angle: number;
  /** Boundary wobble as [amplitude, frequency, phase] harmonics. */
  harmonics: Array<[number, number, number]>;
  brightness: number;
  /** Bounding half-width beyond which a pixel is certainly outside (plus the soft edge). */
  reach: number;
}

export function randomBlob(random: Random, cx: number, cy: number, radius: number, wobble = 0.12): Blob {
  const aspect = randomBetween(random, 1, 1.45);
  const angle = randomBetween(random, 0, Math.PI);
  const harmonics = [2, 3, 5].map((frequency, index): [number, number, number] => [
    (wobble * randomBetween(random, 0.3, 1)) / (index + 1),
    frequency,
    randomBetween(random, 0, 2 * Math.PI),
  ]);
  return makeBlob({ cx, cy, radius, aspect, angle, harmonics, brightness: randomBetween(random, 0.65, 1) });
}

export function makeBlob(blob: Omit<Blob, "reach">): Blob {
  const wobble = blob.harmonics.reduce((sum, [amplitude]) => sum + Math.abs(amplitude), 0);
  return { ...blob, reach: blob.radius * Math.sqrt(blob.aspect) * (1 + wobble) + 3 };
}

/** Boundary radius of a blob at polar angle `theta`. */
export function blobRadiusAt(blob: Blob, theta: number) {
  const a = blob.radius * Math.sqrt(blob.aspect);
  const b = blob.radius / Math.sqrt(blob.aspect);
  const phi = theta - blob.angle;
  const ellipse = (a * b) / Math.hypot(b * Math.cos(phi), a * Math.sin(phi));
  let wobble = 0;
  for (const [amplitude, frequency, phase] of blob.harmonics) wobble += amplitude * Math.sin(frequency * theta + phase);
  return ellipse * (1 + wobble);
}

/** Signed distance-like value: positive inside the blob, in pixels (exact along rays from the centre). */
export function blobInsideDepth(blob: Blob, x: number, y: number, scale = 1) {
  const dx = x - blob.cx;
  const dy = y - blob.cy;
  return blobRadiusAt(blob, Math.atan2(dy, dx)) * scale - Math.hypot(dx, dy);
}

export function blobPolygon(blob: Blob, vertices = 32, scale = 1): Point[] {
  return Array.from({ length: vertices }, (_, index) => {
    const theta = (index / vertices) * 2 * Math.PI;
    const radius = blobRadiusAt(blob, theta) * scale;
    return [blob.cx + radius * Math.cos(theta), blob.cy + radius * Math.sin(theta)] as Point;
  });
}

/** Places non-overlapping blobs with rejection sampling. */
export function scatterBlobs(
  random: Random,
  count: number,
  width: number,
  height: number,
  radiusRange: [number, number],
  wobble = 0.12,
): Blob[] {
  const blobs: Blob[] = [];
  for (let attempt = 0; attempt < count * 60 && blobs.length < count; attempt += 1) {
    const radius = randomBetween(random, radiusRange[0], radiusRange[1]);
    const margin = radius * 1.4 + 1;
    const cx = randomBetween(random, margin, width - margin);
    const cy = randomBetween(random, margin, height - margin);
    const clear = blobs.every((other) => Math.hypot(other.cx - cx, other.cy - cy) > (other.radius + radius) * 1.35 + 2);
    if (clear) blobs.push(randomBlob(random, cx, cy, radius, wobble));
  }
  return blobs;
}

/** Instance label map (1..N, later blobs win), sampled at pixel centres. */
export function blobLabelMap(blobs: Blob[], width: number, height: number, scales?: number[]): LabelMap {
  const labels = new Int32Array(width * height);
  blobs.forEach((blob, index) => {
    const scale = scales?.[index] ?? 1;
    const reach = blob.reach * scale + 1;
    const x0 = Math.max(0, Math.floor(blob.cx - reach));
    const x1 = Math.min(width - 1, Math.ceil(blob.cx + reach));
    const y0 = Math.max(0, Math.floor(blob.cy - reach));
    const y1 = Math.min(height - 1, Math.ceil(blob.cy + reach));
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        if (scale > 0 && blobInsideDepth(blob, x + 0.5, y + 0.5, scale) > 0) labels[y * width + x] = index + 1;
      }
    }
  });
  return labels;
}

export function labelsToMask(labels: LabelMap): Mask {
  return Uint8Array.from(labels, (value) => (value > 0 ? 1 : 0));
}

/** Box mean with radius r (window 2r+1), edges clamped, via an integral image. */
export function boxMean(image: Float32Array, width: number, height: number, r: number): Float32Array {
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let row = 0;
    for (let x = 0; x < width; x += 1) {
      row += image[y * width + x];
      integral[(y + 1) * (width + 1) + x + 1] = integral[y * (width + 1) + x + 1] + row;
    }
  }
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - r);
    const bottom = Math.min(height, y + r + 1);
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - r);
      const right = Math.min(width, x + r + 1);
      const sum =
        integral[bottom * (width + 1) + right] -
        integral[top * (width + 1) + right] -
        integral[bottom * (width + 1) + left] +
        integral[top * (width + 1) + left];
      out[y * width + x] = sum / ((bottom - top) * (right - left));
    }
  }
  return out;
}

export interface MicroscopyOptions {
  noise?: number;
  /** Peak-to-peak strength of a smooth illumination gradient. */
  illumination?: number;
  /** Number of dim, small debris spots that are not labelled. */
  debris?: number;
  blurRadius?: number;
}

/** Renders blobs into a grayscale image in [0, 1]. */
export function renderMicroscopy(
  random: Random,
  blobs: Blob[],
  width: number,
  height: number,
  { noise = 0.06, illumination = 0, debris = 0, blurRadius = 1 }: MicroscopyOptions = {},
): Float32Array {
  const image = new Float32Array(width * height);
  const gradientAngle = randomBetween(random, 0, 2 * Math.PI);
  const spots = Array.from({ length: debris }, () => ({
    x: randomBetween(random, 0, width),
    y: randomBetween(random, 0, height),
    r: randomBetween(random, 1, 2.2),
    b: randomBetween(random, 0.25, 0.45),
  }));
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const u = (x / width - 0.5) * Math.cos(gradientAngle) + (y / height - 0.5) * Math.sin(gradientAngle);
      let value = 0.08 + illumination * (u + 0.5);
      for (const blob of blobs) {
        if (Math.abs(x + 0.5 - blob.cx) > blob.reach || Math.abs(y + 0.5 - blob.cy) > blob.reach) continue;
        const depth = blobInsideDepth(blob, x + 0.5, y + 0.5);
        if (depth > -3) {
          // Soft edge plus a brighter, textured core like a stained nucleus.
          const edge = 1 / (1 + Math.exp(-depth / 0.9));
          const core = Math.min(1, Math.max(0, depth / (blob.radius * 0.8)));
          value += blob.brightness * edge * (0.62 + 0.3 * core + 0.06 * Math.sin(x * 1.7 + y * 2.3));
        }
      }
      for (const spot of spots) {
        const squared = (x - spot.x) ** 2 + (y - spot.y) ** 2;
        if (squared < 16 * spot.r * spot.r) value += spot.b * Math.exp(-squared / (2 * spot.r * spot.r));
      }
      image[y * width + x] = value + noise * randomNormal(random);
    }
  }
  const blurred = blurRadius > 0 ? boxMean(image, width, height, blurRadius) : image;
  for (let index = 0; index < blurred.length; index += 1) {
    blurred[index] = Math.min(1, Math.max(0, blurred[index]));
  }
  return blurred;
}

export interface MicroscopyScene {
  width: number;
  height: number;
  blobs: Blob[];
  image: Float32Array;
}

export function createScene(seed: number, width: number, height: number, count: number, radiusRange: [number, number]): MicroscopyScene {
  const random = createRandom(seed);
  const blobs = scatterBlobs(random, count, width, height, radiusRange);
  const image = renderMicroscopy(random, blobs, width, height, { noise: 0.05, debris: 3 });
  return { width, height, blobs, image };
}
