import type { ChallengeId, EditableMaskLayer, MaskCandidate } from "../types";

const MASK_COLOR_PALETTE: Array<[number, number, number]> = [
  [45, 226, 207],
  [245, 158, 11],
  [251, 113, 133],
  [96, 165, 250],
  [52, 211, 153],
  [244, 114, 182],
  [163, 230, 53],
  [192, 132, 252],
  [248, 113, 113],
  [250, 204, 21],
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function createEmptyMaskData(width: number, height: number) {
  return new Uint8ClampedArray(width * height);
}

export function cloneMaskData(data: Uint8ClampedArray) {
  return new Uint8ClampedArray(data);
}

export function createMaskLayer(
  width: number,
  height: number,
  challengeId: ChallengeId,
  index: number,
  label?: string,
): EditableMaskLayer {
  return {
    mask_id: `mask-${index + 1}`,
    label: label ?? `Mask ${index + 1}`,
    color_rgb: nextMaskColor(challengeId, index),
    data: createEmptyMaskData(width, height),
  };
}

export function normalizeMaskLayerColors(
  challengeId: ChallengeId,
  layers: EditableMaskLayer[],
): EditableMaskLayer[] {
  return layers.map((layer, index) => ({
    ...layer,
    color_rgb: nextMaskColor(challengeId, index),
  }));
}

export function replaceMaskLayerData(
  layer: EditableMaskLayer,
  nextData: Uint8ClampedArray,
): EditableMaskLayer {
  return {
    ...layer,
    data: nextData,
  };
}

function stampBrushCircle(
  target: Uint8ClampedArray,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number,
  draw: boolean,
) {
  const minX = clamp(Math.floor(centerX - radius), 0, width - 1);
  const maxX = clamp(Math.ceil(centerX + radius), 0, width - 1);
  const minY = clamp(Math.floor(centerY - radius), 0, height - 1);
  const maxY = clamp(Math.ceil(centerY + radius), 0, height - 1);
  const radiusSquared = radius * radius;

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const deltaX = x - centerX;
      const deltaY = y - centerY;
      if (deltaX * deltaX + deltaY * deltaY > radiusSquared) {
        continue;
      }
      target[y * width + x] = draw ? 255 : 0;
    }
  }
}

export function updateMaskWithBrush(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number,
  draw: boolean,
): Uint8ClampedArray {
  const next = cloneMaskData(data);
  stampBrushCircle(next, width, height, centerX, centerY, radius, draw);
  return next;
}

/**
 * Stamps the brush along the segment between two pointer samples. Pointer events
 * are coarse relative to image pixels, so stamping only the latest sample leaves
 * gaps in the stroke whenever the cursor moves faster than one brush width per event.
 */
export function updateMaskWithBrushStroke(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  radius: number,
  draw: boolean,
): Uint8ClampedArray {
  const next = cloneMaskData(data);
  const distance = Math.hypot(toX - fromX, toY - fromY);
  const step = Math.max(radius / 2, 0.5);
  const stampCount = Math.min(Math.ceil(distance / step), 4096);

  for (let index = 0; index <= stampCount; index += 1) {
    const ratio = stampCount === 0 ? 1 : index / stampCount;
    stampBrushCircle(
      next,
      width,
      height,
      fromX + (toX - fromX) * ratio,
      fromY + (toY - fromY) * ratio,
      radius,
      draw,
    );
  }

  return next;
}

/**
 * Nearest-neighbour remap of a mask buffer onto a different pixel grid. Mask data is
 * a flat array, so reading it with a stride that does not match the width it was
 * written with skews the mask instead of failing loudly - always remap first.
 */
export function resizeMaskData(
  data: Uint8ClampedArray,
  fromWidth: number,
  fromHeight: number,
  toWidth: number,
  toHeight: number,
): Uint8ClampedArray {
  if (fromWidth === toWidth && fromHeight === toHeight) {
    return cloneMaskData(data);
  }

  const next = createEmptyMaskData(toWidth, toHeight);
  if (fromWidth <= 0 || fromHeight <= 0 || toWidth <= 0 || toHeight <= 0) {
    return next;
  }

  for (let y = 0; y < toHeight; y += 1) {
    const sourceY = Math.min(
      fromHeight - 1,
      Math.floor((y * fromHeight) / toHeight),
    );
    for (let x = 0; x < toWidth; x += 1) {
      const sourceX = Math.min(
        fromWidth - 1,
        Math.floor((x * fromWidth) / toWidth),
      );
      next[y * toWidth + x] = data[sourceY * fromWidth + sourceX];
    }
  }

  return next;
}

export function hasMaskPixels(data: Uint8ClampedArray) {
  return data.some((value) => value > 0);
}

export function countStoredMaskImages(states: Record<string, { stored: boolean }>) {
  return Object.values(states).filter((state) => state.stored).length;
}

export function drawLayersToCanvas(
  context: CanvasRenderingContext2D,
  layers: EditableMaskLayer[],
  width: number,
  height: number,
  opacity: number,
) {
  context.clearRect(0, 0, width, height);
  const imageData = context.createImageData(width, height);
  const expectedLength = width * height;

  layers.forEach((layer) => {
    // Drawing a layer whose buffer was sized for a different image would shear it
    // across rows rather than error, so skip it until the layer is resized to match.
    if (layer.data.length !== expectedLength) {
      return;
    }
    const [red, green, blue] = layer.color_rgb;
    const alpha = Math.round(opacity * 170);
    for (let index = 0; index < layer.data.length; index += 1) {
      if (layer.data[index] === 0) {
        continue;
      }
      const pixelIndex = index * 4;
      imageData.data[pixelIndex] = red;
      imageData.data[pixelIndex + 1] = green;
      imageData.data[pixelIndex + 2] = blue;
      imageData.data[pixelIndex + 3] = alpha;
    }
  });

  context.putImageData(imageData, 0, 0);
}

export function serializeMaskToDataUrl(
  data: Uint8ClampedArray,
  width: number,
  height: number,
) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to create mask serialization canvas.");
  }

  const imageData = context.createImageData(width, height);
  for (let index = 0; index < data.length; index += 1) {
    if (data[index] === 0) {
      continue;
    }
    const pixelIndex = index * 4;
    imageData.data[pixelIndex + 3] = 255;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

async function recolorMaskDataUrl(
  candidate: Pick<MaskCandidate, "image_data_url" | "width" | "height">,
  colorRgb: readonly [number, number, number],
) {
  const image = new Image();
  image.src = candidate.image_data_url;
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = candidate.width;
  canvas.height = candidate.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to recolor SAM mask image.");
  }

  context.drawImage(image, 0, 0, candidate.width, candidate.height);
  const imageData = context.getImageData(0, 0, candidate.width, candidate.height);
  const [red, green, blue] = colorRgb;

  for (let pixelIndex = 0; pixelIndex < candidate.width * candidate.height; pixelIndex += 1) {
    const dataIndex = pixelIndex * 4;
    if (imageData.data[dataIndex + 3] === 0) {
      continue;
    }
    imageData.data[dataIndex] = red;
    imageData.data[dataIndex + 1] = green;
    imageData.data[dataIndex + 2] = blue;
  }

  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

export async function recolorMaskCandidates(
  candidates: MaskCandidate[],
  colorRgb: readonly [number, number, number],
): Promise<MaskCandidate[]> {
  return Promise.all(
    candidates.map(async (candidate) => ({
      ...candidate,
      color_rgb: [...colorRgb],
      image_data_url: await recolorMaskDataUrl(candidate, colorRgb),
    })),
  );
}

/**
 * Decodes a SAM mask onto the caller's pixel grid. `targetWidth`/`targetHeight` must be
 * the dimensions of the buffer the mask will be merged into: a mask decoded at the
 * candidate's own size cannot be indexed with a different width without skewing it.
 */
export async function candidateMaskToMaskData(
  candidate: Pick<MaskCandidate, "image_data_url" | "width" | "height">,
  targetWidth?: number,
  targetHeight?: number,
) {
  const width = targetWidth && targetWidth > 0 ? targetWidth : candidate.width;
  const height = targetHeight && targetHeight > 0 ? targetHeight : candidate.height;

  const image = new Image();
  image.src = candidate.image_data_url;
  await image.decode();

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to decode SAM mask image.");
  }
  context.drawImage(image, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const mask = createEmptyMaskData(width, height);
  for (let pixelIndex = 0; pixelIndex < mask.length; pixelIndex += 1) {
    mask[pixelIndex] = data[pixelIndex * 4 + 3] > 0 ? 255 : 0;
  }
  return mask;
}

export function nextMaskColor(
  challengeId: ChallengeId,
  index: number,
): [number, number, number] {
  if (challengeId === "single-class") {
    return MASK_COLOR_PALETTE[0];
  }
  return MASK_COLOR_PALETTE[index % MASK_COLOR_PALETTE.length];
}