/**
 * A deliberately tiny segmentation "model": per-pixel logistic regression on a handful of
 * hand-made local features, trained with real minibatch gradient descent. The loop has the
 * same shape as any segmentation training run: forward → loss → gradients → update.
 */
import { createRandom, randomBetween, randomNormal } from "./seededRandom";
import { blobLabelMap, boxMean, labelsToMask, renderMicroscopy, scatterBlobs, type Blob } from "./syntheticCells";

export const TOY_SIZE = 64;
const PIXELS = TOY_SIZE * TOY_SIZE;
export const TRAIN_POOL_SIZE = 48;
export const VAL_SIZE = 16;
const BATCH_SIZE = 8;
const DICE_WEIGHT = 1;

export const FEATURE_NAMES = ["pixel", "3×3 mean", "5×5 mean", "3×3 std. dev.", "local contrast", "15×15 mean"];
const FEATURE_COUNT = FEATURE_NAMES.length;

export interface ToySample {
  blobs: Blob[];
  image: Float32Array;
  gt: Uint8Array;
  /** Pixel-major standardized features: features[pixel * FEATURE_COUNT + feature]. */
  features: Float32Array;
}

export interface ToyDataset {
  train: ToySample[];
  val: ToySample[];
}

function rawFeatures(image: Float32Array) {
  const mean3 = boxMean(image, TOY_SIZE, TOY_SIZE, 1);
  const mean5 = boxMean(image, TOY_SIZE, TOY_SIZE, 2);
  const mean15 = boxMean(image, TOY_SIZE, TOY_SIZE, 7);
  const squares = boxMean(Float32Array.from(image, (value) => value * value), TOY_SIZE, TOY_SIZE, 1);
  const features = new Float32Array(PIXELS * FEATURE_COUNT);
  for (let pixel = 0; pixel < PIXELS; pixel += 1) {
    const offset = pixel * FEATURE_COUNT;
    features[offset] = image[pixel];
    features[offset + 1] = mean3[pixel];
    features[offset + 2] = mean5[pixel];
    features[offset + 3] = Math.sqrt(Math.max(0, squares[pixel] - mean3[pixel] * mean3[pixel]));
    features[offset + 4] = mean5[pixel] - mean15[pixel];
    features[offset + 5] = mean15[pixel];
  }
  return features;
}

/** Seeded dataset: 48 training candidates and 16 validation images, features standardized on train. */
export function createToyDataset(seed = 7): ToyDataset {
  const random = createRandom(seed);
  const samples = Array.from({ length: TRAIN_POOL_SIZE + VAL_SIZE }, () => {
    const blobs = scatterBlobs(random, Math.round(randomBetween(random, 2, 5)), TOY_SIZE, TOY_SIZE, [5, 10]);
    const image = renderMicroscopy(random, blobs, TOY_SIZE, TOY_SIZE, {
      noise: 0.16,
      illumination: randomBetween(random, 0.1, 0.4),
      debris: Math.round(randomBetween(random, 1, 4)),
      blurRadius: 1,
    });
    return { blobs, image, gt: labelsToMask(blobLabelMap(blobs, TOY_SIZE, TOY_SIZE)), features: rawFeatures(image) };
  });

  const mean = new Float64Array(FEATURE_COUNT);
  const sq = new Float64Array(FEATURE_COUNT);
  const train = samples.slice(0, TRAIN_POOL_SIZE);
  for (const sample of train) {
    for (let index = 0; index < sample.features.length; index += 1) {
      mean[index % FEATURE_COUNT] += sample.features[index];
      sq[index % FEATURE_COUNT] += sample.features[index] ** 2;
    }
  }
  const count = train.length * PIXELS;
  const std = Array.from(mean, (sum, feature) => Math.sqrt(Math.max(1e-9, sq[feature] / count - (sum / count) ** 2)));
  for (const sample of samples) {
    for (let index = 0; index < sample.features.length; index += 1) {
      const feature = index % FEATURE_COUNT;
      sample.features[index] = (sample.features[index] - mean[feature] / count) / std[feature];
    }
  }
  return { train, val: samples.slice(TRAIN_POOL_SIZE) };
}

/**
 * Simulated annotation noise: sloppy outlines (each nucleus drawn up to ~60% too large or
 * too small) and missed nuclei. Seeded per image, so a given noise level is reproducible.
 */
export function noisyLabels(sample: ToySample, index: number, noise: number): Uint8Array {
  if (noise <= 0) return sample.gt;
  const random = createRandom(1000 + index);
  const scales = sample.blobs.map(() => {
    const missed = random() < noise * 0.35;
    const scale = 1 + noise * randomBetween(random, -0.6, 0.8);
    return missed ? 0 : scale;
  });
  return labelsToMask(blobLabelMap(sample.blobs, TOY_SIZE, TOY_SIZE, scales));
}

export interface TrainConfig {
  learningRate: number;
  trainCount: number;
  labelNoise: number;
  useDiceLoss: boolean;
}

export interface EpochRecord {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  valDice: number;
}

export interface TrainerState {
  /** FEATURE_COUNT weights followed by the bias. */
  weights: Float64Array;
  epoch: number;
  history: EpochRecord[];
  trainLabels: Uint8Array[];
  random: () => number;
}

export function createTrainerState(dataset: ToyDataset, config: TrainConfig, seed = 11): TrainerState {
  const random = createRandom(seed);
  const weights = Float64Array.from({ length: FEATURE_COUNT + 1 }, () => 0.3 * randomNormal(random));
  return {
    weights,
    epoch: 0,
    history: [],
    trainLabels: dataset.train.map((sample, index) => noisyLabels(sample, index, config.labelNoise)),
    random,
  };
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const EPS = 1e-7;

/** Forward pass: probability per pixel. */
export function predictProbabilities(weights: Float64Array, sample: ToySample, out = new Float32Array(PIXELS)) {
  const bias = weights[FEATURE_COUNT];
  for (let pixel = 0; pixel < PIXELS; pixel += 1) {
    let z = bias;
    const offset = pixel * FEATURE_COUNT;
    for (let feature = 0; feature < FEATURE_COUNT; feature += 1) z += weights[feature] * sample.features[offset + feature];
    out[pixel] = sigmoid(z);
  }
  return out;
}

interface LossParts {
  bce: number;
  dice: number;
  sumP: number;
  sumY: number;
  sumPY: number;
}

function lossParts(probabilities: Float32Array[], labels: Uint8Array[]): LossParts {
  let bce = 0;
  let sumP = 0;
  let sumY = 0;
  let sumPY = 0;
  probabilities.forEach((probs, imageIndex) => {
    const label = labels[imageIndex];
    for (let pixel = 0; pixel < PIXELS; pixel += 1) {
      const p = Math.min(1 - EPS, Math.max(EPS, probs[pixel]));
      const y = label[pixel];
      bce -= y ? Math.log(p) : Math.log(1 - p);
      sumP += p;
      sumY += y;
      sumPY += p * y;
    }
  });
  const n = probabilities.length * PIXELS;
  const softDice = (2 * sumPY + 1) / (sumP + sumY + 1);
  return { bce: bce / n, dice: 1 - softDice, sumP, sumY, sumPY };
}

function totalLoss(parts: LossParts, useDice: boolean) {
  return parts.bce + (useDice ? DICE_WEIGHT * parts.dice : 0);
}

function hardDice(probs: Float32Array, gt: Uint8Array) {
  let intersection = 0;
  let predicted = 0;
  let actual = 0;
  for (let pixel = 0; pixel < PIXELS; pixel += 1) {
    const p = probs[pixel] > 0.5 ? 1 : 0;
    intersection += p & gt[pixel];
    predicted += p;
    actual += gt[pixel];
  }
  return predicted + actual === 0 ? 1 : (2 * intersection) / (predicted + actual);
}

/**
 * One pass over the training images in shuffled minibatches, then a validation pass.
 * Yields after every minibatch and validation image so callers can spread an epoch
 * over several animation frames.
 */
export function* epochSteps(state: TrainerState, dataset: ToyDataset, config: TrainConfig): Generator<void, EpochRecord> {
  const order = Array.from({ length: Math.min(config.trainCount, dataset.train.length) }, (_, index) => index);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(state.random() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }

  let lossSum = 0;
  let batches = 0;
  const gradient = new Float64Array(FEATURE_COUNT + 1);
  for (let start = 0; start < order.length; start += BATCH_SIZE) {
    const batch = order.slice(start, start + BATCH_SIZE);
    const samples = batch.map((index) => dataset.train[index]);
    const labels = batch.map((index) => state.trainLabels[index]);

    // 1. Forward pass.
    const probabilities = samples.map((sample) => predictProbabilities(state.weights, sample));
    // 2. Loss.
    const parts = lossParts(probabilities, labels);
    lossSum += totalLoss(parts, config.useDiceLoss);
    batches += 1;

    // 3. Gradients: dLoss/dz per pixel, chained into the weights.
    gradient.fill(0);
    const n = samples.length * PIXELS;
    const diceNumerator = 2 * parts.sumPY + 1;
    const diceDenominator = parts.sumP + parts.sumY + 1;
    samples.forEach((sample, imageIndex) => {
      const probs = probabilities[imageIndex];
      const label = labels[imageIndex];
      for (let pixel = 0; pixel < PIXELS; pixel += 1) {
        const p = probs[pixel];
        let dz = (p - label[pixel]) / n;
        if (config.useDiceLoss) {
          const dDiceDp = (2 * label[pixel] * diceDenominator - diceNumerator) / (diceDenominator * diceDenominator);
          dz -= DICE_WEIGHT * dDiceDp * p * (1 - p);
        }
        const offset = pixel * FEATURE_COUNT;
        for (let feature = 0; feature < FEATURE_COUNT; feature += 1) gradient[feature] += dz * sample.features[offset + feature];
        gradient[FEATURE_COUNT] += dz;
      }
    });

    // 4. Update (plain SGD).
    for (let index = 0; index < gradient.length; index += 1) state.weights[index] -= config.learningRate * gradient[index];
    yield;
  }

  const valProbabilities: Float32Array[] = [];
  for (const sample of dataset.val) {
    valProbabilities.push(predictProbabilities(state.weights, sample));
    yield;
  }
  const valParts = lossParts(valProbabilities, dataset.val.map((sample) => sample.gt));
  const valDice =
    valProbabilities.reduce((sum, probs, index) => sum + hardDice(probs, dataset.val[index].gt), 0) / dataset.val.length;

  state.epoch += 1;
  const record = {
    epoch: state.epoch,
    trainLoss: lossSum / Math.max(batches, 1),
    valLoss: totalLoss(valParts, config.useDiceLoss),
    valDice,
  };
  state.history.push(record);
  return record;
}

/** Runs a whole epoch synchronously. */
export function runEpoch(state: TrainerState, dataset: ToyDataset, config: TrainConfig): EpochRecord {
  const steps = epochSteps(state, dataset, config);
  for (;;) {
    const result = steps.next();
    if (result.done) return result.value;
  }
}
