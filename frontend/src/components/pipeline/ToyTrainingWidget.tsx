import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { microscopyRgba, overlapRgba, probabilityRgba } from "../../utils/maskRender";
import {
  FEATURE_NAMES,
  TOY_SIZE,
  TRAIN_POOL_SIZE,
  createToyDataset,
  createTrainerState,
  epochSteps,
  noisyLabels,
  predictProbabilities,
  runEpoch,
  type EpochRecord,
  type ToyDataset,
  type TrainConfig,
  type TrainerState,
} from "../../utils/toyTrainer";
import { PipelineChart, type ChartMarker } from "./PipelineChart";
import { PixelCanvas } from "./PixelCanvas";

const MAX_EPOCHS = 150;
/** Training work per animation frame, so the page keeps rendering at 60 fps. */
const FRAME_BUDGET_MS = 6;
/** Minimum time between epochs, so the curves are watchable live (~15 s per run). */
const EPOCH_INTERVAL_MS = 80;
const DEFAULT_CONFIG: TrainConfig = { learningRate: 0.05, trainCount: TRAIN_POOL_SIZE, labelNoise: 0, useDiceLoss: false };
const LOOP_STEPS = ["Forward", "Loss", "Gradients", "Update"];

const TRAIN_COLOR = "#3987e5";
const VAL_COLOR = "#d95926";
const DICE_COLOR = "#199e70";

function formatLearningRate(value: number) {
  return value >= 1 ? value.toFixed(value >= 10 ? 0 : 1) : value.toPrecision(1);
}

function bestRecord(history: EpochRecord[]) {
  return history.reduce<EpochRecord | null>((best, record) => (best === null || record.valDice > best.valDice ? record : best), null);
}

interface ToyTrainingWidgetProps {
  isVisible: boolean;
}

export function ToyTrainingWidget({ isVisible }: ToyTrainingWidgetProps) {
  const [dataset, setDataset] = useState<ToyDataset | null>(null);
  const [config, setConfig] = useState<TrainConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<EpochRecord[]>([]);
  const [weights, setWeights] = useState<Float64Array | null>(null);
  const [valIndex, setValIndex] = useState(0);
  const stateRef = useRef<TrainerState | null>(null);
  const configRef = useRef(config);
  configRef.current = config;

  // Build the synthetic dataset the first time the step is shown (~0.2 s, once).
  useEffect(() => {
    if (!isVisible || dataset) {
      return;
    }
    const timer = window.setTimeout(() => setDataset(createToyDataset()), 30);
    return () => window.clearTimeout(timer);
  }, [isVisible, dataset]);

  const reset = useCallback(
    (nextConfig: TrainConfig = configRef.current) => {
      if (!dataset) {
        return;
      }
      stateRef.current = createTrainerState(dataset, nextConfig);
      setHistory([]);
      setWeights(Float64Array.from(stateRef.current.weights));
    },
    [dataset],
  );

  useEffect(() => {
    reset();
  }, [reset]);

  const publish = useCallback((state: TrainerState) => {
    setHistory([...state.history]);
    setWeights(Float64Array.from(state.weights));
  }, []);

  const step = useCallback(() => {
    const state = stateRef.current;
    if (!dataset || !state || state.epoch >= MAX_EPOCHS) {
      return;
    }
    runEpoch(state, dataset, configRef.current);
    publish(state);
  }, [dataset, publish]);

  // Spread each epoch over animation frames: minibatches run until the frame budget is spent.
  useEffect(() => {
    if (!running || !isVisible || !dataset) {
      return;
    }
    let frame = 0;
    let steps: Generator<void, EpochRecord> | null = null;
    let stepsState: TrainerState | null = null;
    let lastEpochAt = 0;
    const tick = (now: number) => {
      const state = stateRef.current;
      if (!state || state.epoch >= MAX_EPOCHS) {
        setRunning(false);
        return;
      }
      // A reset swaps in a new state; drop any half-finished epoch of the old one.
      if (stepsState !== state) {
        steps = null;
        stepsState = state;
      }
      const start = performance.now();
      while (now - lastEpochAt >= EPOCH_INTERVAL_MS && performance.now() - start < FRAME_BUDGET_MS) {
        steps ??= epochSteps(state, dataset, configRef.current);
        if (steps.next().done) {
          steps = null;
          lastEpochAt = now;
          publish(state);
        }
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [running, isVisible, dataset, publish]);

  useEffect(() => {
    if (history.length >= MAX_EPOCHS) {
      setRunning(false);
    }
  }, [history.length]);

  const updateConfig = (patch: Partial<TrainConfig>, restart: boolean) => {
    const next = { ...config, ...patch };
    setConfig(next);
    configRef.current = next;
    if (restart) {
      reset(next);
    } else if (patch.labelNoise !== undefined && dataset && stateRef.current) {
      stateRef.current.trainLabels = dataset.train.map((sample, index) => noisyLabels(sample, index, next.labelNoise));
    }
  };

  const sample = dataset?.val[valIndex] ?? null;
  const panels = useMemo(() => {
    if (!sample || !weights) {
      return null;
    }
    const probabilities = predictProbabilities(weights, sample);
    const predicted = Uint8Array.from(probabilities, (value) => (value > 0.5 ? 1 : 0));
    const input = microscopyRgba(sample.image);
    return {
      input,
      probability: probabilityRgba(probabilities),
      overlap: overlapRgba(sample.gt, predicted, input),
    };
  }, [sample, weights]);

  const best = bestRecord(history);
  const latest = history[history.length - 1];
  const epochs = history.map((record) => record.epoch);
  const maxLoss = Math.min(2, Math.max(0.8, ...history.flatMap((record) => [record.trainLoss, record.valLoss])));
  const diceMarkers: ChartMarker[] = best
    ? [{ key: "best", x: best.epoch, y: best.valDice, color: DICE_COLOR, label: `best: epoch ${best.epoch}` }]
    : [];
  const xDomain: [number, number] = [0, Math.max(20, Math.ceil((epochs[epochs.length - 1] ?? 0) / 10) * 10)];
  const maxWeight = weights ? Math.max(1, ...Array.from(weights, Math.abs)) : 1;
  const finished = history.length >= MAX_EPOCHS;

  if (!dataset) {
    return <div className="toy-training is-loading">Generating 64 synthetic training images…</div>;
  }

  return (
    <div className="toy-training">
      <div className="toy-controls">
        <div className="toy-buttons">
          <button
            type="button"
            className="primary-button"
            onClick={() => setRunning((current) => !current)}
            disabled={finished}
          >
            {running ? "Pause" : finished ? "Done" : history.length ? "Resume" : "Train"}
          </button>
          <button type="button" className="secondary-button" onClick={step} disabled={running || finished}>
            +1 epoch
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => {
              setRunning(false);
              reset();
            }}
          >
            Reset
          </button>
        </div>
        <label className="toy-slider">
          <span>
            Learning rate <b>{formatLearningRate(config.learningRate)}</b>
          </span>
          <input
            type="range"
            className="range-input"
            min={-3}
            max={2}
            step={0.1}
            value={Math.log10(config.learningRate)}
            onChange={(event) => updateConfig({ learningRate: Number((10 ** Number(event.target.value)).toPrecision(2)) }, false)}
          />
        </label>
        <label className="toy-slider">
          <span>
            Train images <b>{config.trainCount}</b>
          </span>
          <input
            type="range"
            className="range-input"
            min={1}
            max={TRAIN_POOL_SIZE}
            step={1}
            value={config.trainCount}
            onChange={(event) => updateConfig({ trainCount: Number(event.target.value) }, false)}
          />
        </label>
        <label className="toy-slider">
          <span>
            Label noise <b>{`${Math.round(config.labelNoise * 100)}%`}</b>
          </span>
          <input
            type="range"
            className="range-input"
            min={0}
            max={1}
            step={0.05}
            value={config.labelNoise}
            onChange={(event) => updateConfig({ labelNoise: Number(event.target.value) }, false)}
          />
        </label>
        <label className="toy-checkbox">
          <input
            type="checkbox"
            checked={config.useDiceLoss}
            onChange={(event) => updateConfig({ useDiceLoss: event.target.checked }, true)}
          />
          + Dice loss
        </label>
      </div>

      <div className="toy-main">
        <div className="toy-loop" aria-hidden="true">
          {LOOP_STEPS.map((label, index) => (
            <span key={label} className={`toy-loop-step${running ? " is-running" : ""}`} style={{ animationDelay: `${index * 0.25}s` }}>
              {label}
            </span>
          ))}
          <span className="toy-epoch">
            Epoch <b>{latest?.epoch ?? 0}</b>
            <small>{` / ${MAX_EPOCHS}`}</small>
          </span>
        </div>

        <div className="toy-panels">
          <figure className="pipeline-tile">
            <PixelCanvas width={TOY_SIZE} height={TOY_SIZE} pixels={panels?.input ?? null} label="Validation image" />
            <figcaption>
              <strong>Val image</strong>
              <button
                type="button"
                className="link-button"
                onClick={() => setValIndex((current) => (current + 1) % dataset.val.length)}
              >
                {`#${valIndex + 1} · next →`}
              </button>
            </figcaption>
          </figure>
          <figure className="pipeline-tile">
            <PixelCanvas width={TOY_SIZE} height={TOY_SIZE} pixels={panels?.probability ?? null} label="Predicted probability" />
            <figcaption>
              <strong>Probability</strong>
              <span>p(nucleus) per pixel</span>
            </figcaption>
          </figure>
          <figure className="pipeline-tile">
            <PixelCanvas width={TOY_SIZE} height={TOY_SIZE} pixels={panels?.overlap ?? null} label="Thresholded prediction vs ground truth" />
            <figcaption>
              <strong>p &gt; 0.5 vs GT</strong>
              <span className="overlap-legend">
                <i className="is-tp" />
                TP <i className="is-fp" />
                FP <i className="is-fn" />
                FN
              </span>
            </figcaption>
          </figure>
          <div className="toy-weights">
            <strong>The whole model: 7 numbers</strong>
            {weights
              ? [...FEATURE_NAMES, "bias"].map((name, index) => (
                  <div key={name} className="toy-weight-row">
                    <span>{name}</span>
                    <span className="toy-weight-bar">
                      <i
                        className={weights[index] < 0 ? "is-negative" : ""}
                        style={{
                          width: `${(Math.abs(weights[index]) / maxWeight) * 50}%`,
                          left: weights[index] < 0 ? `${50 - (Math.abs(weights[index]) / maxWeight) * 50}%` : "50%",
                        }}
                      />
                    </span>
                    <b>{weights[index].toFixed(2)}</b>
                  </div>
                ))
              : null}
          </div>
        </div>

        <div className="toy-charts">
          <PipelineChart
            title="Loss"
            subtitle={config.useDiceLoss ? "BCE + Dice loss" : "Binary cross-entropy"}
            series={[
              { key: "train", label: "train", color: TRAIN_COLOR, points: history.map((record) => ({ x: record.epoch, y: record.trainLoss })) },
              { key: "val", label: "val", color: VAL_COLOR, points: history.map((record) => ({ x: record.epoch, y: record.valLoss })) },
            ]}
            xDomain={xDomain}
            yDomain={[0, Number(maxLoss.toFixed(1))]}
            xLabel="epoch"
            height={170}
          />
          <PipelineChart
            title="Val Dice"
            subtitle={best ? `best ${best.valDice.toFixed(3)} at epoch ${best.epoch} → best checkpoint` : "higher is better"}
            series={[{ key: "dice", label: "val Dice (p > 0.5)", color: DICE_COLOR, points: history.map((record) => ({ x: record.epoch, y: record.valDice })) }]}
            markers={diceMarkers}
            xDomain={xDomain}
            yDomain={[0, 1]}
            xLabel="epoch"
            height={170}
          />
        </div>
      </div>
    </div>
  );
}
