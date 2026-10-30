import { useEffect, useMemo, useRef, useState } from "react";

import {
  getNucleiImageMetrics,
  getNucleiMetrics,
  listNucleiImages,
  predictNuclei,
} from "../api/client";
import { NucleiLineChart, type LineChartSeries } from "../components/NucleiLineChart";
import type {
  NucleiAggregateMetrics,
  NucleiImageListResponse,
  NucleiImageMetricsResponse,
  NucleiImageType,
  NucleiMetricsSummary,
  NucleiModelKey,
  NucleiPredictionResponse,
} from "../types";

type TypeFilter = "all" | NucleiImageType;
type ViewMode = "prediction" | "ground_truth" | "both" | "comparison";

// Categorical slots 1-3 of the chart palette, stepped for the dark surface.
const MODEL_ORDER: NucleiModelKey[] = ["sam3_lora_pseudo", "sam3_zero_shot", "sam3_lora_gt"];
const MODEL_COLORS: Record<NucleiModelKey, string> = {
  sam3_lora_pseudo: "#3987e5",
  sam3_zero_shot: "#d95926",
  sam3_lora_gt: "#199e70",
};
const MODEL_SHORT_LABELS: Record<NucleiModelKey, string> = {
  sam3_lora_pseudo: "Trained (pseudo-labels)",
  sam3_zero_shot: "Zero-shot SAM3",
  sam3_lora_gt: "Trained (ground truth)",
};

const IMAGE_TYPES: NucleiImageType[] = ["fluorescent", "histology", "brightfield"];
const TYPE_LABELS: Record<TypeFilter, string> = {
  all: "All types",
  fluorescent: "Fluorescent",
  histology: "Histology",
  brightfield: "Brightfield",
};

const VIEW_MODES: Array<{ id: ViewMode; label: string }> = [
  { id: "prediction", label: "Prediction" },
  { id: "ground_truth", label: "Ground truth" },
  { id: "both", label: "Both" },
  { id: "comparison", label: "TP / FP / FN" },
];

const METRIC_COLUMNS: Array<{
  key: keyof NucleiAggregateMetrics;
  label: string;
  format: (value: number) => string;
}> = [
  { key: "dsb_score", label: "DSB score", format: (value) => value.toFixed(3) },
  { key: "f1_at_50", label: "F1@0.5", format: (value) => value.toFixed(3) },
  { key: "pixel_dice", label: "Pixel Dice", format: (value) => value.toFixed(3) },
  { key: "pixel_iou", label: "Pixel IoU", format: (value) => value.toFixed(3) },
  { key: "abs_count_error", label: "|Count error|", format: (value) => value.toFixed(1) },
  { key: "mean_count_error", label: "Count bias", format: formatSigned },
];

function formatSigned(value: number) {
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function formatScore(value: number | undefined | null) {
  return value === undefined || value === null ? "–" : value.toFixed(3);
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) {
    return "–";
  }
  if (seconds < 120) {
    return `${seconds.toFixed(0)} s`;
  }
  const minutes = seconds / 60;
  return minutes < 120 ? `${minutes.toFixed(0)} min` : `${(minutes / 60).toFixed(1)} h`;
}

function aggregateFor(
  metrics: NucleiMetricsSummary,
  modelKey: NucleiModelKey,
  typeFilter: TypeFilter,
): NucleiAggregateMetrics | null {
  const model = metrics.models[modelKey];
  if (!model) {
    return null;
  }
  return typeFilter === "all" ? model.aggregate : model.by_image_type[typeFilter] ?? null;
}

function PipelineCard({ metrics }: { metrics: NucleiMetricsSummary | null }) {
  const annotation = metrics?.pseudo_labels?.annotation;
  const training = metrics?.training.sam3_lora_pseudo;
  const steps = [
    {
      kicker: "1 · Auto-annotate",
      title: `SAM3 + "${metrics?.prompt ?? "blobs"}"`,
      note: annotation
        ? `${annotation.images} stage1_train images in ${formatDuration(annotation.timing.wall_seconds_total)} (${annotation.timing.inference_seconds_mean.toFixed(2)} s/image)`
        : "Text-prompt grounding on every stage1_train image",
    },
    {
      kicker: "2 · Pseudo-labels",
      title: annotation ? `${annotation.annotations.toLocaleString()} nuclei` : "COCO RLE masks",
      note: metrics?.pseudo_labels
        ? `F1@0.5 ${formatScore(metrics.pseudo_labels.quality.overall.f1_at_50)} vs. hidden ground truth`
        : "No ground truth used",
    },
    {
      kicker: "3 · Train",
      title: "SAM3 + LoRA",
      note: training
        ? `${training.config.train_image_count} images · best epoch ${training.best_epoch} of ${training.history.length} · ${formatDuration(training.training_seconds)}`
        : "Fine-tuned on pseudo-labels only",
    },
    {
      kicker: "4 · Score",
      title: "DSB stage1_test",
      note: metrics
        ? `${metrics.test_set.images} held-out images · ${metrics.test_set.nuclei.toLocaleString()} nuclei`
        : "Official DSB 2018 metric",
    },
  ];

  return (
    <section className="panel-block page-intro-card nuclei-pipeline-card">
      <div className="section-heading compact-heading">
        <span className="section-kicker">BBBC038 · 2018 Data Science Bowl</span>
        <h2>Label nuclei without labels</h2>
      </div>
      <p className="placeholder-copy">
        SAM3 annotates the training images from a one-word text prompt, a model is fine-tuned on those
        pseudo-labels alone, and the result is scored against the official stage1 test ground truth. The
        test set is never used for training, model selection or threshold tuning.
      </p>
      <ol className="nuclei-pipeline-steps">
        {steps.map((step) => (
          <li key={step.kicker}>
            <span className="section-kicker">{step.kicker}</span>
            <strong>{step.title}</strong>
            <span className="field-note">{step.note}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function ModelSwatch({ modelKey }: { modelKey: NucleiModelKey }) {
  return <span className="nuclei-model-swatch" style={{ background: MODEL_COLORS[modelKey] }} aria-hidden="true" />;
}

function MetricsComparison({ metrics }: { metrics: NucleiMetricsSummary }) {
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const availableModels = MODEL_ORDER.filter((key) => metrics.models[key]);
  const imageCount = typeFilter === "all" ? metrics.test_set.images : metrics.test_set.image_types[typeFilter];

  const apSeries: LineChartSeries[] = availableModels
    .map((key): LineChartSeries | null => {
      const aggregate = aggregateFor(metrics, key, typeFilter);
      return aggregate?.ap_per_threshold
        ? { key, label: MODEL_SHORT_LABELS[key], color: MODEL_COLORS[key], values: aggregate.ap_per_threshold }
        : null;
    })
    .filter((item): item is LineChartSeries => item !== null);

  return (
    <section className="panel-block nuclei-section">
      <div className="nuclei-section-header">
        <div className="section-heading compact-heading">
          <span className="section-kicker">Test results</span>
          <h2>stage1_test scores</h2>
        </div>
        <div className="inline-toggle-group">
          {(["all", ...IMAGE_TYPES] as TypeFilter[]).map((option) => {
            const count = option === "all" ? metrics.test_set.images : metrics.test_set.image_types[option];
            return (
              <button
                key={option}
                type="button"
                className={`inline-toggle-button${typeFilter === option ? " is-active" : ""}`}
                onClick={() => setTypeFilter(option)}
                disabled={count === 0}
                title={count === 0 ? "stage1_test has no images of this type" : undefined}
              >
                {`${TYPE_LABELS[option]} (${count})`}
              </button>
            );
          })}
        </div>
      </div>

      <div className="leaderboard-table-wrapper nuclei-table-wrapper">
        <table className="leaderboard-table nuclei-metrics-table">
          <thead>
            <tr>
              <th scope="col">Model</th>
              {METRIC_COLUMNS.map((column) => (
                <th key={column.key} scope="col">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODEL_ORDER.map((key) => {
              const aggregate = aggregateFor(metrics, key, typeFilter);
              return (
                <tr key={key} className={key === "sam3_lora_pseudo" ? "is-highlighted" : undefined}>
                  <td>
                    <span className="nuclei-model-cell">
                      <ModelSwatch modelKey={key} />
                      {MODEL_SHORT_LABELS[key]}
                      {key === "sam3_lora_gt" ? <em>upper bound</em> : null}
                    </span>
                  </td>
                  {METRIC_COLUMNS.map((column) => {
                    const value = aggregate?.[column.key];
                    return (
                      <td key={column.key}>
                        {typeof value === "number" ? column.format(value) : aggregate ? "–" : "not trained"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="field-note nuclei-footnote">
        {`${imageCount} images. DSB score = mean over IoU thresholds 0.50–0.95 of TP / (TP + FP + FN), averaged per image. `}
        {`Count bias is predicted minus true nuclei per image. Prompt "${metrics.prompt}", confidence ≥ ${metrics.confidence_threshold}.`}
      </p>

      {apSeries.length > 0 ? (
        <NucleiLineChart
          title="Precision vs. IoU threshold"
          subtitle="The DSB score is the mean of each line"
          xValues={metrics.iou_thresholds}
          series={apSeries}
          formatX={(value) => value.toFixed(2)}
          formatY={(value) => value.toFixed(2)}
          yDomain={[0, 1]}
          xLabel="IoU"
        />
      ) : null}
    </section>
  );
}

function TrainingCurves({ metrics }: { metrics: NucleiMetricsSummary }) {
  const pseudo = metrics.training.sam3_lora_pseudo;
  const groundTruth = metrics.training.sam3_lora_gt;
  const runs = [pseudo, groundTruth].filter((run) => run && run.history.length > 0);
  if (runs.length === 0) {
    return (
      <section className="panel-block nuclei-section">
        <div className="section-heading compact-heading">
          <span className="section-kicker">Training</span>
          <h2>Training curves</h2>
        </div>
        <p className="field-note">No training history has been recorded yet.</p>
      </section>
    );
  }

  const epochCount = Math.max(...runs.map((run) => run?.history.length ?? 0));
  const epochs = Array.from({ length: epochCount }, (_, index) => index + 1);
  const valuesFor = (run: typeof pseudo, pick: (epoch: NonNullable<typeof pseudo>["history"][number]) => number | undefined) =>
    epochs.map((epoch) => {
      const record = run?.history.find((item) => item.epoch === epoch);
      const value = record ? pick(record) : undefined;
      return value === undefined ? null : value;
    });
  const bestIndex = (run: typeof pseudo) => (run?.best_epoch ? run.best_epoch - 1 : null);

  const validationSeries: LineChartSeries[] = [];
  const lossSeries: LineChartSeries[] = [];
  if (pseudo && pseudo.history.length > 0) {
    validationSeries.push({
      key: "pseudo-val",
      label: "Pseudo-label model · val pseudo-labels (selection)",
      color: MODEL_COLORS.sam3_lora_pseudo,
      values: valuesFor(pseudo, (record) => record.val.dsb_score),
      markIndex: bestIndex(pseudo),
    });
    if (pseudo.history.some((record) => record.val_diagnostic)) {
      validationSeries.push({
        key: "pseudo-val-gt",
        label: "Pseudo-label model · val ground truth (not used)",
        color: MODEL_COLORS.sam3_lora_pseudo,
        values: valuesFor(pseudo, (record) => record.val_diagnostic?.dsb_score),
        dashed: true,
      });
    }
    lossSeries.push({
      key: "pseudo-loss",
      label: "Pseudo-label model",
      color: MODEL_COLORS.sam3_lora_pseudo,
      values: valuesFor(pseudo, (record) => record.train_loss),
    });
  }
  if (groundTruth && groundTruth.history.length > 0) {
    validationSeries.push({
      key: "gt-val",
      label: "Ground-truth model · val ground truth",
      color: MODEL_COLORS.sam3_lora_gt,
      values: valuesFor(groundTruth, (record) => record.val.dsb_score),
      markIndex: bestIndex(groundTruth),
    });
    lossSeries.push({
      key: "gt-loss",
      label: "Ground-truth model",
      color: MODEL_COLORS.sam3_lora_gt,
      values: valuesFor(groundTruth, (record) => record.train_loss),
    });
  }

  const config = pseudo?.config ?? groundTruth?.config;
  return (
    <section className="panel-block nuclei-section">
      <div className="section-heading compact-heading">
        <span className="section-kicker">Training</span>
        <h2>Training curves</h2>
      </div>
      <div className="nuclei-chart-grid">
        <NucleiLineChart
          title="Validation DSB score"
          subtitle="Filled marker = selected checkpoint"
          xValues={epochs}
          series={validationSeries}
          formatX={(value) => String(value)}
          formatY={(value) => value.toFixed(3)}
          yDomain={[0, 1]}
          xLabel="Epoch"
        />
        <NucleiLineChart
          title="Training loss"
          subtitle="SAM3 detection + mask loss, mean per step"
          xValues={epochs}
          series={lossSeries}
          formatX={(value) => String(value)}
          formatY={(value) => value.toFixed(0)}
          xLabel="Epoch"
        />
      </div>
      {config ? (
        <p className="field-note nuclei-footnote">
          {`LoRA rank ${config.lora_rank} on ${config.lora_module_count} linear layers (${Number(config.trainable_parameters).toLocaleString()} of ${Number(config.total_parameters).toLocaleString()} parameters trainable), `}
          {`lr ${config.learning_rate}, ${config.resolution}px, horizontal flips${config.gpu ? `, ${config.gpu}` : ""}.`}
        </p>
      ) : null}
    </section>
  );
}

function PseudoLabelQuality({ metrics }: { metrics: NucleiMetricsSummary }) {
  const pseudo = metrics.pseudo_labels;
  if (!pseudo) {
    return null;
  }
  const overall = pseudo.quality.overall;
  const tiles = [
    { label: "DSB score", value: formatScore(overall.dsb_score) },
    { label: "F1@0.5", value: formatScore(overall.f1_at_50) },
    { label: "Mean instance IoU", value: formatScore(overall.instance_mean_iou) },
    { label: "Pixel IoU", value: formatScore(overall.pixel_iou) },
  ];
  return (
    <section className="panel-block nuclei-section">
      <div className="section-heading compact-heading">
        <span className="section-kicker">Pseudo-labels</span>
        <h2>SAM3 annotation quality on stage1_train</h2>
      </div>
      <div className="nuclei-stat-row">
        {tiles.map((tile) => (
          <div key={tile.label} className="nuclei-stat-tile">
            <span>{tile.label}</span>
            <strong>{tile.value}</strong>
          </div>
        ))}
      </div>
      <div className="leaderboard-table-wrapper nuclei-table-wrapper">
        <table className="leaderboard-table nuclei-metrics-table">
          <thead>
            <tr>
              <th scope="col">Image type</th>
              <th scope="col">Images</th>
              <th scope="col">DSB score</th>
              <th scope="col">F1@0.5</th>
              <th scope="col">Mean instance IoU</th>
              <th scope="col">True nuclei</th>
              <th scope="col">Pseudo nuclei</th>
            </tr>
          </thead>
          <tbody>
            {IMAGE_TYPES.map((type) => {
              const row = pseudo.quality.by_image_type[type];
              if (!row) {
                return null;
              }
              return (
                <tr key={type}>
                  <td>{TYPE_LABELS[type]}</td>
                  <td>{row.image_count}</td>
                  <td>{formatScore(row.dsb_score)}</td>
                  <td>{formatScore(row.f1_at_50)}</td>
                  <td>{formatScore(row.instance_mean_iou)}</td>
                  <td>{row.gt_count_total?.toLocaleString()}</td>
                  <td>{row.pred_count_total?.toLocaleString()}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="field-note nuclei-footnote">
        {`Ground truth is used only to report this table. ${pseudo.annotation.images_without_detections} of ${pseudo.annotation.images} images got no detections at confidence ≥ ${pseudo.annotation.confidence_threshold} and were left out of training. `}
        {`Annotation took ${formatDuration(pseudo.annotation.timing.inference_seconds_total)} of inference (${formatDuration(pseudo.annotation.timing.wall_seconds_total)} wall, including model load) on ${pseudo.annotation.gpu ?? pseudo.annotation.device}.`}
      </p>
    </section>
  );
}

function overlayLayers(prediction: NucleiPredictionResponse, viewMode: ViewMode): string[] {
  switch (viewMode) {
    case "prediction":
      return [prediction.overlays.prediction];
    case "ground_truth":
      return [prediction.overlays.ground_truth];
    case "both":
      return [prediction.overlays.prediction, prediction.overlays.ground_truth_outline];
    case "comparison":
      return [prediction.overlays.comparison];
  }
}

function NucleiExplorer({
  catalog,
  metrics,
}: {
  catalog: NucleiImageListResponse;
  metrics: NucleiMetricsSummary | null;
}) {
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const [selectedId, setSelectedId] = useState<string>(catalog.images[0]?.image_id ?? "");
  const defaultModel = catalog.models.find((model) => model.model_key === "sam3_lora_pseudo" && model.available)
    ? "sam3_lora_pseudo"
    : "sam3_zero_shot";
  const [modelKey, setModelKey] = useState<NucleiModelKey>(defaultModel);
  const [viewMode, setViewMode] = useState<ViewMode>("comparison");
  const [opacity, setOpacity] = useState(0.7);
  const [prediction, setPrediction] = useState<NucleiPredictionResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [imageMetrics, setImageMetrics] = useState<NucleiImageMetricsResponse | null>(null);
  const [runToken, setRunToken] = useState(0);
  const abortRef = useRef<AbortController | null>(null);

  const visibleImages = useMemo(
    () => catalog.images.filter((image) => typeFilter === "all" || image.image_type === typeFilter),
    [catalog.images, typeFilter],
  );
  const selectedImage = catalog.images.find((image) => image.image_id === selectedId) ?? null;

  useEffect(() => {
    if (!selectedId) {
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setIsRunning(true);
    setErrorMessage(null);

    predictNuclei(selectedId, modelKey, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) {
          setPrediction(response);
        }
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setErrorMessage(error instanceof Error ? error.message : "Inference failed.");
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsRunning(false);
        }
      });

    return () => controller.abort();
  }, [modelKey, runToken, selectedId]);

  useEffect(() => {
    if (!selectedId || !metrics) {
      setImageMetrics(null);
      return;
    }
    let isCurrent = true;
    getNucleiImageMetrics(selectedId)
      .then((response) => {
        if (isCurrent) {
          setImageMetrics(response);
        }
      })
      .catch(() => {
        if (isCurrent) {
          setImageMetrics(null);
        }
      });
    return () => {
      isCurrent = false;
    };
  }, [metrics, selectedId]);

  const currentPrediction =
    prediction && prediction.image_id === selectedId && prediction.model_key === modelKey ? prediction : null;

  return (
    <section className="panel-block nuclei-section">
      <div className="nuclei-section-header">
        <div className="section-heading compact-heading">
          <span className="section-kicker">Explore</span>
          <h2>Run a model on a test image</h2>
        </div>
        <div className="inline-toggle-group">
          {(["all", ...IMAGE_TYPES] as TypeFilter[]).map((option) => {
            const count =
              option === "all" ? catalog.images.length : catalog.image_type_counts[option] ?? 0;
            return (
              <button
                key={option}
                type="button"
                className={`inline-toggle-button${typeFilter === option ? " is-active" : ""}`}
                onClick={() => setTypeFilter(option)}
                disabled={count === 0}
              >
                {`${TYPE_LABELS[option]} (${count})`}
              </button>
            );
          })}
        </div>
      </div>

      <div className="nuclei-explorer">
        <div className="nuclei-image-grid" role="listbox" aria-label="stage1_test images">
          {visibleImages.map((image) => (
            <button
              key={image.image_id}
              type="button"
              role="option"
              aria-selected={image.image_id === selectedId}
              className={`nuclei-thumb${image.image_id === selectedId ? " is-active" : ""}`}
              onClick={() => setSelectedId(image.image_id)}
              title={`${image.image_id} · ${image.width}×${image.height}`}
            >
              <img src={image.thumbnail_url} alt="" loading="lazy" />
              <span className="nuclei-thumb-meta">
                <span>{TYPE_LABELS[image.image_type]}</span>
                <span>{`${image.gt_count} nuclei`}</span>
              </span>
            </button>
          ))}
        </div>

        <div className="nuclei-viewer">
          <div className="nuclei-viewer-controls">
            <div className="inline-toggle-group" role="group" aria-label="Model">
              {catalog.models.map((model) => (
                <button
                  key={model.model_key}
                  type="button"
                  className={`inline-toggle-button nuclei-model-toggle${modelKey === model.model_key ? " is-active" : ""}`}
                  onClick={() => setModelKey(model.model_key)}
                  disabled={!model.available}
                  title={model.available ? model.description : "This model has not been trained yet."}
                >
                  <ModelSwatch modelKey={model.model_key} />
                  {MODEL_SHORT_LABELS[model.model_key]}
                </button>
              ))}
            </div>
            <div className="inline-toggle-group" role="group" aria-label="Overlay">
              {VIEW_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`inline-toggle-button${viewMode === mode.id ? " is-active" : ""}`}
                  onClick={() => setViewMode(mode.id)}
                >
                  {mode.label}
                </button>
              ))}
            </div>
            <label className="nuclei-opacity-control">
              <span className="field-note">{`Overlay opacity ${Math.round(opacity * 100)}%`}</span>
              <input
                className="range-input"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
              />
            </label>
            <button
              type="button"
              className="secondary-button compact-button"
              onClick={() => setRunToken((value) => value + 1)}
              disabled={isRunning || !selectedId}
            >
              {isRunning ? "Running…" : "Run again"}
            </button>
          </div>

          {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}

          {selectedImage ? (
            <div className="nuclei-stage">
              <div
                className="nuclei-stage-frame"
                style={{ aspectRatio: `${selectedImage.width} / ${selectedImage.height}` }}
              >
                <img className="nuclei-stage-image" src={selectedImage.image_url} alt={`Test image ${selectedImage.image_id}`} />
                {currentPrediction
                  ? overlayLayers(currentPrediction, viewMode).map((layer, index) => (
                      <img key={index} className="nuclei-stage-image" src={layer} alt="" style={{ opacity }} />
                    ))
                  : null}
                {isRunning ? <div className="nuclei-stage-status">{`Running ${MODEL_SHORT_LABELS[modelKey]}…`}</div> : null}
              </div>
              {viewMode === "comparison" ? (
                <ul className="nuclei-comparison-legend">
                  <li>
                    <span className="nuclei-legend-chip is-tp" aria-hidden="true">✓</span>
                    {`True positive${currentPrediction ? ` (${currentPrediction.counts.tp})` : ""}`}
                  </li>
                  <li>
                    <span className="nuclei-legend-chip is-fp" aria-hidden="true">+</span>
                    {`False positive${currentPrediction ? ` (${currentPrediction.counts.fp})` : ""}`}
                  </li>
                  <li>
                    <span className="nuclei-legend-chip is-fn" aria-hidden="true">−</span>
                    {`Missed nucleus${currentPrediction ? ` (${currentPrediction.counts.fn})` : ""}`}
                  </li>
                  <li className="field-note">Matched at IoU &gt; 0.5</li>
                </ul>
              ) : viewMode === "both" ? (
                <ul className="nuclei-comparison-legend">
                  <li className="field-note">Fill = prediction (one color per nucleus), white outline = ground truth</li>
                </ul>
              ) : null}
            </div>
          ) : (
            <p className="field-note">Pick an image to run inference.</p>
          )}
        </div>

        <aside className="nuclei-image-metrics">
          <div className="section-heading compact-heading">
            <span className="section-kicker">This image</span>
            <h3>{selectedImage ? TYPE_LABELS[selectedImage.image_type] : "–"}</h3>
            {selectedImage ? (
              <span className="field-note nuclei-image-id">{`${selectedImage.image_id.slice(0, 16)}… · ${selectedImage.width}×${selectedImage.height}`}</span>
            ) : null}
          </div>
          {currentPrediction ? (
            <>
              <div className="nuclei-stat-row is-compact">
                <div className="nuclei-stat-tile">
                  <span>DSB score</span>
                  <strong>{currentPrediction.metrics.dsb_score.toFixed(3)}</strong>
                </div>
                <div className="nuclei-stat-tile">
                  <span>F1@0.5</span>
                  <strong>{currentPrediction.metrics.f1_at_50.toFixed(3)}</strong>
                </div>
                <div className="nuclei-stat-tile">
                  <span>Pixel Dice</span>
                  <strong>{currentPrediction.metrics.pixel_dice.toFixed(3)}</strong>
                </div>
                <div className="nuclei-stat-tile">
                  <span>Count</span>
                  <strong>{`${currentPrediction.counts.pred} / ${currentPrediction.counts.gt}`}</strong>
                </div>
              </div>
              <p className="field-note">
                {`${currentPrediction.model_label}: ${currentPrediction.counts.pred} predicted vs. ${currentPrediction.counts.gt} true nuclei, pixel IoU ${currentPrediction.metrics.pixel_iou.toFixed(3)}, ${currentPrediction.elapsed_seconds.toFixed(1)} s round trip.`}
              </p>
              {currentPrediction.warnings.length > 0 ? (
                <p className="field-note">{currentPrediction.warnings.join(" ")}</p>
              ) : null}
            </>
          ) : (
            <p className="field-note">{isRunning ? "Waiting for the model…" : "No prediction yet."}</p>
          )}

          {imageMetrics ? (
            <table className="nuclei-mini-table">
              <caption className="field-note">Precomputed for all models</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">DSB</th>
                  <th scope="col">F1@0.5</th>
                  <th scope="col">Count</th>
                </tr>
              </thead>
              <tbody>
                {MODEL_ORDER.map((key) => {
                  const row = imageMetrics.models[key];
                  return (
                    <tr key={key} className={key === modelKey ? "is-active" : undefined}>
                      <td>
                        <span className="nuclei-model-cell">
                          <ModelSwatch modelKey={key} />
                          {MODEL_SHORT_LABELS[key]}
                        </span>
                      </td>
                      <td>{row ? row.dsb_score.toFixed(3) : "–"}</td>
                      <td>{row ? row.f1_at_50.toFixed(3) : "–"}</td>
                      <td>{row ? `${row.pred_count} / ${row.gt_count}` : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : null}
        </aside>
      </div>
    </section>
  );
}

export function NucleiDemoPage({ isActive }: { isActive: boolean }) {
  const [catalog, setCatalog] = useState<NucleiImageListResponse | null>(null);
  const [metrics, setMetrics] = useState<NucleiMetricsSummary | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);
  const [hasLoaded, setHasLoaded] = useState(false);

  useEffect(() => {
    if (!isActive || hasLoaded) {
      return;
    }
    setHasLoaded(true);
    listNucleiImages()
      .then(setCatalog)
      .catch((error: unknown) =>
        setErrorMessage(error instanceof Error ? error.message : "Failed to load stage1_test images."),
      );
    getNucleiMetrics()
      .then(setMetrics)
      .catch((error: unknown) =>
        setMetricsError(error instanceof Error ? error.message : "Failed to load metrics."),
      );
  }, [hasLoaded, isActive]);

  return (
    <section className="page-content-stack nuclei-page">
      <PipelineCard metrics={metrics} />
      {metricsError ? <div className="error-banner">{metricsError}</div> : null}
      {metrics ? (
        <>
          <MetricsComparison metrics={metrics} />
          <div className="nuclei-two-column">
            <TrainingCurves metrics={metrics} />
            <PseudoLabelQuality metrics={metrics} />
          </div>
        </>
      ) : null}
      {errorMessage ? <div className="error-banner">{errorMessage}</div> : null}
      {catalog ? <NucleiExplorer catalog={catalog} metrics={metrics} /> : null}
      {!catalog && !errorMessage ? <div className="panel-block">Loading stage1_test images…</div> : null}
    </section>
  );
}
