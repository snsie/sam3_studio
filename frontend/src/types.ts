export type JobStatus =
  | "queued"
  | "loading_model"
  | "loading_image_embeddings"
  | "running_segmentation"
  | "packaging_masks"
  | "complete"
  | "failed";

export type InteractionMode =
  | "click"
  | "positive_point"
  | "negative_point"
  | "box"
  | "text";

export type ChallengeId = "single-class" | "multi-class";

export type BrushMode = "off" | "draw" | "erase";

/** One brush segment in image pixel space, from the previous pointer sample to the current one. */
export interface BrushStrokeSegment {
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  draw: boolean;
}

export type PointLabel = "positive" | "negative";

export type PromptPointSource = "click" | "positive_point" | "negative_point";

export interface PromptPoint {
  id: string;
  x: number;
  y: number;
  label: PointLabel;
  source: PromptPointSource;
}

export interface PromptBox {
  id: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  label: PointLabel;
}

export interface SegmentationRequest {
  session_id: string;
  interaction_mode: InteractionMode;
  points: Array<Pick<PromptPoint, "x" | "y" | "label">>;
  boxes: Array<Pick<PromptBox, "x0" | "y0" | "x1" | "y1" | "label">>;
  text_prompt?: string | null;
  multimask_output: boolean;
}

export interface MaskCandidate {
  candidate_id: string;
  score: number | null;
  width: number;
  height: number;
  area: number;
  image_data_url: string;
  color_rgb: number[];
  centroid: number[] | null;
  bounding_box: number[] | null;
}

export interface SegmentationResult {
  session_id: string;
  mode: InteractionMode;
  source: "interactive" | "text_grounding";
  text_prompt_supported: boolean;
  supports_point_refinement: boolean;
  supports_box_refinement: boolean;
  warnings: string[];
  candidates: MaskCandidate[];
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  content_type: string;
  width: number;
  height: number;
  image_url: string;
  ground_truth_overlay_data_url?: string | null;
  created_at: string;
}

export interface SessionInfoResponse extends UploadResponse {
  cache_ready: boolean;
}

export interface SessionResetResponse {
  session_id: string;
  reset: boolean;
}

export interface JobStatusResponse {
  job_id: string;
  session_id: string;
  status: JobStatus;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  result: SegmentationResult | null;
  error: string | null;
}

export interface ChallengeSummary {
  challenge_id: ChallengeId;
  title: string;
  description: string;
  image_count: number;
  example_count: number;
}

export interface ChallengeImageItem {
  image_id: string;
  name: string;
  index: number;
  mask_count: number;
}

export interface ChallengeImageListResponse {
  challenge: ChallengeSummary;
  images: ChallengeImageItem[];
}

export interface ChallengeExampleMask {
  mask_id: string;
  label: string;
  color_rgb: number[];
  image_url: string;
}

export interface ChallengeExample {
  image_id: string;
  title: string;
  image_url: string;
  overlay_data_url: string;
  masks: ChallengeExampleMask[];
}

export interface ChallengeExamplesResponse {
  challenge: ChallengeSummary;
  examples: ChallengeExample[];
}

export interface SubmittedMaskLayerPayload {
  mask_id: string;
  label?: string | null;
  image_data_url: string;
}

export interface SubmittedImageMaskPayload {
  image_id: string;
  masks: SubmittedMaskLayerPayload[];
  width?: number;
  height?: number;
}

export interface ChallengeScoreRequestPayload {
  user_name: string;
  submissions: SubmittedImageMaskPayload[];
}

export interface ChallengeScoreSummary {
  challenge_id: ChallengeId;
  user_name: string;
  dataset_size: number;
  images_labeled: number;
  percent_completed: number;
  mean_iou: number;
  mean_dice: number;
  pixel_accuracy: number;
  labeled_mean_iou: number;
  labeled_mean_dice: number;
  labeled_pixel_accuracy: number;
  overall_score: number;
  submitted_at: string;
  rank: number | null;
  warnings: string[];
}

export interface LeaderboardResponse {
  challenge_id?: ChallengeId | null;
  entries: ChallengeScoreSummary[];
}

export interface EditableMaskLayer {
  mask_id: string;
  label: string;
  color_rgb: [number, number, number];
  data: Uint8ClampedArray;
}

export interface ChallengeImageMaskState {
  image_id: string;
  width: number;
  height: number;
  layers: EditableMaskLayer[];
  active_mask_id: string | null;
  stored: boolean;
  stored_at: string | null;
}

export type NucleiModelKey = "sam3_lora_pseudo" | "sam3_zero_shot" | "sam3_lora_gt";
export type NucleiImageType = "fluorescent" | "brightfield" | "histology";

export interface NucleiImageItem {
  image_id: string;
  image_type: NucleiImageType;
  width: number;
  height: number;
  gt_count: number;
  thumbnail_url: string;
  image_url: string;
}

export interface NucleiModelOption {
  model_key: NucleiModelKey;
  label: string;
  description: string;
  available: boolean;
}

export interface NucleiImageListResponse {
  images: NucleiImageItem[];
  image_type_counts: Record<NucleiImageType, number>;
  models: NucleiModelOption[];
  prompt: string;
}

export interface NucleiImageMetrics {
  dsb_score: number;
  f1_at_50: number;
  precision_at_50: number;
  recall_at_50: number;
  pixel_dice: number;
  pixel_iou: number;
  instance_mean_iou: number;
  gt_count: number;
  pred_count: number;
  count_error: number;
  abs_count_error: number;
  ap_per_threshold: number[];
}

export interface NucleiAggregateMetrics {
  image_count: number;
  dsb_score?: number;
  f1_at_50?: number;
  precision_at_50?: number;
  recall_at_50?: number;
  pixel_dice?: number;
  pixel_iou?: number;
  instance_mean_iou?: number;
  abs_count_error?: number;
  mean_count_error?: number;
  relative_abs_count_error?: number;
  gt_count_total?: number;
  pred_count_total?: number;
  ap_per_threshold?: number[];
}

export interface NucleiModelMetrics {
  label: string;
  description: string;
  lora_checkpoint: string | null;
  aggregate: NucleiAggregateMetrics;
  by_image_type: Record<NucleiImageType, NucleiAggregateMetrics>;
  inference_seconds_mean: number;
}

export interface NucleiEvalScores {
  dsb_score: number;
  f1_at_50: number;
  pixel_dice: number;
  pixel_iou: number;
  abs_count_error: number;
}

export interface NucleiTrainingEpoch {
  epoch: number;
  train_loss: number;
  train_seconds: number;
  epoch_seconds: number;
  val: NucleiEvalScores;
  val_diagnostic?: NucleiEvalScores;
}

export interface NucleiTrainingSummary {
  history: NucleiTrainingEpoch[];
  best_epoch: number | null;
  selection_metric: string | null;
  training_seconds: number | null;
  config: Record<string, string | number | boolean | null>;
}

export interface NucleiPseudoLabelSummary {
  quality: {
    overall: NucleiAggregateMetrics;
    by_image_type: Record<string, NucleiAggregateMetrics>;
    by_split: Record<string, NucleiAggregateMetrics>;
  };
  annotation: {
    prompt: string;
    confidence_threshold: number;
    checkpoint_path: string | null;
    device: string;
    gpu: string | null;
    images: number;
    annotations: number;
    images_without_detections: number;
    images_at_query_cap: number;
    timing: {
      inference_seconds_total: number;
      inference_seconds_mean: number;
      inference_seconds_median: number;
      wall_seconds_total: number;
    };
  };
}

export interface NucleiMetricsSummary {
  generated_at: string;
  prompt: string;
  confidence_threshold: number;
  test_set: {
    name: string;
    ground_truth: string;
    images: number;
    nuclei: number;
    image_types: Record<NucleiImageType, number>;
  };
  split_counts: Record<"train" | "val" | "test", Record<NucleiImageType, number>>;
  metric_definitions: Record<string, string>;
  iou_thresholds: number[];
  models: Partial<Record<NucleiModelKey, NucleiModelMetrics>>;
  training: Partial<Record<NucleiModelKey, NucleiTrainingSummary | null>>;
  pseudo_labels: NucleiPseudoLabelSummary | null;
}

export interface NucleiImageMetricsResponse {
  image_id: string;
  image_type: NucleiImageType;
  gt_count: number;
  models: Partial<Record<NucleiModelKey, NucleiImageMetrics & { inference_seconds: number }>>;
}

export interface NucleiInstance {
  instance_id: number;
  status: "tp" | "fp" | "fn" | "matched";
  area: number;
  bounding_box: number[];
  score: number | null;
  matched_id: number | null;
  iou: number | null;
}

export interface NucleiPredictionResponse {
  image_id: string;
  image_type: NucleiImageType;
  width: number;
  height: number;
  model_key: NucleiModelKey;
  model_label: string;
  prompt: string;
  job_id: string;
  elapsed_seconds: number;
  metrics: NucleiImageMetrics;
  counts: { tp: number; fp: number; fn: number; pred: number; gt: number };
  overlays: {
    prediction: string;
    ground_truth: string;
    ground_truth_outline: string;
    prediction_outline: string;
    comparison: string;
  };
  predictions: NucleiInstance[];
  ground_truth: NucleiInstance[];
  warnings: string[];
}
