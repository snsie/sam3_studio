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
