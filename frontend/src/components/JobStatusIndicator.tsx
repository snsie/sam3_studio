import type { JobStatusResponse } from "../types";

const LABELS: Record<JobStatusResponse["status"], string> = {
  queued: "Queued",
  loading_model: "Loading model",
  loading_image_embeddings: "Computing image embeddings",
  running_segmentation: "Running segmentation",
  packaging_masks: "Packaging masks",
  complete: "Complete",
  failed: "Failed",
};

interface JobStatusIndicatorProps {
  job: JobStatusResponse | null;
}

export function JobStatusIndicator({ job }: JobStatusIndicatorProps) {
  if (!job) {
    return (
      <div className="status-shell idle">
        <div>
          <span className="status-label">Idle</span>
          <p>Upload an image and submit prompts to start a local SAM3 job.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`status-shell ${job.status}`}>
      <div className="status-copy">
        <span className="status-label">{LABELS[job.status]}</span>
        <p>{job.error ?? job.message}</p>
      </div>
      <div className="progress-track" aria-hidden="true">
        <span
          className="progress-fill"
          style={{ width: `${Math.max(8, job.progress * 100)}%` }}
        />
      </div>
    </div>
  );
}
