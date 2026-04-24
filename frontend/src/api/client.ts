import type {
  JobStatusResponse,
  JobSubmissionResponse,
  SegmentationRequest,
  SessionResetResponse,
  UploadResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) {
    return (await response.json()) as T;
  }

  let message = response.statusText;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) {
      message = payload.detail;
    }
  } catch {
    message = response.statusText;
  }

  throw new ApiError(message || "Request failed", response.status);
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  return parseResponse<UploadResponse>(response);
}

export async function createSegmentationJob(
  payload: SegmentationRequest,
): Promise<JobSubmissionResponse> {
  const response = await fetch(`${API_BASE}/segment/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse<JobSubmissionResponse>(response);
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  return parseResponse<JobStatusResponse>(response);
}

export async function resetSessionCache(
  sessionId: string,
): Promise<SessionResetResponse> {
  const response = await fetch(`${API_BASE}/sessions/${sessionId}/reset-cache`, {
    method: "POST",
  });
  return parseResponse<SessionResetResponse>(response);
}

export { ApiError };
