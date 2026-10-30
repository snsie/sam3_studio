import type {
  ChallengeExamplesResponse,
  ChallengeId,
  ChallengeImageListResponse,
  ChallengeScoreRequestPayload,
  ChallengeScoreSummary,
  ChallengeSummary,
  JobStatusResponse,
  LeaderboardResponse,
  NucleiImageListResponse,
  NucleiImageMetricsResponse,
  NucleiMetricsSummary,
  NucleiModelKey,
  NucleiPredictionResponse,
  SegmentationRequest,
  SessionResetResponse,
  UploadResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
const DEFAULT_SEGMENTATION_TIMEOUT_MS = 300_000;

function resolveSegmentationTimeoutMs(value: string | undefined): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_SEGMENTATION_TIMEOUT_MS;
}

const SEGMENTATION_TIMEOUT_MS = resolveSegmentationTimeoutMs(
  import.meta.env.VITE_SEGMENTATION_TIMEOUT_MS,
);

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

async function fetchWithTimeout(
  input: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const externalSignal = init.signal;
  let timedOut = false;

  const handleExternalAbort = () => {
    controller.abort(externalSignal?.reason);
  };

  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort(externalSignal.reason);
    } else {
      externalSignal.addEventListener("abort", handleExternalAbort, { once: true });
    }
  }

  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      if (timedOut) {
        throw new ApiError(
          "Segmentation request timed out while waiting for an available model.",
          504,
        );
      }
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", handleExternalAbort);
  }
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

export async function listSampleImages(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/samples`);
  return parseResponse<string[]>(response);
}

export async function loadSampleImage(filename: string): Promise<UploadResponse> {
  const response = await fetch(`${API_BASE}/samples/${encodeURIComponent(filename)}/load`, {
    method: "POST",
  });

  return parseResponse<UploadResponse>(response);
}

export async function createSegmentationJob(
  payload: SegmentationRequest,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  const response = await fetchWithTimeout(`${API_BASE}/segment/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  }, SEGMENTATION_TIMEOUT_MS);

  return parseResponse<JobStatusResponse>(response);
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

export async function listChallenges(): Promise<ChallengeSummary[]> {
  const response = await fetch(`${API_BASE}/challenges`);
  return parseResponse<ChallengeSummary[]>(response);
}

export async function listChallengeImages(
  challengeId: ChallengeId,
): Promise<ChallengeImageListResponse> {
  const response = await fetch(`${API_BASE}/challenges/${challengeId}/images`);
  return parseResponse<ChallengeImageListResponse>(response);
}

export async function loadChallengeImage(
  challengeId: ChallengeId,
  imageId: string,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  const response = await fetch(
    `${API_BASE}/challenges/${challengeId}/images/${encodeURIComponent(imageId)}/load`,
    {
      method: "POST",
      signal,
    },
  );
  return parseResponse<UploadResponse>(response);
}

export async function listChallengeExamples(
  challengeId: ChallengeId,
): Promise<ChallengeExamplesResponse> {
  const response = await fetch(`${API_BASE}/challenges/${challengeId}/examples`);
  return parseResponse<ChallengeExamplesResponse>(response);
}

export async function scoreChallengeDataset(
  challengeId: ChallengeId,
  payload: ChallengeScoreRequestPayload,
): Promise<ChallengeScoreSummary> {
  const response = await fetchWithTimeout(
    `${API_BASE}/challenges/${challengeId}/score`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    SEGMENTATION_TIMEOUT_MS,
  );
  return parseResponse<ChallengeScoreSummary>(response);
}

export async function listLeaderboardEntries(
  challengeId?: ChallengeId,
): Promise<LeaderboardResponse> {
  const path = challengeId ? `${API_BASE}/leaderboard/${challengeId}` : `${API_BASE}/leaderboard`;
  const response = await fetch(path);
  return parseResponse<LeaderboardResponse>(response);
}

export async function listNucleiImages(): Promise<NucleiImageListResponse> {
  const response = await fetch(`${API_BASE}/nuclei-demo/images`);
  return parseResponse<NucleiImageListResponse>(response);
}

export async function getNucleiMetrics(): Promise<NucleiMetricsSummary> {
  const response = await fetch(`${API_BASE}/nuclei-demo/metrics`);
  return parseResponse<NucleiMetricsSummary>(response);
}

export async function getNucleiImageMetrics(imageId: string): Promise<NucleiImageMetricsResponse> {
  const response = await fetch(`${API_BASE}/nuclei-demo/metrics/${encodeURIComponent(imageId)}`);
  return parseResponse<NucleiImageMetricsResponse>(response);
}

export async function predictNuclei(
  imageId: string,
  modelKey: NucleiModelKey,
  signal?: AbortSignal,
): Promise<NucleiPredictionResponse> {
  const response = await fetchWithTimeout(
    `${API_BASE}/nuclei-demo/images/${encodeURIComponent(imageId)}/predict`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_key: modelKey }),
      signal,
    },
    SEGMENTATION_TIMEOUT_MS,
  );
  return parseResponse<NucleiPredictionResponse>(response);
}

export { ApiError };
