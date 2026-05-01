import { startTransition, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import {
  createSegmentationJob,
  listSampleImages,
  loadSampleImage,
  resetSessionCache,
  uploadImage,
} from "../api/client";
import { JobStatusIndicator } from "../components/JobStatusIndicator";
import { PromptToolbar } from "../components/PromptToolbar";
import { ResultsPanel } from "../components/ResultsPanel";
import { SamplesPanel } from "../components/SamplesPanel";
import { SegmentationCanvas } from "../components/SegmentationCanvas";
import { TextPromptInput } from "../components/TextPromptInput";
import type { SandboxHeaderState } from "../headerContext";
import type {
  InteractionMode,
  JobStatusResponse,
  PromptBox,
  PromptPoint,
  SegmentationRequest,
  SegmentationResult,
  UploadResponse,
} from "../types";

const DEFAULT_CONFIDENCE_THRESHOLD = 0.5;

function resolveInteractionMode(
  mode: InteractionMode,
  points: PromptPoint[],
  boxes: PromptBox[],
  textPrompt: string,
): InteractionMode {
  const hasText = Boolean(textPrompt.trim());
  const hasInteractivePrompts = points.length > 0 || boxes.length > 0;

  if (hasText && !hasInteractivePrompts) {
    return "text";
  }

  return mode === "text" ? "click" : mode;
}

function shouldUseMultimaskOutput(
  mode: InteractionMode,
  points: PromptPoint[],
  boxes: PromptBox[],
  textPrompt: string,
) {
  const resolvedMode = resolveInteractionMode(mode, points, boxes, textPrompt);

  if (textPrompt.trim() || boxes.length > 0 || points.length !== 1) {
    return false;
  }

  const [point] = points;
  return Boolean(
    point &&
      point.label === "positive" &&
      (resolvedMode === "click" || resolvedMode === "positive_point"),
  );
}

function promptsToRequest(
  session: UploadResponse,
  mode: InteractionMode,
  points: PromptPoint[],
  boxes: PromptBox[],
  textPrompt: string,
): SegmentationRequest {
  const normalizedTextPrompt = textPrompt.trim();
  return {
    session_id: session.session_id,
    interaction_mode: resolveInteractionMode(mode, points, boxes, textPrompt),
    points: points.map(({ x, y, label }) => ({ x, y, label })),
    boxes: boxes.map(({ x0, y0, x1, y1, label }) => ({ x0, y0, x1, y1, label })),
    text_prompt: normalizedTextPrompt || null,
    multimask_output: shouldUseMultimaskOutput(mode, points, boxes, textPrompt),
  };
}

function isAccumulatedPointMode(mode: InteractionMode) {
  return mode === "positive_point" || mode === "negative_point";
}

function createPendingJobStatus(sessionId: string): JobStatusResponse {
  const now = new Date().toISOString();
  return {
    job_id: `pending-${Date.now()}`,
    session_id: sessionId,
    status: "queued",
    progress: 0.05,
    message: "Waiting for available model...",
    created_at: now,
    updated_at: now,
    result: null,
    error: null,
  };
}

interface SandboxPageProps {
  onHeaderStateChange?: (state: SandboxHeaderState | null) => void;
}

export function SandboxPage({ onHeaderStateChange }: SandboxPageProps) {
  const [session, setSession] = useState<UploadResponse | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("click");
  const [textPrompt, setTextPrompt] = useState("");
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [boxes, setBoxes] = useState<PromptBox[]>([]);
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(DEFAULT_CONFIDENCE_THRESHOLD);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [samples, setSamples] = useState<string[]>([]);
  const [samplesErrorMessage, setSamplesErrorMessage] = useState<string | null>(null);
  const [isLoadingSamples, setIsLoadingSamples] = useState(true);
  const hasAttemptedInitialSampleLoad = useRef(false);

  const deferredResult = useDeferredValue(result);

  useEffect(() => {
    let isCurrent = true;

    const loadSamples = async () => {
      try {
        const nextSamples = await listSampleImages();
        if (!isCurrent) {
          return;
        }
        setSamples(nextSamples);
        setSamplesErrorMessage(null);
      } catch (error) {
        if (!isCurrent) {
          return;
        }
        setSamplesErrorMessage(
          error instanceof Error ? error.message : "Unable to load sample images.",
        );
      } finally {
        if (isCurrent) {
          setIsLoadingSamples(false);
        }
      }
    };

    void loadSamples();

    return () => {
      isCurrent = false;
    };
  }, []);

  const visibleCandidates = useMemo(() => {
    if (!deferredResult || deferredResult.candidates.length === 0) {
      return [];
    }

    return deferredResult.candidates.filter(
      (candidate) => typeof candidate.score === "number" && candidate.score > confidenceThreshold,
    );
  }, [confidenceThreshold, deferredResult]);

  useEffect(() => {
    if (selectedCandidateId === null) {
      return;
    }

    const hasSelectedCandidate = visibleCandidates.some(
      (candidate) => candidate.candidate_id === selectedCandidateId,
    );
    if (!hasSelectedCandidate) {
      setSelectedCandidateId(null);
    }
  }, [selectedCandidateId, visibleCandidates]);

  const visibleMasks = useMemo(() => {
    if (visibleCandidates.length === 0) {
      return [];
    }
    if (selectedCandidateId === null) {
      return visibleCandidates;
    }
    const selectedCandidate = visibleCandidates.find(
      (candidate) => candidate.candidate_id === selectedCandidateId,
    );
    return selectedCandidate ? [selectedCandidate] : visibleCandidates;
  }, [selectedCandidateId, visibleCandidates]);

  const applyUploadedSession = useCallback((uploaded: UploadResponse) => {
    setSession(uploaded);
    setPoints([]);
    setBoxes([]);
    setJob(null);
    setResult(null);
    setSelectedCandidateId(null);
    setTextPrompt("");
  }, []);

  const handleUpload = useCallback(async (file: File) => {
    try {
      setIsUploading(true);
      setErrorMessage(null);
      const uploaded = await uploadImage(file);
      applyUploadedSession(uploaded);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }, [applyUploadedSession]);

  const handleSampleSelect = useCallback(async (filename: string) => {
    try {
      setIsUploading(true);
      setErrorMessage(null);
      const uploaded = await loadSampleImage(filename);
      applyUploadedSession(uploaded);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Sample load failed.");
    } finally {
      setIsUploading(false);
    }
  }, [applyUploadedSession]);

  useEffect(() => {
    if (hasAttemptedInitialSampleLoad.current || session || samples.length === 0) {
      return;
    }

    hasAttemptedInitialSampleLoad.current = true;
    void handleSampleSelect(samples[0]);
  }, [handleSampleSelect, samples, session]);

  const handleSubmit = async (request: SegmentationRequest) => {
    try {
      setErrorMessage(null);
      setIsSubmitting(true);
      setJob(createPendingJobStatus(request.session_id));

      const completed = await createSegmentationJob(request);
      startTransition(() => {
        setJob(completed);
        if (completed.status === "complete" && completed.result) {
          setResult(completed.result);
          setSelectedCandidateId(null);
          return;
        }
        if (completed.status === "failed") {
          setErrorMessage(completed.error ?? completed.message);
          return;
        }
        setResult(null);
        setErrorMessage("Segmentation finished without a usable result.");
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Segmentation request failed.";
      setJob((current) =>
        current
          ? {
              ...current,
              status: "failed",
              progress: 1,
              message: "Segmentation failed",
              updated_at: new Date().toISOString(),
              error: message,
            }
          : null,
      );
      setErrorMessage(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const runCurrentPrompts = async () => {
    if (!session) {
      return;
    }

    await handleSubmit(
      promptsToRequest(session, interactionMode, points, boxes, textPrompt),
    );
  };

  const handleImmediateClick = async (point: PromptPoint) => {
    if (!session) {
      return;
    }
    setPoints([point]);
    setBoxes([]);
    await handleSubmit(promptsToRequest(session, "click", [point], [], textPrompt));
  };

  const handleResetImageState = async () => {
    if (!session) {
      return;
    }
    try {
      setErrorMessage(null);
      await resetSessionCache(session.session_id);
      setPoints([]);
      setBoxes([]);
      setResult(null);
      setJob(null);
      setSelectedCandidateId(null);
      setTextPrompt("");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Reset failed.");
    }
  };

  const handleInteractionModeChange = (nextMode: InteractionMode) => {
    setInteractionMode(nextMode);
    if (!isAccumulatedPointMode(nextMode)) {
      return;
    }

    setPoints((current) => current.filter((point) => point.source !== "click"));
  };

  const actionDisabled = !session || isSubmitting;
  const statusBusy = isSubmitting || isUploading;
  const submitButtonLabel = isSubmitting ? "Waiting for model..." : "Run Segmentation";

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    onHeaderStateChange({
      disabled: statusBusy,
      currentFilename: session?.filename ?? null,
      onUpload: handleUpload,
    });
  }, [handleUpload, onHeaderStateChange, session?.filename, statusBusy]);

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    return () => {
      onHeaderStateChange(null);
    };
  }, [onHeaderStateChange]);

  return (
    <main className="workspace-grid sandbox-page-shell">
      <aside className="control-panel">
        <TextPromptInput
          disabled={!session || statusBusy}
          value={textPrompt}
          onChange={setTextPrompt}
          onSubmit={runCurrentPrompts}
        />
        <SamplesPanel
          samples={samples}
          currentFilename={session?.filename ?? null}
          disabled={statusBusy}
          isLoading={isLoadingSamples}
          errorMessage={samplesErrorMessage}
          onSelectSample={handleSampleSelect}
        />
      </aside>

      <section className="canvas-column">
        <PromptToolbar
          disabled={statusBusy || !session}
          mode={interactionMode}
          onModeChange={handleInteractionModeChange}
        />
        <SegmentationCanvas
          imageUrl={session?.image_url ?? null}
          imageWidth={session?.width ?? 0}
          imageHeight={session?.height ?? 0}
          mode={interactionMode}
          points={points}
          boxes={boxes}
          masks={visibleMasks}
          selectedCandidateId={selectedCandidateId}
          overlayOpacity={overlayOpacity}
          disabled={!session || statusBusy}
          uploadDisabled={statusBusy}
          onSelectMask={setSelectedCandidateId}
          onUploadImage={handleUpload}
          onAddPoint={(point) => {
            setPoints((current) => [...current, point]);
            setErrorMessage(null);
          }}
          onRemovePoint={(pointId) => {
            setPoints((current) => current.filter((point) => point.id !== pointId));
            setErrorMessage(null);
          }}
          onSingleClickSegment={handleImmediateClick}
          onAddBox={(box) => {
            setBoxes((current) => [...current, box]);
            setErrorMessage(null);
          }}
          onRemoveBox={(boxId) => {
            setBoxes((current) => current.filter((box) => box.id !== boxId));
            setErrorMessage(null);
          }}
        />
      </section>

      <aside className="sandbox-right-column">
        <section className="panel-block actions-block">
          <div className="section-heading">
            <span className="section-kicker">Sandbox</span>
            <h2>Run & Refine</h2>
          </div>
          <div className="prompt-summary">
            <span>{points.length} points</span>
            <span>{boxes.length} boxes</span>
            <span>{textPrompt.trim() ? "text ready" : "no text"}</span>
          </div>
          <button
            type="button"
            className="primary-button"
            disabled={actionDisabled}
            onClick={runCurrentPrompts}
          >
            {submitButtonLabel}
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!session || statusBusy}
            onClick={() => {
              setPoints([]);
              setBoxes([]);
              setTextPrompt("");
            }}
          >
            Clear Current Prompts
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={!session || statusBusy}
            onClick={handleResetImageState}
          >
            Reset Image State
          </button>
          <div className="actions-status-group">
            <JobStatusIndicator job={job} />
            {errorMessage ? <div className="error-banner inline-error">{errorMessage}</div> : null}
          </div>
        </section>

        <ResultsPanel
          result={result}
          visibleCandidates={visibleCandidates}
          selectedCandidateId={selectedCandidateId}
          confidenceThreshold={confidenceThreshold}
          overlayOpacity={overlayOpacity}
          onSelectCandidate={setSelectedCandidateId}
          onConfidenceThresholdChange={setConfidenceThreshold}
          onOverlayOpacityChange={setOverlayOpacity}
          onShowAllMasks={() => setSelectedCandidateId(null)}
        />
      </aside>
    </main>
  );
}