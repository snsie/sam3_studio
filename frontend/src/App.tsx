import { startTransition, useDeferredValue, useMemo, useState } from "react";

import {
  createSegmentationJob,
  getJobStatus,
  resetSessionCache,
  uploadImage,
} from "./api/client";
import { ImageUploader } from "./components/ImageUploader";
import { JobStatusIndicator } from "./components/JobStatusIndicator";
import { MaskOverlayControls } from "./components/MaskOverlayControls";
import { PromptToolbar } from "./components/PromptToolbar";
import { ResultsPanel } from "./components/ResultsPanel";
import { SegmentationCanvas } from "./components/SegmentationCanvas";
import { TextPromptInput } from "./components/TextPromptInput";
import type {
  InteractionMode,
  JobStatusResponse,
  PromptBox,
  PromptPoint,
  SegmentationRequest,
  SegmentationResult,
  UploadResponse,
} from "./types";

const POLL_INTERVAL_MS = 700;

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
    multimask_output: true,
  };
}

export default function App() {
  const [session, setSession] = useState<UploadResponse | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("click");
  const [textPrompt, setTextPrompt] = useState("");
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [boxes, setBoxes] = useState<PromptBox[]>([]);
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [maskVisible, setMaskVisible] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.65);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const deferredResult = useDeferredValue(result);

  const visibleMasks = useMemo(() => {
    if (!deferredResult || deferredResult.candidates.length === 0) {
      return [];
    }
    if (selectedCandidateId === null) {
      return deferredResult.candidates;
    }
    const selectedCandidate = deferredResult.candidates.find(
      (candidate) => candidate.candidate_id === selectedCandidateId,
    );
    return selectedCandidate ? [selectedCandidate] : deferredResult.candidates;
  }, [deferredResult, selectedCandidateId]);

  const beginPolling = (jobId: string) => {
    const timer = window.setInterval(async () => {
      try {
        const nextJob = await getJobStatus(jobId);
        startTransition(() => {
          setJob(nextJob);
          if (nextJob.status === "complete") {
            setResult(nextJob.result);
            setSelectedCandidateId(null);
            setIsSubmitting(false);
            window.clearInterval(timer);
          }
          if (nextJob.status === "failed") {
            setErrorMessage(nextJob.error ?? nextJob.message);
            setIsSubmitting(false);
            window.clearInterval(timer);
          }
        });
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : "Polling failed.");
        setIsSubmitting(false);
        window.clearInterval(timer);
      }
    }, POLL_INTERVAL_MS);
  };

  const handleUpload = async (file: File) => {
    try {
      setIsUploading(true);
      setErrorMessage(null);
      const uploaded = await uploadImage(file);
      setSession(uploaded);
      setPoints([]);
      setBoxes([]);
      setJob(null);
      setResult(null);
      setSelectedCandidateId(null);
      setTextPrompt("");
      setMaskVisible(true);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (request: SegmentationRequest) => {
    try {
      setErrorMessage(null);
      setIsSubmitting(true);
      setJob(null);
      const submitted = await createSegmentationJob(request);
      setJob({
        job_id: submitted.job_id,
        session_id: request.session_id,
        status: submitted.status,
        progress: 0,
        message: "Queued",
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        result: null,
        error: null,
      });
      beginPolling(submitted.job_id);
    } catch (error) {
      setIsSubmitting(false);
      const message = error instanceof Error ? error.message : "Job submission failed.";
      setErrorMessage(message);
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

  const actionDisabled = !session || isSubmitting;
  const statusBusy = isSubmitting || isUploading;

  return (
    <div className="app-shell">
      <div className="background-grid" aria-hidden="true" />
      <header className="hero-header">
        <h1>SAM3 Studio</h1>
      </header>

      <main className="workspace-grid">
        <aside className="control-panel">
          <ImageUploader
            disabled={statusBusy}
            currentFilename={session?.filename ?? null}
            onUpload={handleUpload}
          />
          <TextPromptInput
            disabled={!session || statusBusy}
            value={textPrompt}
            onChange={setTextPrompt}
          />
          <MaskOverlayControls
            disabled={statusBusy}
            hasMask={visibleMasks.length > 0}
            maskVisible={maskVisible}
            overlayOpacity={overlayOpacity}
            onMaskVisibleChange={setMaskVisible}
            onOverlayOpacityChange={setOverlayOpacity}
          />

          <section className="panel-block actions-block">
            <div className="section-heading">
              <span className="section-kicker">Actions</span>
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
              {isSubmitting ? "Running..." : "Run Segmentation"}
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
        </aside>

        <section className="canvas-column">
          <PromptToolbar
            disabled={statusBusy || !session}
            mode={interactionMode}
            onModeChange={setInteractionMode}
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
            maskVisible={maskVisible}
            overlayOpacity={overlayOpacity}
            disabled={!session || statusBusy}
            onSelectMask={setSelectedCandidateId}
            onAddPoint={(point) => {
              setPoints((current) => [...current, point]);
              setErrorMessage(null);
            }}
            onSingleClickSegment={handleImmediateClick}
            onAddBox={(box) => {
              setBoxes((current) => [...current, box]);
              setErrorMessage(null);
            }}
          />
        </section>

        <ResultsPanel
          result={result}
          selectedCandidateId={selectedCandidateId}
          onSelectCandidate={setSelectedCandidateId}
          onShowAllMasks={() => setSelectedCandidateId(null)}
        />
      </main>
    </div>
  );
}
