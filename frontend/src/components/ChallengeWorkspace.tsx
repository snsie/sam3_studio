import { startTransition, useDeferredValue, useEffect, useMemo, useState } from "react";

import {
  createSegmentationJob,
  loadChallengeImage,
  resetSessionCache,
  scoreChallengeDataset,
} from "../api/client";
import type {
  BrushMode,
  ChallengeExample,
  ChallengeId,
  ChallengeImageItem,
  ChallengeImageMaskState,
  ChallengeScoreSummary,
  ChallengeSummary,
  InteractionMode,
  JobStatusResponse,
  MaskCandidate,
  PromptBox,
  PromptPoint,
  SegmentationRequest,
  SegmentationResult,
  UploadResponse,
} from "../types";
import {
  candidateMaskToMaskData,
  countStoredMaskImages,
  createMaskLayer,
  normalizeMaskLayerColors,
  recolorMaskCandidates,
  replaceMaskLayerData,
  serializeMaskToDataUrl,
  updateMaskWithBrush,
} from "../utils/challengeMasks";
import { AnnotationCanvas } from "./AnnotationCanvas";
import { BrushControls } from "./BrushControls";
import { ExamplesModal } from "./ExamplesModal";
import { ImageListSidebar } from "./ImageListSidebar";
import { MaskIdPanel } from "./MaskIdPanel";
import { PromptToolbar } from "./PromptToolbar";
import { ResultsPanel } from "./ResultsPanel";
import { ScorePanel } from "./ScorePanel";
import { TextPromptInput } from "./TextPromptInput";

interface ChallengeWorkspaceProps {
  challenge: ChallengeSummary;
  images: ChallengeImageItem[];
  examples: ChallengeExample[];
  imageStates: Record<string, ChallengeImageMaskState>;
  imagesLoading: boolean;
  imagesErrorMessage: string | null;
  selectedImageIndex: number;
  userName: string;
  scoreResult: ChallengeScoreSummary | null;
  onSelectedImageIndexChange: (index: number) => void;
  onUserNameChange: (value: string) => void;
  onScoreResultChange: (result: ChallengeScoreSummary | null) => void;
  onUpdateImageState: (
    imageId: string,
    updater: (current: ChallengeImageMaskState | undefined) => ChallengeImageMaskState,
  ) => void;
}

const CHALLENGE_MASK_CONFIDENCE_THRESHOLD = 0.4;
const CHALLENGE_OVERLAY_OPACITY = 0.78;

const CHALLENGE_PROMPT_MODES: InteractionMode[] = [
  "positive_point",
  "negative_point",
  "box",
];

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

function createDefaultImageState(
  challengeId: ChallengeId,
  imageId: string,
  width: number,
  height: number,
): ChallengeImageMaskState {
  const layer = createMaskLayer(
    width,
    height,
    challengeId,
    0,
    challengeId === "single-class" ? "Foreground" : undefined,
  );
  return {
    image_id: imageId,
    width,
    height,
    layers: [layer],
    active_mask_id: layer.mask_id,
    stored: false,
    stored_at: null,
  };
}

function mergeMaskData(
  masks: Uint8ClampedArray[],
  width: number,
  height: number,
) {
  const merged = new Uint8ClampedArray(width * height);
  masks.forEach((mask) => {
    for (let index = 0; index < merged.length; index += 1) {
      if (mask[index] > 0) {
        merged[index] = 255;
      }
    }
  });
  return merged;
}

function applyStoredMasks(
  current: ChallengeImageMaskState,
  decodedMasks: Uint8ClampedArray[],
): ChallengeImageMaskState {
  if (decodedMasks.length === 0) {
    return current;
  }

  const activeMaskId = current.active_mask_id ?? current.layers[0]?.mask_id ?? null;
  if (!activeMaskId) {
    return current;
  }

  const mergedMask = mergeMaskData(decodedMasks, current.width, current.height);

  return {
    ...current,
    layers: current.layers.map((layer) =>
      layer.mask_id === activeMaskId ? replaceMaskLayerData(layer, mergedMask) : layer,
    ),
  };
}

export function ChallengeWorkspace({
  challenge,
  images,
  examples,
  imageStates,
  imagesLoading,
  imagesErrorMessage,
  selectedImageIndex,
  userName,
  scoreResult,
  onSelectedImageIndexChange,
  onUserNameChange,
  onScoreResultChange,
  onUpdateImageState,
}: ChallengeWorkspaceProps) {
  const [session, setSession] = useState<UploadResponse | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("positive_point");
  const [brushMode, setBrushMode] = useState<BrushMode>("off");
  const [brushRadius, setBrushRadius] = useState(12);
  const [textPrompt, setTextPrompt] = useState("");
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [boxes, setBoxes] = useState<PromptBox[]>([]);
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);
  const [isLoadingImage, setIsLoadingImage] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isScoring, setIsScoring] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [scoreErrorMessage, setScoreErrorMessage] = useState<string | null>(null);
  const [completionMessage, setCompletionMessage] = useState<string | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [displayCandidates, setDisplayCandidates] = useState<MaskCandidate[]>([]);
  const [brushSeedMask, setBrushSeedMask] = useState<Uint8ClampedArray | null>(null);

  const currentImage = images[selectedImageIndex] ?? null;
  const deferredResult = useDeferredValue(result);

  const visibleCandidates = useMemo(() => {
    if (!deferredResult || deferredResult.candidates.length === 0) {
      return [];
    }
    return deferredResult.candidates.filter(
      (candidate) =>
        typeof candidate.score === "number" &&
        candidate.score > CHALLENGE_MASK_CONFIDENCE_THRESHOLD,
    );
  }, [deferredResult]);

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
    if (displayCandidates.length === 0) {
      return [];
    }
    if (selectedCandidateId === null) {
      return displayCandidates;
    }
    const selectedCandidate = displayCandidates.find(
      (candidate) => candidate.candidate_id === selectedCandidateId,
    );
    return selectedCandidate ? [selectedCandidate] : displayCandidates;
  }, [displayCandidates, selectedCandidateId]);

  const editableCandidates = useMemo(() => {
    if (visibleCandidates.length === 0) {
      return [];
    }
    if (selectedCandidateId === null) {
      return visibleCandidates;
    }
    const selectedCandidate = visibleCandidates.find(
      (candidate) => candidate.candidate_id === selectedCandidateId,
    );
    return selectedCandidate ? [selectedCandidate] : [];
  }, [selectedCandidateId, visibleCandidates]);

  const completedImageIds = useMemo(
    () =>
      new Set(
        Object.values(imageStates)
          .filter((state) => state.stored)
          .map((state) => state.image_id),
      ),
    [imageStates],
  );

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.key !== "Enter" ||
        event.shiftKey ||
        event.isComposing ||
        !session ||
        isLoadingImage ||
        isSubmitting
      ) {
        return;
      }

      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.closest("textarea") ||
          target.closest("input") ||
          target.closest("select") ||
          target.closest("button") ||
          target.isContentEditable)
      ) {
        return;
      }

      event.preventDefault();
      void runCurrentPrompts();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [session, isLoadingImage, isSubmitting, interactionMode, points, boxes, textPrompt]);

  useEffect(() => {
    if (!currentImage) {
      setSession(null);
      return;
    }

    let isCurrent = true;
    const loadImage = async () => {
      try {
        setIsLoadingImage(true);
        setErrorMessage(null);
        setCompletionMessage(null);
        const uploaded = await loadChallengeImage(challenge.challenge_id, currentImage.image_id);
        if (!isCurrent) {
          return;
        }
        setSession(uploaded);
        setPoints([]);
        setBoxes([]);
        setJob(null);
        setResult(null);
        setSelectedCandidateId(null);
        setTextPrompt("");
      } catch (error) {
        if (!isCurrent) {
          return;
        }
        setSession(null);
        setErrorMessage(error instanceof Error ? error.message : "Failed to load challenge image.");
      } finally {
        if (isCurrent) {
          setIsLoadingImage(false);
        }
      }
    };

    void loadImage();
    return () => {
      isCurrent = false;
    };
  }, [challenge.challenge_id, currentImage]);

  const currentImageState = useMemo(() => {
    if (!currentImage || !session) {
      return null;
    }
    return (
      imageStates[currentImage.image_id] ??
      createDefaultImageState(challenge.challenge_id, currentImage.image_id, session.width, session.height)
    );
  }, [challenge.challenge_id, currentImage, imageStates, session]);

  useEffect(() => {
    if (!currentImage || !session || imageStates[currentImage.image_id]) {
      return;
    }
    onUpdateImageState(currentImage.image_id, () =>
      createDefaultImageState(challenge.challenge_id, currentImage.image_id, session.width, session.height),
    );
  }, [challenge.challenge_id, currentImage, imageStates, onUpdateImageState, session]);

  const activeLayer = useMemo(() => {
    if (!currentImageState) {
      return null;
    }
    return (
      currentImageState.layers.find((layer) => layer.mask_id === currentImageState.active_mask_id) ??
      currentImageState.layers[0] ??
      null
    );
  }, [currentImageState]);

  const canvasLayers = useMemo(() => {
    if (!currentImageState) {
      return [];
    }
    if (challenge.challenge_id !== "multi-class" || brushMode === "off" || !activeLayer) {
      return currentImageState.layers;
    }
    return [activeLayer];
  }, [activeLayer, brushMode, challenge.challenge_id, currentImageState]);

  const canvasSamMasks = useMemo(() => {
    if (challenge.challenge_id === "multi-class" && brushMode !== "off") {
      return [];
    }
    return visibleMasks;
  }, [brushMode, challenge.challenge_id, visibleMasks]);

  useEffect(() => {
    if (visibleCandidates.length === 0) {
      setDisplayCandidates([]);
      return;
    }

    if (!activeLayer) {
      setDisplayCandidates(visibleCandidates);
      return;
    }

    let isCurrent = true;
    setDisplayCandidates(visibleCandidates);

    const recolorCandidates = async () => {
      try {
        const nextCandidates = await recolorMaskCandidates(visibleCandidates, activeLayer.color_rgb);
        if (isCurrent) {
          setDisplayCandidates(nextCandidates);
        }
      } catch {
        if (isCurrent) {
          setDisplayCandidates(visibleCandidates);
        }
      }
    };

    void recolorCandidates();
    return () => {
      isCurrent = false;
    };
  }, [activeLayer, visibleCandidates]);

  useEffect(() => {
    if (editableCandidates.length === 0) {
      setBrushSeedMask(null);
      return;
    }

    let isCurrent = true;

    const seedBrushMask = async () => {
      try {
        const decodedMasks = await Promise.all(
          editableCandidates.map((candidate) => candidateMaskToMaskData(candidate)),
        );
        if (!isCurrent) {
          return;
        }
        setBrushSeedMask(
          mergeMaskData(decodedMasks, editableCandidates[0].width, editableCandidates[0].height),
        );
      } catch {
        if (isCurrent) {
          setBrushSeedMask(null);
        }
      }
    };

    void seedBrushMask();
    return () => {
      isCurrent = false;
    };
  }, [editableCandidates]);

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

  const handleResetImageState = async () => {
    if (!session || !currentImage || !currentImageState) {
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
      onUpdateImageState(currentImage.image_id, () =>
        createDefaultImageState(challenge.challenge_id, currentImage.image_id, currentImageState.width, currentImageState.height),
      );
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Reset failed.");
    }
  };

  const updateCurrentImageState = (
    updater: (current: ChallengeImageMaskState) => ChallengeImageMaskState,
  ) => {
    if (!currentImage) {
      return;
    }
    onUpdateImageState(currentImage.image_id, (current) => {
      if (current) {
        return updater(current);
      }
      if (!session) {
        throw new Error("Challenge image state cannot be initialized before the image session loads.");
      }
      return updater(
        createDefaultImageState(challenge.challenge_id, currentImage.image_id, session.width, session.height),
      );
    });
  };

  const handleBrushStroke = (x: number, y: number, draw: boolean) => {
    if (!activeLayer || !currentImageState) {
      return;
    }
    const nextBrushSeedMask = brushSeedMask;
    const shouldCommitCurrentCandidates =
      nextBrushSeedMask !== null && editableCandidates.length > 0;

    updateCurrentImageState((current) => ({
      ...current,
      layers: current.layers.map((layer) =>
        layer.mask_id === activeLayer.mask_id
          ? replaceMaskLayerData(
              layer,
              updateMaskWithBrush(
                shouldCommitCurrentCandidates
                  ? mergeMaskData([layer.data, nextBrushSeedMask], current.width, current.height)
                  : layer.data,
                current.width,
                current.height,
                x,
                y,
                brushRadius,
                draw,
              ),
            )
          : layer,
      ),
    }));

    if (shouldCommitCurrentCandidates) {
      setResult(null);
      setSelectedCandidateId(null);
      setDisplayCandidates([]);
    }
  };

  const clearMaskEditingState = () => {
    setPoints([]);
    setBoxes([]);
    setTextPrompt("");
    setResult(null);
    setJob(null);
    setSelectedCandidateId(null);
    setDisplayCandidates([]);
    setErrorMessage(null);
  };

  const handleAddMask = () => {
    if (!currentImageState) {
      return;
    }
    clearMaskEditingState();
    updateCurrentImageState((current) => {
      const nextIndex = current.layers.length;
      const nextLayer = createMaskLayer(
        current.width,
        current.height,
        challenge.challenge_id,
        nextIndex,
      );
      const nextLayers = normalizeMaskLayerColors(challenge.challenge_id, [
        ...current.layers,
        nextLayer,
      ]);
      return {
        ...current,
        layers: nextLayers,
        active_mask_id: nextLayer.mask_id,
      };
    });
  };

  const handleRenameMask = (maskId: string, label: string) => {
    updateCurrentImageState((current) => ({
      ...current,
      layers: current.layers.map((layer) =>
        layer.mask_id === maskId
          ? {
              ...layer,
              label,
            }
          : layer,
      ),
    }));
  };

  const handleDeleteMask = (maskId: string) => {
    if (!currentImageState) {
      return;
    }
    updateCurrentImageState((current) => {
      const nextLayers = normalizeMaskLayerColors(
        challenge.challenge_id,
        current.layers.filter((layer) => layer.mask_id !== maskId),
      );
      if (nextLayers.length === 0) {
        const fallbackLayer = createMaskLayer(
          current.width,
          current.height,
          challenge.challenge_id,
          0,
          challenge.challenge_id === "single-class" ? "Foreground" : undefined,
        );
        return {
          ...current,
          layers: [fallbackLayer],
          active_mask_id: fallbackLayer.mask_id,
        };
      }
      return {
        ...current,
        layers: nextLayers,
        active_mask_id:
          current.active_mask_id === maskId ? nextLayers[0].mask_id : current.active_mask_id,
      };
    });
  };

  const handleStoreMask = async () => {
    if (!currentImage || !currentImageState) {
      return;
    }

    try {
      setErrorMessage(null);
      setCompletionMessage(null);

      const selectedCandidates =
        selectedCandidateId === null
          ? visibleCandidates
          : visibleCandidates.filter((candidate) => candidate.candidate_id === selectedCandidateId);
      const decodedMasks = await Promise.all(
        selectedCandidates.map((candidate) => candidateMaskToMaskData(candidate)),
      );

      const storedAt = new Date().toISOString();
      updateCurrentImageState((current) => {
        const nextState = applyStoredMasks(current, decodedMasks);
        return {
          ...nextState,
          stored: true,
          stored_at: storedAt,
        };
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Failed to store mask.");
      return;
    }

    const allImagesStored = images.every(
      (image) => image.image_id === currentImage.image_id || Boolean(imageStates[image.image_id]?.stored),
    );

    if (allImagesStored) {
      setCompletionMessage("All images have a stored mask. You can continue refining or score the dataset.");
    }
  };

  const handleScoreDataset = async () => {
    try {
      setScoreErrorMessage(null);
      setIsScoring(true);

      const submissions = images
        .map((image) => ({ image, state: imageStates[image.image_id] }))
        .filter((entry): entry is { image: ChallengeImageItem; state: ChallengeImageMaskState } => Boolean(entry.state?.stored))
        .map(({ image, state }) => ({
          image_id: image.image_id,
          width: state.width,
          height: state.height,
          masks: state.layers.map((layer) => ({
            mask_id: layer.mask_id,
            label: layer.label,
            image_data_url: serializeMaskToDataUrl(layer.data, state.width, state.height),
          })),
        }));

      const summary = await scoreChallengeDataset(challenge.challenge_id, {
        user_name: userName,
        submissions,
      });
      onScoreResultChange(summary);
    } catch (error) {
      setScoreErrorMessage(error instanceof Error ? error.message : "Scoring failed.");
    } finally {
      setIsScoring(false);
    }
  };

  const storedImageCount = countStoredMaskImages(imageStates);
  const scoreDisabled = !userName.trim() || storedImageCount === 0 || isScoring;
  const isBusy = isLoadingImage || isSubmitting;
  const handleInteractionModeChange = (nextMode: InteractionMode) => {
    setBrushMode("off");
    setInteractionMode(nextMode);
    if (!isAccumulatedPointMode(nextMode)) {
      return;
    }

    setPoints((current) => current.filter((point) => point.source !== "click"));
  };

  return (
    <section className="challenge-workspace-shell">
      <div className="challenge-workspace-grid">
        <div className="challenge-left-column">
          <TextPromptInput
            disabled={!session || isBusy}
            value={textPrompt}
            onChange={setTextPrompt}
            onSubmit={runCurrentPrompts}
          />
          <BrushControls
            brushMode={brushMode}
            brushRadius={brushRadius}
            disabled={!session || isBusy || !activeLayer}
            onBrushModeChange={setBrushMode}
            onBrushRadiusChange={setBrushRadius}
          />
          <ImageListSidebar
            images={images}
            activeImageId={currentImage?.image_id ?? null}
            completedImageIds={completedImageIds}
            isLoading={imagesLoading}
            errorMessage={imagesErrorMessage}
            onSelectImage={(imageId) => {
              const nextIndex = images.findIndex((image) => image.image_id === imageId);
              if (nextIndex >= 0) {
                onSelectedImageIndexChange(nextIndex);
              }
            }}
          />
        </div>

        <div className="challenge-canvas-column">
          <div className="panel-block challenge-toolbar-card">
            <PromptToolbar
              disabled={isBusy || !session}
              mode={interactionMode}
              availableModes={CHALLENGE_PROMPT_MODES}
              onModeChange={handleInteractionModeChange}
            />
          </div>
          <AnnotationCanvas
            imageUrl={session?.image_url ?? null}
            groundTruthOverlayUrl={session?.ground_truth_overlay_data_url ?? null}
            imageWidth={session?.width ?? currentImageState?.width ?? 0}
            imageHeight={session?.height ?? currentImageState?.height ?? 0}
            interactionMode={interactionMode}
            brushMode={brushMode}
            brushRadius={brushRadius}
            overlayOpacity={CHALLENGE_OVERLAY_OPACITY}
            disabled={!session || isBusy}
            points={points}
            boxes={boxes}
            samMasks={canvasSamMasks}
            layers={canvasLayers}
            onAddPoint={(point) => {
              setPoints((current) => [...current, point]);
              setErrorMessage(null);
            }}
            onRemovePoint={(pointId) => {
              setPoints((current) => current.filter((point) => point.id !== pointId));
              setErrorMessage(null);
            }}
            onAddBox={(box) => {
              setBoxes((current) => [...current, box]);
              setErrorMessage(null);
            }}
            onRemoveBox={(boxId) => {
              setBoxes((current) => current.filter((box) => box.id !== boxId));
              setErrorMessage(null);
            }}
            onBrushStroke={handleBrushStroke}
          />
          <div className="challenge-navigation-row panel-block">
            <button
              type="button"
              className="secondary-button compact-button"
              disabled={selectedImageIndex === 0}
              onClick={() => onSelectedImageIndexChange(Math.max(0, selectedImageIndex - 1))}
            >
              Previous
            </button>
            <span className="field-note">
              {currentImage ? `${currentImage.index} / ${images.length}` : "No image selected"}
            </span>
            <button
              type="button"
              className="secondary-button compact-button"
              disabled={selectedImageIndex >= images.length - 1}
              onClick={() => onSelectedImageIndexChange(Math.min(images.length - 1, selectedImageIndex + 1))}
            >
              Next
            </button>
          </div>
          {completionMessage ? <div className="panel-block completion-banner">{completionMessage}</div> : null}
        </div>

        <div className="challenge-right-column">
          <div className="panel-block challenge-actions-card">
            <div className="section-heading compact-heading">
              <span className="section-kicker">SAM3</span>
              <h2>Prompt Actions</h2>
            </div>
            <div className="challenge-header-actions">
              <button type="button" className="secondary-button compact-button" onClick={() => setShowExamples(true)}>
                Display Examples
              </button>
            </div>
            <div className="prompt-summary">
              <span>{points.length} points</span>
              <span>{boxes.length} boxes</span>
              <span>{textPrompt.trim() ? "text ready" : "no text"}</span>
            </div>
            <button
              type="button"
              className="primary-button"
              disabled={!session || isBusy}
              onClick={runCurrentPrompts}
            >
              {isSubmitting ? "Waiting for model..." : "Run Segmentation"}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={!session || isBusy}
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
              disabled={!session || isBusy}
              onClick={handleResetImageState}
            >
              Reset Current Image
            </button>
            <button
              type="button"
                className="secondary-button store-mask-button"
              disabled={!session || isBusy}
              onClick={() => void handleStoreMask()}
            >
              Store Mask
            </button>
            {job ? <div className="field-note">{job.message}</div> : null}
            {errorMessage ? <div className="error-banner inline-error">{errorMessage}</div> : null}
          </div>
          <ResultsPanel
            result={result}
            visibleCandidates={displayCandidates}
            listCandidates={displayCandidates}
            selectedCandidateId={selectedCandidateId}
            confidenceThreshold={CHALLENGE_MASK_CONFIDENCE_THRESHOLD}
            onSelectCandidate={setSelectedCandidateId}
            onShowAllMasks={() => setSelectedCandidateId(null)}
            allMasksButtonLabel="All Masks"
            showConfidenceThresholdControl={false}
            showOverlayOpacityControl={false}
            visibleCandidatesLabel="above 40% confidence"
            emptyThresholdMessage="No masks are above 40% confidence."
          />
          {challenge.challenge_id === "multi-class" && currentImageState ? (
            <MaskIdPanel
              layers={currentImageState.layers}
              activeMaskId={currentImageState.active_mask_id}
              disabled={!session || isBusy}
              onAddMask={handleAddMask}
              onSelectMask={(maskId) => {
                clearMaskEditingState();
                updateCurrentImageState((current) => ({
                  ...current,
                  active_mask_id: maskId,
                }));
              }}
              onRenameMask={handleRenameMask}
              onDeleteMask={handleDeleteMask}
            />
          ) : null}
          <ScorePanel
            userName={userName}
            storedImageCount={storedImageCount}
            datasetSize={images.length}
            disabled={scoreDisabled}
            isSubmitting={isScoring}
            errorMessage={scoreErrorMessage}
            result={scoreResult}
            onUserNameChange={onUserNameChange}
            onSubmit={() => void handleScoreDataset()}
          />
        </div>
      </div>

      <ExamplesModal
        challengeTitle={challenge.title}
        examples={examples}
        isOpen={showExamples}
        onClose={() => setShowExamples(false)}
      />
    </section>
  );
}