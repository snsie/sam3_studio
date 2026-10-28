import { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

import {
  createSegmentationJob,
  loadChallengeImage,
  resetSessionCache,
  scoreChallengeDataset,
} from "../api/client";
import type {
  BrushMode,
  BrushStrokeSegment,
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
  resizeMaskData,
  serializeMaskToDataUrl,
  updateMaskWithBrushStroke,
} from "../utils/challengeMasks";
import { AnnotationCanvas } from "./AnnotationCanvas";
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
    // A mask laid out for a different pixel grid cannot be indexed with this width:
    // doing so shears the mask across rows instead of failing, so drop it instead.
    if (mask.length !== merged.length) {
      return;
    }
    for (let index = 0; index < merged.length; index += 1) {
      if (mask[index] > 0) {
        merged[index] = 255;
      }
    }
  });
  return merged;
}

async function mergeMaskCandidates(
  candidates: MaskCandidate[],
  width: number,
  height: number,
) {
  const decodedMasks = await Promise.all(
    candidates.map((candidate) => candidateMaskToMaskData(candidate, width, height)),
  );
  return mergeMaskData(decodedMasks, width, height);
}

function resizeImageMaskState(
  state: ChallengeImageMaskState,
  width: number,
  height: number,
): ChallengeImageMaskState {
  if (state.width === width && state.height === height) {
    return state;
  }
  return {
    ...state,
    width,
    height,
    layers: state.layers.map((layer) =>
      replaceMaskLayerData(
        layer,
        resizeMaskData(layer.data, state.width, state.height, width, height),
      ),
    ),
  };
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

  const activeLayer =
    current.layers.find((layer) => layer.mask_id === activeMaskId) ?? null;
  const mergedMask = mergeMaskData(
    activeLayer ? [activeLayer.data, ...decodedMasks] : decodedMasks,
    current.width,
    current.height,
  );

  return {
    ...current,
    layers: current.layers.map((layer) =>
      layer.mask_id === activeMaskId ? replaceMaskLayerData(layer, mergedMask) : layer,
    ),
  };
}

interface MaskDraftState {
  result: SegmentationResult | null;
  displayCandidates: MaskCandidate[];
  selectedCandidateId: string | null;
  mergedMaskData: Uint8ClampedArray | null;
}

const EMPTY_MASK_DRAFTS: Record<string, MaskDraftState> = {};

interface ImageSessionState {
  imageId: string;
  upload: UploadResponse;
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
  const [sessionState, setSessionState] = useState<ImageSessionState | null>(null);
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("positive_point");
  const [brushMode, setBrushMode] = useState<BrushMode>("off");
  const [brushRadius, setBrushRadius] = useState(6);
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
  const [storeToast, setStoreToast] = useState<{ id: number; message: string } | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [displayCandidates, setDisplayCandidates] = useState<MaskCandidate[]>([]);
  const [draftImageId, setDraftImageId] = useState<string | null>(null);
  const [maskDraftsById, setMaskDraftsById] = useState<Record<string, MaskDraftState>>({});
  const draftResultScopeRef = useRef<string | null>(null);
  const draftMaskScopeRef = useRef<string | null>(null);
  const loadImageAbortRef = useRef<AbortController | null>(null);
  const segmentationAbortRef = useRef<AbortController | null>(null);
  const segmentationRequestSeqRef = useRef(0);
  const currentImageIdRef = useRef<string | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);

  const currentImage = images[selectedImageIndex] ?? null;
  // Sessions are loaded asynchronously, so on the render that switches images the
  // previous image's session is still in state. Anything derived from a session -
  // above all the mask buffer dimensions - must never read the wrong image's session,
  // so scope it to the image it was actually loaded for.
  const session =
    sessionState && sessionState.imageId === currentImage?.image_id
      ? sessionState.upload
      : null;
  const currentImageDrafts = useMemo(
    () => (draftImageId === currentImage?.image_id ? maskDraftsById : EMPTY_MASK_DRAFTS),
    [draftImageId, currentImage?.image_id, maskDraftsById],
  );
  const deferredResult = useDeferredValue(result);

  useEffect(() => {
    if (!storeToast) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      setStoreToast(null);
    }, 1800);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [storeToast]);

  useEffect(() => {
    setDraftImageId(currentImage?.image_id ?? null);
    setMaskDraftsById({});
    setDisplayCandidates([]);
  }, [currentImage?.image_id]);

  useEffect(() => {
    currentImageIdRef.current = currentImage?.image_id ?? null;
  }, [currentImage?.image_id]);

  useEffect(() => {
    currentSessionIdRef.current = session?.session_id ?? null;
  }, [session?.session_id]);

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

  const completedImageIds = useMemo(
    () =>
      new Set(
        Object.values(imageStates)
          .filter((state) => state.stored)
          .map((state) => state.image_id),
      ),
    [imageStates],
  );

  const highlightedImageId = useMemo(() => {
    let latestStoredImageId: string | null = null;
    let latestStoredAt: string | null = null;

    Object.values(imageStates).forEach((state) => {
      if (!state.stored || !state.stored_at) {
        return;
      }
      if (latestStoredAt === null || state.stored_at > latestStoredAt) {
        latestStoredAt = state.stored_at;
        latestStoredImageId = state.image_id;
      }
    });

    return latestStoredImageId;
  }, [imageStates]);

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
      loadImageAbortRef.current?.abort();
      loadImageAbortRef.current = null;
      segmentationRequestSeqRef.current += 1;
      segmentationAbortRef.current?.abort();
      segmentationAbortRef.current = null;
      setSessionState(null);
      setJob(null);
      setResult(null);
      setSelectedCandidateId(null);
      setDisplayCandidates([]);
      setIsSubmitting(false);
      return;
    }

    let isCurrent = true;
    const controller = new AbortController();
    loadImageAbortRef.current?.abort();
    loadImageAbortRef.current = controller;
    segmentationRequestSeqRef.current += 1;
    segmentationAbortRef.current?.abort();
    segmentationAbortRef.current = null;

    setSessionState(null);
    setPoints([]);
    setBoxes([]);
    setJob(null);
    setResult(null);
    setSelectedCandidateId(null);
    setTextPrompt("");
    setDisplayCandidates([]);
    setIsSubmitting(false);

    const loadImage = async () => {
      try {
        setIsLoadingImage(true);
        setErrorMessage(null);
        setCompletionMessage(null);
        const uploaded = await loadChallengeImage(
          challenge.challenge_id,
          currentImage.image_id,
          controller.signal,
        );
        if (!isCurrent) {
          return;
        }
        setSessionState({ imageId: currentImage.image_id, upload: uploaded });
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
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
        setSessionState(null);
        setErrorMessage(error instanceof Error ? error.message : "Failed to load challenge image.");
      } finally {
        if (isCurrent) {
          setIsLoadingImage(false);
          if (loadImageAbortRef.current === controller) {
            loadImageAbortRef.current = null;
          }
        }
      }
    };

    void loadImage();
    return () => {
      isCurrent = false;
      controller.abort();
      if (loadImageAbortRef.current === controller) {
        loadImageAbortRef.current = null;
      }
    };
  }, [challenge.challenge_id, currentImage]);

  useEffect(() => () => {
    loadImageAbortRef.current?.abort();
    segmentationAbortRef.current?.abort();
  }, []);

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
    if (!currentImage || !session) {
      return;
    }
    const existingState = imageStates[currentImage.image_id];
    if (!existingState) {
      onUpdateImageState(currentImage.image_id, () =>
        createDefaultImageState(challenge.challenge_id, currentImage.image_id, session.width, session.height),
      );
      return;
    }
    // Self-heal any state that was sized against a different image so mask buffers
    // always match the pixel grid the canvas and the scorer use.
    if (existingState.width === session.width && existingState.height === session.height) {
      return;
    }
    onUpdateImageState(currentImage.image_id, (current) =>
      resizeImageMaskState(current ?? existingState, session.width, session.height),
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

  const activeDraftMask = useMemo(() => {
    if (!activeLayer) {
      return null;
    }
    return currentImageDrafts[activeLayer.mask_id]?.mergedMaskData ?? null;
  }, [activeLayer, currentImageDrafts]);

  const activeDraftScopeKey = useMemo(() => {
    if (!currentImage?.image_id || !currentImageState?.active_mask_id) {
      return null;
    }
    return `${currentImage.image_id}:${currentImageState.active_mask_id}`;
  }, [currentImage?.image_id, currentImageState?.active_mask_id]);

  const canvasLayers = useMemo(() => {
    if (!currentImageState) {
      return [];
    }

    const mergedLayers = currentImageState.layers.map((layer) => {
      const draftMask = currentImageDrafts[layer.mask_id]?.mergedMaskData;
      if (!draftMask) {
        return layer;
      }

      return replaceMaskLayerData(
        layer,
        mergeMaskData([layer.data, draftMask], currentImageState.width, currentImageState.height),
      );
    });

    return mergedLayers;
  }, [challenge.challenge_id, currentImageDrafts, currentImageState]);

  const canvasSamMasks = useMemo(() => {
    if (challenge.challenge_id === "multi-class") {
      return [];
    }
    if (brushMode !== "off") {
      return [];
    }
    if (activeDraftMask) {
      return [];
    }
    return visibleMasks;
  }, [activeDraftMask, brushMode, challenge.challenge_id, visibleMasks]);

  useEffect(() => {
    if (challenge.challenge_id !== "multi-class") {
      return;
    }

    const activeMaskId = currentImageState?.active_mask_id;
    if (!activeMaskId) {
      setResult(null);
      setSelectedCandidateId(null);
      setDisplayCandidates([]);
      return;
    }

    const draft = currentImageDrafts[activeMaskId];
    setResult(draft?.result ?? null);
    setSelectedCandidateId(draft?.selectedCandidateId ?? null);
    setDisplayCandidates(draft?.displayCandidates ?? []);
  }, [challenge.challenge_id, currentImage?.image_id, currentImageState?.active_mask_id]);

  useEffect(() => {
    if (
      draftImageId !== currentImage?.image_id ||
      !currentImageState?.active_mask_id ||
      !activeDraftScopeKey
    ) {
      return;
    }

    if (draftResultScopeRef.current !== activeDraftScopeKey) {
      draftResultScopeRef.current = activeDraftScopeKey;
      return;
    }

    const activeMaskId = currentImageState.active_mask_id;

    setMaskDraftsById((current) => {
      const previous = current[activeMaskId];
      const previousMergedMask = previous?.mergedMaskData ?? null;

      if (result === null && selectedCandidateId === null && previousMergedMask === null) {
        if (!(activeMaskId in current)) {
          return current;
        }
        const { [activeMaskId]: _removed, ...rest } = current;
        return rest;
      }

      if (previous?.result === result && previous?.selectedCandidateId === selectedCandidateId) {
        return current;
      }

      return {
        ...current,
        [activeMaskId]: {
          result,
          displayCandidates: previous?.displayCandidates ?? [],
          selectedCandidateId,
          mergedMaskData: previousMergedMask,
        },
      };
    });
  }, [
    challenge.challenge_id,
    currentImage?.image_id,
    currentImageState?.active_mask_id,
    activeDraftScopeKey,
    draftImageId,
    result,
    selectedCandidateId,
  ]);

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
    if (
      challenge.challenge_id !== "multi-class" ||
      draftImageId !== currentImage?.image_id ||
      !currentImageState?.active_mask_id
    ) {
      return;
    }

    const activeMaskId = currentImageState.active_mask_id;

    setMaskDraftsById((current) => {
      const previous = current[activeMaskId];
      const nextDisplayCandidates = displayCandidates;

      if (previous?.displayCandidates === nextDisplayCandidates) {
        return current;
      }

      return {
        ...current,
        [activeMaskId]: {
          result: previous?.result ?? result,
          displayCandidates: nextDisplayCandidates,
          selectedCandidateId: previous?.selectedCandidateId ?? selectedCandidateId,
          mergedMaskData: previous?.mergedMaskData ?? null,
        },
      };
    });
  }, [
    challenge.challenge_id,
    currentImage?.image_id,
    currentImageState?.active_mask_id,
    displayCandidates,
    draftImageId,
    result,
    selectedCandidateId,
  ]);

  useEffect(() => {
    if (
      draftImageId !== currentImage?.image_id ||
      !currentImageState?.active_mask_id ||
      !activeDraftScopeKey
    ) {
      return;
    }

    if (draftMaskScopeRef.current !== activeDraftScopeKey) {
      draftMaskScopeRef.current = activeDraftScopeKey;
      return;
    }

    const activeMaskId = currentImageState.active_mask_id;

    if (!result || visibleMasks.length === 0) {
      return;
    }

    let isCurrent = true;

    const cacheDraftMask = async () => {
      try {
        const decodedMasks = await Promise.all(
          visibleMasks.map((candidate) =>
            candidateMaskToMaskData(
              candidate,
              currentImageState.width,
              currentImageState.height,
            ),
          ),
        );
        if (!isCurrent) {
          return;
        }
        setMaskDraftsById((current) => {
          const previous = current[activeMaskId];
          const activeLayerData =
            currentImageState.layers.find((layer) => layer.mask_id === activeMaskId)?.data ?? null;
          const mergedMask = mergeMaskData(
            [
              ...(activeLayerData ? [activeLayerData] : []),
              ...(previous?.mergedMaskData && previous.result !== result
                ? [previous.mergedMaskData]
                : []),
              ...decodedMasks,
            ],
            currentImageState.width,
            currentImageState.height,
          );

          return {
            ...current,
            [activeMaskId]: {
              result,
              displayCandidates: previous?.displayCandidates ?? displayCandidates,
              selectedCandidateId,
              mergedMaskData: mergedMask,
            },
          };
        });
      } catch {
        return;
      }
    };

    void cacheDraftMask();
    return () => {
      isCurrent = false;
    };
  }, [
    currentImage?.image_id,
    currentImageState?.active_mask_id,
    currentImageState?.height,
    currentImageState?.width,
    activeDraftScopeKey,
    draftImageId,
    displayCandidates,
    result,
    selectedCandidateId,
    visibleMasks,
  ]);

  const handleSubmit = async (request: SegmentationRequest) => {
    const submitImageId = currentImageIdRef.current;
    const submitSessionId = request.session_id;
    const submitSequence = segmentationRequestSeqRef.current + 1;
    segmentationRequestSeqRef.current = submitSequence;

    const controller = new AbortController();
    segmentationAbortRef.current?.abort();
    segmentationAbortRef.current = controller;

    try {
      setErrorMessage(null);
      setIsSubmitting(true);
      setJob(createPendingJobStatus(request.session_id));

      const completed = await createSegmentationJob(request, controller.signal);
      const isLatestRequest = segmentationRequestSeqRef.current === submitSequence;
      const isCurrentScope =
        currentImageIdRef.current === submitImageId &&
        currentSessionIdRef.current === submitSessionId;

      if (!isLatestRequest || !isCurrentScope) {
        return;
      }

      startTransition(() => {
        if (
          segmentationRequestSeqRef.current !== submitSequence ||
          currentImageIdRef.current !== submitImageId ||
          currentSessionIdRef.current !== submitSessionId
        ) {
          return;
        }

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
      const isLatestRequest = segmentationRequestSeqRef.current === submitSequence;
      const isCurrentScope =
        currentImageIdRef.current === submitImageId &&
        currentSessionIdRef.current === submitSessionId;

      if (
        error instanceof DOMException &&
        error.name === "AbortError"
      ) {
        return;
      }

      if (!isLatestRequest || !isCurrentScope) {
        return;
      }

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
      if (segmentationRequestSeqRef.current === submitSequence) {
        setIsSubmitting(false);
      }
      if (segmentationAbortRef.current === controller) {
        segmentationAbortRef.current = null;
      }
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
      setDisplayCandidates([]);
      setMaskDraftsById({});
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

  const clearPromptState = () => {
    setPoints([]);
    setBoxes([]);
    setTextPrompt("");
    setJob(null);
    setDisplayCandidates([]);
    setErrorMessage(null);
  };

  const clearMaskDraft = (maskId: string) => {
    setMaskDraftsById((current) => {
      if (!(maskId in current)) {
        return current;
      }
      const { [maskId]: _removed, ...rest } = current;
      return rest;
    });

    if (currentImageState?.active_mask_id === maskId) {
      setResult(null);
      setSelectedCandidateId(null);
      setDisplayCandidates([]);
    }
  };

  const handleBrushStroke = (stroke: BrushStrokeSegment) => {
    if (!activeLayer || !currentImageState) {
      return;
    }
    const shouldCommitCurrentDraft = activeDraftMask !== null;

    updateCurrentImageState((current) => ({
      ...current,
      layers: current.layers.map((layer) =>
        layer.mask_id === activeLayer.mask_id
          ? replaceMaskLayerData(
            layer,
            updateMaskWithBrushStroke(
              shouldCommitCurrentDraft && activeDraftMask
                ? mergeMaskData([layer.data, activeDraftMask], current.width, current.height)
                : layer.data,
              current.width,
              current.height,
              stroke.fromX,
              stroke.fromY,
              stroke.toX,
              stroke.toY,
              brushRadius,
              stroke.draw,
            ),
          )
          : layer,
      ),
    }));

    if (shouldCommitCurrentDraft) {
      clearMaskDraft(activeLayer.mask_id);
    }
  };

  const handleAddMask = () => {
    if (!currentImageState) {
      return;
    }
    clearPromptState();
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
    clearMaskDraft(maskId);
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
      setStoreToast(null);

      const nextDraftMask =
        activeDraftMask ??
        (visibleMasks.length > 0
          ? await mergeMaskCandidates(
            visibleMasks,
            currentImageState.width,
            currentImageState.height,
          )
          : null);

      const storedAt = new Date().toISOString();
      updateCurrentImageState((current) => {
        const nextState = nextDraftMask ? applyStoredMasks(current, [nextDraftMask]) : current;
        return {
          ...nextState,
          stored: true,
          stored_at: storedAt,
        };
      });
      if (activeLayer) {
        clearMaskDraft(activeLayer.mask_id);
      }
      setStoreToast({
        id: Date.now(),
        message: "Mask stored",
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
          <ImageListSidebar
            images={images}
            activeImageId={currentImage?.image_id ?? null}
            completedImageIds={completedImageIds}
            highlightedImageId={highlightedImageId}
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
              brushMode={brushMode}
              brushRadius={brushRadius}
              brushDisabled={!session || isBusy || !activeLayer}
              onModeChange={handleInteractionModeChange}
              onBrushModeChange={setBrushMode}
              onBrushRadiusChange={setBrushRadius}
            />
          </div>
          <AnnotationCanvas
            imageUrl={session?.image_url ?? null}
            groundTruthOverlayUrl={session?.ground_truth_overlay_data_url ?? null}
            imageWidth={session?.width ?? currentImageState?.width ?? 0}
            imageHeight={session?.height ?? currentImageState?.height ?? 0}
            headerActions={(
              <div className="annotation-canvas-nav">
                <button
                  type="button"
                  className="secondary-button compact-button"
                  disabled={selectedImageIndex === 0}
                  onClick={() => onSelectedImageIndexChange(Math.max(0, selectedImageIndex - 1))}
                >
                  Previous
                </button>
                <span className="field-note annotation-canvas-nav-status">
                  {currentImage ? `${currentImage.index} / ${images.length}` : "No image selected"}
                </span>
                <button
                  type="button"
                  className="secondary-button compact-button"
                  disabled={selectedImageIndex >= images.length - 1}
                  onClick={() =>
                    onSelectedImageIndexChange(Math.min(images.length - 1, selectedImageIndex + 1))
                  }
                >
                  Next
                </button>
              </div>
            )}
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
          {completionMessage ? <div className="panel-block completion-banner">{completionMessage}</div> : null}
        </div>

        <div className="challenge-right-column">
          <div className="panel-block challenge-actions-card">
            <div className="section-heading compact-heading">
              {/* <span className="section-kicker">SAM3</span> */}
              <h2>Prompt Actions</h2>
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
            <div className="store-mask-toast-anchor">
              <button
                type="button"
                className="secondary-button store-mask-button"
                disabled={!session || isBusy}
                onClick={() => void handleStoreMask()}
              >
                {storeToast ? "Stored" : "Store Mask"}
              </button>
              {storeToast ? (
                <div key={storeToast.id} className="store-mask-toast" aria-live="polite">
                  {storeToast.message}
                </div>
              ) : null}
            </div>
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
                clearPromptState();
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