import { useCallback, useEffect, useMemo, useState } from "react";

import { listChallengeExamples, listChallengeImages, listChallenges } from "../api/client";
import { ChallengeSelector } from "../components/ChallengeSelector";
import { ChallengeWorkspace } from "../components/ChallengeWorkspace";
import type { ActiveChallengeHeaderState } from "../headerContext";
import type {
  ChallengeExamplesResponse,
  ChallengeId,
  ChallengeImageItem,
  ChallengeImageMaskState,
  ChallengeScoreSummary,
  ChallengeSummary,
} from "../types";

interface ChallengesPageProps {
  onHeaderStateChange?: (state: ActiveChallengeHeaderState | null) => void;
}

export function ChallengesPage({ onHeaderStateChange }: ChallengesPageProps) {
  const [challenges, setChallenges] = useState<ChallengeSummary[]>([]);
  const [selectedChallengeId, setSelectedChallengeId] = useState<ChallengeId | null>(null);
  const [isLoadingChallenges, setIsLoadingChallenges] = useState(true);
  const [challengeErrorMessage, setChallengeErrorMessage] = useState<string | null>(null);
  const [imagesByChallenge, setImagesByChallenge] = useState<Partial<Record<ChallengeId, ChallengeImageItem[]>>>({});
  const [examplesByChallenge, setExamplesByChallenge] = useState<Partial<Record<ChallengeId, ChallengeExamplesResponse["examples"]>>>({});
  const [imagesLoadingByChallenge, setImagesLoadingByChallenge] = useState<Partial<Record<ChallengeId, boolean>>>({});
  const [imagesErrorByChallenge, setImagesErrorByChallenge] = useState<Partial<Record<ChallengeId, string | null>>>({});
  const [imageStatesByChallenge, setImageStatesByChallenge] = useState<Record<ChallengeId, Record<string, ChallengeImageMaskState>>>({
    "single-class": {},
    "multi-class": {},
  });
  const [selectedImageIndexByChallenge, setSelectedImageIndexByChallenge] = useState<Record<ChallengeId, number>>({
    "single-class": 0,
    "multi-class": 0,
  });
  const [userNamesByChallenge, setUserNamesByChallenge] = useState<Record<ChallengeId, string>>({
    "single-class": "",
    "multi-class": "",
  });
  const [scoreResultsByChallenge, setScoreResultsByChallenge] = useState<Record<ChallengeId, ChallengeScoreSummary | null>>({
    "single-class": null,
    "multi-class": null,
  });

  useEffect(() => {
    let isCurrent = true;
    const loadChallenges = async () => {
      try {
        setIsLoadingChallenges(true);
        const nextChallenges = await listChallenges();
        if (!isCurrent) {
          return;
        }
        setChallenges(nextChallenges);
        setSelectedChallengeId((current) =>
          current && nextChallenges.some((challenge) => challenge.challenge_id === current)
            ? current
            : null,
        );
        setChallengeErrorMessage(null);
      } catch (error) {
        if (!isCurrent) {
          return;
        }
        setChallengeErrorMessage(error instanceof Error ? error.message : "Failed to load challenge list.");
      } finally {
        if (isCurrent) {
          setIsLoadingChallenges(false);
        }
      }
    };

    void loadChallenges();
    return () => {
      isCurrent = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedChallengeId || imagesByChallenge[selectedChallengeId]) {
      return;
    }

    let isCurrent = true;
    const loadImages = async () => {
      try {
        setImagesLoadingByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: true,
        }));
        const response = await listChallengeImages(selectedChallengeId);
        if (!isCurrent) {
          return;
        }
        setImagesByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: response.images,
        }));
        setImagesErrorByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: null,
        }));
      } catch (error) {
        if (!isCurrent) {
          return;
        }
        setImagesErrorByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: error instanceof Error ? error.message : "Failed to load challenge images.",
        }));
      } finally {
        if (isCurrent) {
          setImagesLoadingByChallenge((current) => ({
            ...current,
            [selectedChallengeId]: false,
          }));
        }
      }
    };

    void loadImages();
    return () => {
      isCurrent = false;
    };
  }, [imagesByChallenge, selectedChallengeId]);

  useEffect(() => {
    if (!selectedChallengeId || examplesByChallenge[selectedChallengeId]) {
      return;
    }

    let isCurrent = true;
    const loadExamples = async () => {
      try {
        const response = await listChallengeExamples(selectedChallengeId);
        if (!isCurrent) {
          return;
        }
        setExamplesByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: response.examples,
        }));
      } catch {
        if (!isCurrent) {
          return;
        }
        setExamplesByChallenge((current) => ({
          ...current,
          [selectedChallengeId]: [],
        }));
      }
    };

    void loadExamples();
    return () => {
      isCurrent = false;
    };
  }, [examplesByChallenge, selectedChallengeId]);

  const activeChallenge = useMemo(
    () =>
      selectedChallengeId
        ? challenges.find((challenge) => challenge.challenge_id === selectedChallengeId) ?? null
        : null,
    [challenges, selectedChallengeId],
  );

  const activeChallengeId = activeChallenge?.challenge_id ?? null;

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    onHeaderStateChange(
      activeChallenge
        ? {
            title: activeChallenge.title,
            description: activeChallenge.description,
            onBackToMenu: () => setSelectedChallengeId(null),
          }
        : null,
    );
  }, [activeChallenge, onHeaderStateChange]);

  useEffect(() => {
    if (!onHeaderStateChange) {
      return;
    }

    return () => {
      onHeaderStateChange(null);
    };
  }, [onHeaderStateChange]);

  const updateImageState = useCallback(
    (
      challengeId: ChallengeId,
      imageId: string,
      updater: (current: ChallengeImageMaskState | undefined) => ChallengeImageMaskState,
    ) => {
      setImageStatesByChallenge((current) => ({
        ...current,
        [challengeId]: {
          ...current[challengeId],
          [imageId]: updater(current[challengeId][imageId]),
        },
      }));
    },
    [],
  );

  return (
    <section className="page-content-stack">
      {!activeChallenge ? (
        <>
          <section className="panel-block page-intro-card challenge-menu-hero">
            <div className="section-heading">
              <span className="section-kicker">Workshop</span>
              <h2>Choose Your Challenge</h2>
            </div>
            <p className="placeholder-copy">
              Start in the selection menu, pick one dataset track, and then move into the full annotation workspace for that challenge only.
            </p>
          </section>

          {isLoadingChallenges ? <div className="panel-block">Loading challenge definitions...</div> : null}
          {challengeErrorMessage ? <div className="error-banner">{challengeErrorMessage}</div> : null}

          {challenges.length > 0 ? (
            <ChallengeSelector
              challenges={challenges}
              onSelectChallenge={setSelectedChallengeId}
            />
          ) : null}
        </>
      ) : activeChallengeId ? (
        <>
          {challengeErrorMessage ? <div className="error-banner">{challengeErrorMessage}</div> : null}

          <ChallengeWorkspace
            challenge={activeChallenge}
            images={imagesByChallenge[activeChallengeId] ?? []}
            examples={examplesByChallenge[activeChallengeId] ?? []}
            imageStates={imageStatesByChallenge[activeChallengeId]}
            imagesLoading={Boolean(imagesLoadingByChallenge[activeChallengeId])}
            imagesErrorMessage={imagesErrorByChallenge[activeChallengeId] ?? null}
            selectedImageIndex={selectedImageIndexByChallenge[activeChallengeId] ?? 0}
            userName={userNamesByChallenge[activeChallengeId]}
            scoreResult={scoreResultsByChallenge[activeChallengeId]}
            onSelectedImageIndexChange={(index) =>
              setSelectedImageIndexByChallenge((current) => ({
                ...current,
                [activeChallengeId]: index,
              }))
            }
            onUserNameChange={(value) =>
              setUserNamesByChallenge((current) => ({
                ...current,
                [activeChallengeId]: value,
              }))
            }
            onScoreResultChange={(result) =>
              setScoreResultsByChallenge((current) => ({
                ...current,
                [activeChallengeId]: result,
              }))
            }
            onUpdateImageState={(imageId, updater) => updateImageState(activeChallengeId, imageId, updater)}
          />
        </>
      ) : null}
    </section>
  );
}