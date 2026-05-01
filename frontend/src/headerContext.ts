import type { ChallengeId, ChallengeScoreSummary } from "./types";

export interface ActiveChallengeHeaderState {
  title: string;
  description: string;
  onBackToMenu: () => void;
}

export interface SandboxHeaderState {
  disabled: boolean;
  currentFilename: string | null;
  onUpload: (file: File) => Promise<void>;
}

export interface LeaderboardHeaderState {
  activeChallengeId: ChallengeId;
  isLoading: boolean;
  errorMessage: string | null;
  topEntry: ChallengeScoreSummary | null;
  onChallengeChange: (challengeId: ChallengeId) => void;
}

export const HEADER_LEADERBOARD_OPTIONS: Array<{ id: ChallengeId; label: string }> = [
  { id: "single-class", label: "Single-Class Challenge" },
  { id: "multi-class", label: "Multi-Class Challenge" },
];