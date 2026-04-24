import type { MaskCandidate } from "./types";

export function getMaskDisplayNumber(
  candidate: Pick<MaskCandidate, "candidate_id">,
  fallbackIndex: number,
): number {
  const match = /^mask-(\d+)$/i.exec(candidate.candidate_id);
  if (!match) {
    return fallbackIndex + 1;
  }
  return Number(match[1]);
}
