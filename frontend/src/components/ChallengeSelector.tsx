import type { ChallengeId, ChallengeSummary } from "../types";

interface ChallengeSelectorProps {
  challenges: ChallengeSummary[];
  onSelectChallenge: (challengeId: ChallengeId) => void;
}

const CHALLENGE_ORDER: ChallengeId[] = ["single-class", "multi-class"];

const DATASET_BRIEFS: Record<
  ChallengeId,
  {
    headline: string;
    overview: string;
    facts: string[];
    buttonLabel: string;
  }
> = {
  "single-class": {
    headline: "Fast foreground segmentation warmup",
    overview:
      "This lane focuses on fast, confident coverage. Treat each microscopy tile as one foreground-vs-background decision and aim for smooth, complete object boundaries.",
    facts: [
      "Imagery: bright pathology crops with a single dominant target family.",
      "Annotation goal: one clean binary mask per image, optimized for speed.",
      "Workshop note: most examples were curated to reward broad coverage over tiny edge corrections.",
    ],
    buttonLabel: "Enter Single-Class Challenge",
  },
  "multi-class": {
    headline: "Dense instance labeling sprint",
    overview:
      "This lane is designed for crowded scenes. Split nearby structures into separate IDs and use the manual brush to keep overlapping objects visually distinct.",
    facts: [
      "Imagery: high-density cell fields with frequent touching structures.",
      "Annotation goal: store multiple mask IDs per frame and keep identities stable.",
      "Workshop note: the scoring rubric favors consistent instance separation more than raw mask count.",
    ],
    buttonLabel: "Enter Multi-Class Challenge",
  },
};

export function ChallengeSelector({
  challenges,
  onSelectChallenge,
}: ChallengeSelectorProps) {
  const orderedChallenges = CHALLENGE_ORDER.map((challengeId) =>
    challenges.find((challenge) => challenge.challenge_id === challengeId),
  ).filter((challenge): challenge is ChallengeSummary => Boolean(challenge));

  return (
    <section className="challenge-selector">
      <div className="section-heading challenge-selector-heading">
        <span className="section-kicker">Challenges</span>
        <h2>Select a Dataset Track</h2>
      </div>
      <div className="challenge-selector-grid">
        {orderedChallenges.map((challenge) => {
          const datasetBrief = DATASET_BRIEFS[challenge.challenge_id];
          return (
            <article
              key={challenge.challenge_id}
              className="challenge-card panel-block"
            >
              <div className="challenge-card-copy">
                <span className="challenge-card-kicker">{challenge.image_count} images</span>
                <strong>{challenge.title}</strong>
                <p className="challenge-card-headline">{datasetBrief.headline}</p>
                <p className="challenge-card-overview">{datasetBrief.overview}</p>
              </div>

              <div className="challenge-card-facts" aria-label={`${challenge.title} dataset notes`}>
                {datasetBrief.facts.map((fact) => (
                  <p key={fact} className="challenge-card-fact">
                    {fact}
                  </p>
                ))}
              </div>

              <div className="challenge-card-footer">
                <p className="challenge-card-footnote">{challenge.description}</p>
                <button
                  type="button"
                  className="primary-button challenge-card-button"
                  onClick={() => onSelectChallenge(challenge.challenge_id)}
                >
                  {datasetBrief.buttonLabel}
                </button>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}