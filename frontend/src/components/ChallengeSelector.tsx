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
    headline: "BBBC038 nuclei images from the 2018 Data Science Bowl",
    overview:
      "This track uses Broad Bioimage Benchmark Collection BBBC038 images: a deliberately diverse set of nuclei microscopy samples spanning different organisms, stains, imaging conditions, and tissue contexts.",
    facts: [
      "Source: Broad Bioimage Benchmark Collection BBBC038, released for the Kaggle 2018 Data Science Bowl nuclei segmentation task.",
      "Original annotations: one non-overlapping mask per nucleus in each training image.",
      "Workshop adaptation: submit one combined foreground mask that covers all nuclei in the image.",
    ],
    buttonLabel: "Enter Single-Class Challenge",
  },
  "multi-class": {
    headline: "CoNIC colon nuclei segmentation and classification patches",
    overview:
      "This track uses CoNIC data: 256x256 colon histology patches created for nuclei identification, segmentation, classification, and counting in crowded tissue regions.",
    facts: [
      "Source: Tissue Image Analytics CoNIC challenge for colon nuclei analysis.",
      "Original labels combine instance segmentation with six cell classes: neutrophil, epithelial, lymphocyte, plasma, eosinophil, and connective.",
      "Workshop adaptation: annotate the provided class regions as separate mask targets rather than full panoptic instances.",
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
                {/* <p className="challenge-card-footnote">{challenge.description}</p> */}
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