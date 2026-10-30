import { useCallback, useEffect, useState, type ReactNode } from "react";

import { AnnotationWidget } from "../components/pipeline/AnnotationWidget";
import { DatasetLayouts, SplitJsonNote } from "../components/pipeline/DatasetLayouts";
import { InstanceMatchingWidget } from "../components/pipeline/InstanceMatchingWidget";
import { OverlapMetricsWidget } from "../components/pipeline/OverlapMetricsWidget";
import { ToyTrainingWidget } from "../components/pipeline/ToyTrainingWidget";

interface StepDefinition {
  label: string;
  title: string;
  bullets: ReactNode[];
  aside?: ReactNode;
  render: (isVisible: boolean) => ReactNode;
}

const DATASET_FLOW = [
  { title: "Raw images", note: "microscope, scanner" },
  { title: "Annotate", note: "polygons / brush, or SAM3 pseudo-labels + fixes" },
  { title: "QA", note: "second annotator, spot checks" },
  { title: "Split", note: "train / val / test" },
];

function DatasetFlow() {
  return (
    <ol className="dataset-flow">
      {DATASET_FLOW.map((item) => (
        <li key={item.title}>
          <strong>{item.title}</strong>
          <span>{item.note}</span>
        </li>
      ))}
    </ol>
  );
}

interface PipelinePageProps {
  isActive: boolean;
  onOpenNucleiDemo?: () => void;
}

export function PipelinePage({ isActive, onOpenNucleiDemo }: PipelinePageProps) {
  const [stepIndex, setStepIndex] = useState(0);

  const steps: StepDefinition[] = [
    {
      label: "Dataset",
      title: "Creating a labeled dataset",
      bullets: [
        "A label is just data: an outline becomes a mask, an ID map or an RLE string.",
        "Draw by hand, or let a model (SAM3) propose and a human correct.",
        "Pitfalls: near-duplicate images across splits (leakage), class imbalance, annotators who disagree.",
      ],
      render: () => (
        <div className="pipeline-stack">
          <DatasetFlow />
          <AnnotationWidget />
        </div>
      ),
    },
    {
      label: "Files",
      title: "Files and folder structure",
      bullets: [
        "Every layout pairs an image with its labels: by filename, by folder, or by id in a JSON.",
        "PNG masks, polygons and RLE are the same information, encoded differently.",
        "Write the split down once, with a fixed seed.",
      ],
      aside: <SplitJsonNote />,
      render: () => <DatasetLayouts />,
    },
    {
      label: "Training",
      title: "Training: the loop",
      bullets: [
        "Forward → loss → gradients → update, for every batch, every epoch.",
        "Watch train vs. val: validation is the honest number.",
        "Augment (flips, rotations, brightness), save checkpoints, keep the best val epoch.",
        <>
          SAM3 + LoRA fine-tuning in this repo (<code>finetune_sam3_lora.py</code>) is the same loop, at scale, on a GPU.
        </>,
      ],
      render: (isVisible) => <ToyTrainingWidget isVisible={isVisible} />,
    },
    {
      label: "Pixel metrics",
      title: "Validation: how scoring works",
      bullets: [
        "Compare prediction and ground truth pixel by pixel: TP, FP, FN, TN.",
        "IoU and Dice ignore TN, so background can't inflate them. Pixel accuracy can: try a bigger image.",
        "HD95 measures boundary error in pixels.",
      ],
      render: () => <OverlapMetricsWidget />,
    },
    {
      label: "Instance metrics",
      title: "From pixels to objects",
      bullets: [
        "Match objects first: a prediction is a TP if its IoU with a GT object is above the threshold.",
        "Unmatched predictions → FP. Unmatched GT → FN.",
        "DSB 2018 averages TP / (TP + FP + FN) over IoU 0.50 … 0.95, which rewards tight outlines.",
      ],
      render: () => <InstanceMatchingWidget onOpenNucleiDemo={onOpenNucleiDemo} />,
    },
  ];

  const goTo = useCallback((index: number) => {
    setStepIndex(Math.min(4, Math.max(0, index)));
  }, []);

  useEffect(() => {
    if (!isActive) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) {
        return;
      }
      // Text fields keep their arrow keys. Sliders don't: after a drag the slider keeps focus,
      // and the presenter still expects ←/→ to change step.
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.closest('textarea, select, input:not([type="range"]):not([type="checkbox"])') || target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        setStepIndex((current) => Math.min(4, current + 1));
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        setStepIndex((current) => Math.max(0, current - 1));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isActive]);

  return (
    <section className="page-content-stack pipeline-page">
      <nav className="pipeline-stepper" aria-label="Pipeline steps">
        <button
          type="button"
          className="pipeline-stepper-arrow"
          onClick={() => goTo(stepIndex - 1)}
          disabled={stepIndex === 0}
          aria-label="Previous step"
        >
          ←
        </button>
        <ol>
          {steps.map((step, index) => (
            <li key={step.label}>
              <button
                type="button"
                className={`pipeline-stepper-step${index === stepIndex ? " is-active" : ""}${index < stepIndex ? " is-done" : ""}`}
                aria-current={index === stepIndex ? "step" : undefined}
                onClick={() => goTo(index)}
              >
                <span className="pipeline-stepper-number">{index + 1}</span>
                <span className="pipeline-stepper-label">{step.label}</span>
              </button>
            </li>
          ))}
        </ol>
        <button
          type="button"
          className="pipeline-stepper-arrow"
          onClick={() => goTo(stepIndex + 1)}
          disabled={stepIndex === steps.length - 1}
          aria-label="Next step"
        >
          →
        </button>
      </nav>

      {/* Steps stay mounted so widget state (e.g. a training run) survives navigation. */}
      {steps.map((step, index) => {
        const isVisible = isActive && index === stepIndex;
        return (
          <article key={step.label} className="pipeline-step" hidden={index !== stepIndex}>
            <div className="pipeline-step-copy">
              <span className="section-kicker">{`Step ${index + 1} of ${steps.length}`}</span>
              <h2>{step.title}</h2>
              <ul>
                {step.bullets.map((bullet, bulletIndex) => (
                  <li key={bulletIndex}>{bullet}</li>
                ))}
              </ul>
              {step.aside}
            </div>
            <div className="pipeline-step-stage panel-block">{step.render(isVisible)}</div>
          </article>
        );
      })}
    </section>
  );
}
