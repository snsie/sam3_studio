import { useMemo, useState, type ReactNode } from "react";

import { binaryRgba, labelRgba, microscopyRgba } from "../../utils/maskRender";
import { cocoRleCounts, rleEncode, type LabelMap } from "../../utils/segMetrics";
import { blobLabelMap, blobPolygon, createScene, type MicroscopyScene } from "../../utils/syntheticCells";
import { PixelCanvas } from "./PixelCanvas";

const SIZE = 96;
type LayoutId = "paired" | "dsb" | "coco" | "yolo";
type ThumbKind = "image" | "binary" | "semantic" | "instance" | "single";

interface Thumb {
  kind: ThumbKind;
  sample: number;
  /** For "single": which instance (1-based). */
  instance?: number;
}

interface TreeRow {
  name: string;
  depth: number;
  folder?: boolean;
  thumb?: Thumb;
  note?: string;
  muted?: boolean;
}

const LAYOUTS: Array<{ id: LayoutId; label: string; caption: string }> = [
  { id: "paired", label: "Paired folders", caption: "Semantic segmentation · U-Net style" },
  { id: "dsb", label: "Per-image folders", caption: "DSB 2018 / BBBC038 · what this repo uses" },
  { id: "coco", label: "COCO JSON", caption: "Instance segmentation · Detectron2, SAM fine-tuning" },
  { id: "yolo", label: "YOLO-seg", caption: "One .txt of polygons per image · Ultralytics" },
];

// Real ids from stage1_train.zip (the first image has 27 nucleus masks).
const DSB_IMAGE_ID = "00071198d059…";
const DSB_IMAGE_ID_2 = "003cee89357d…";
const DSB_MASK_IDS = ["07a9bf1d7594…", "0e548d0af63a…", "0ea1f9e30124…"];

const TREES: Record<LayoutId, TreeRow[]> = {
  paired: [
    { name: "dataset/", depth: 0, folder: true },
    { name: "images/", depth: 1, folder: true },
    { name: "train/", depth: 2, folder: true },
    { name: "img_0001.png", depth: 3, thumb: { kind: "image", sample: 0 } },
    { name: "img_0002.png", depth: 3, thumb: { kind: "image", sample: 1 } },
    { name: "val/", depth: 2, folder: true },
    { name: "img_0101.png", depth: 3, thumb: { kind: "image", sample: 2 } },
    { name: "masks/", depth: 1, folder: true },
    { name: "train/", depth: 2, folder: true },
    { name: "img_0001.png", depth: 3, thumb: { kind: "semantic", sample: 0 }, note: "same name" },
    { name: "img_0002.png", depth: 3, thumb: { kind: "semantic", sample: 1 } },
    { name: "val/", depth: 2, folder: true },
    { name: "img_0101.png", depth: 3, thumb: { kind: "semantic", sample: 2 } },
  ],
  dsb: [
    { name: "bbbc038/raw/stage1_train/", depth: 0, folder: true },
    { name: `${DSB_IMAGE_ID}/`, depth: 1, folder: true },
    { name: "images/", depth: 2, folder: true },
    { name: `${DSB_IMAGE_ID}.png`, depth: 3, thumb: { kind: "image", sample: 0 } },
    { name: "masks/", depth: 2, folder: true },
    { name: `${DSB_MASK_IDS[0]}.png`, depth: 3, thumb: { kind: "single", sample: 0, instance: 1 } },
    { name: `${DSB_MASK_IDS[1]}.png`, depth: 3, thumb: { kind: "single", sample: 0, instance: 2 } },
    { name: `${DSB_MASK_IDS[2]}.png`, depth: 3, thumb: { kind: "single", sample: 0, instance: 3 } },
    { name: "… 27 PNGs, one per nucleus", depth: 3, muted: true },
    { name: `${DSB_IMAGE_ID_2}/`, depth: 1, folder: true, note: "images/ + masks/" },
    { name: "bbbc038/raw/stage1_test/", depth: 0, folder: true, note: "same layout" },
    { name: "bbbc038/split.json", depth: 0, note: "train / val / test ids" },
    { name: "bbbc038/sam3_annotations/stage1_train_blobs.coco.json", depth: 0, note: "pseudo-labels" },
  ],
  coco: [
    { name: "coco/", depth: 0, folder: true },
    { name: "images/", depth: 1, folder: true },
    { name: "train/", depth: 2, folder: true },
    { name: "img_0001.png", depth: 3, thumb: { kind: "image", sample: 0 } },
    { name: "img_0002.png", depth: 3, thumb: { kind: "image", sample: 1 } },
    { name: "val/", depth: 2, folder: true },
    { name: "img_0101.png", depth: 3, thumb: { kind: "image", sample: 2 } },
    { name: "annotations/", depth: 1, folder: true },
    { name: "instances_train.json", depth: 2, thumb: { kind: "instance", sample: 0 }, note: "every object, every image" },
    { name: "instances_val.json", depth: 2, thumb: { kind: "instance", sample: 2 } },
  ],
  yolo: [
    { name: "yolo/", depth: 0, folder: true },
    { name: "data.yaml", depth: 1, note: "paths + class names" },
    { name: "images/", depth: 1, folder: true },
    { name: "train/", depth: 2, folder: true },
    { name: "img_0001.png", depth: 3, thumb: { kind: "image", sample: 0 } },
    { name: "img_0002.png", depth: 3, thumb: { kind: "image", sample: 1 } },
    { name: "val/ …", depth: 2, folder: true },
    { name: "labels/", depth: 1, folder: true },
    { name: "train/", depth: 2, folder: true },
    { name: "img_0001.txt", depth: 3, thumb: { kind: "instance", sample: 0 }, note: "same stem" },
    { name: "img_0002.txt", depth: 3, thumb: { kind: "instance", sample: 1 } },
    { name: "val/ …", depth: 2, folder: true },
  ],
};

/** Minimal JSON/YAML-ish highlighter: keys, strings, numbers. */
function highlight(source: string): ReactNode[] {
  const pattern = /("(?:[^"\\]|\\.)*")(\s*:)?|(-?\b\d+(?:\.\d+)?\b)|(^\s*[a-z_]+:)/gm;
  const nodes: ReactNode[] = [];
  let last = 0;
  for (const match of source.matchAll(pattern)) {
    const index = match.index ?? 0;
    nodes.push(source.slice(last, index));
    if (match[1]) {
      nodes.push(
        <span key={index} className={match[2] ? "code-key" : "code-string"}>
          {match[1]}
        </span>,
      );
      if (match[2]) nodes.push(match[2]);
    } else if (match[3]) {
      nodes.push(
        <span key={index} className="code-number">
          {match[3]}
        </span>,
      );
    } else {
      nodes.push(
        <span key={index} className="code-key">
          {match[4]}
        </span>,
      );
    }
    last = index + match[0].length;
  }
  nodes.push(source.slice(last));
  return nodes;
}

function round(value: number, digits = 1) {
  return Number(value.toFixed(digits));
}

interface SceneData {
  scene: MicroscopyScene;
  labels: LabelMap;
  /** 0 = background, 1 = nucleus, 2 = dividing nucleus (one blob). */
  semantic: Uint8Array;
}

function useSamples(): SceneData[] {
  return useMemo(
    () =>
      [31, 32, 33].map((seed) => {
        const scene = createScene(seed, SIZE, SIZE, 4, [10, 16]);
        const labels = blobLabelMap(scene.blobs, SIZE, SIZE);
        const semantic = Uint8Array.from(labels, (value) => (value === 0 ? 0 : value === 2 ? 2 : 1));
        return { scene, labels, semantic };
      }),
    [],
  );
}

const SEMANTIC_COLORS: Array<[number, number, number]> = [
  [5, 12, 18],
  [45, 226, 207],
  [245, 158, 11],
];

function semanticRgba(semantic: Uint8Array) {
  const out = new Uint8ClampedArray(semantic.length * 4);
  semantic.forEach((value, index) => {
    out.set(SEMANTIC_COLORS[value], index * 4);
    out[index * 4 + 3] = 255;
  });
  return out;
}

function thumbPixels(thumb: Thumb, samples: SceneData[]) {
  const sample = samples[thumb.sample];
  switch (thumb.kind) {
    case "image":
      return microscopyRgba(sample.scene.image);
    case "binary":
      return binaryRgba(Uint8Array.from(sample.labels, (value) => (value > 0 ? 1 : 0)));
    case "semantic":
      return semanticRgba(sample.semantic);
    case "instance":
      return labelRgba(sample.labels);
    case "single":
      return binaryRgba(Uint8Array.from(sample.labels, (value) => (value === thumb.instance ? 1 : 0)));
  }
}

function FileTree({ rows, samples }: { rows: TreeRow[]; samples: SceneData[] }) {
  const pixels = useMemo(() => rows.map((row) => (row.thumb ? thumbPixels(row.thumb, samples) : null)), [rows, samples]);
  return (
    <ul className="file-tree">
      {rows.map((row, index) => (
        <li
          key={`${row.name}-${index}`}
          className={`${row.folder ? "is-folder" : "is-file"}${row.muted ? " is-muted" : ""}`}
          style={{ paddingLeft: `${row.depth * 1.15 + 0.4}em` }}
        >
          <span className="file-tree-icon" aria-hidden="true">
            {row.thumb ? (
              <PixelCanvas width={SIZE} height={SIZE} pixels={pixels[index]} className="file-tree-thumb" smooth />
            ) : row.folder ? (
              <svg viewBox="0 0 16 16">
                <path d="M1.5 3.5h4.2l1.4 1.6h7.4v7.9h-13z" />
              </svg>
            ) : row.muted ? null : (
              <svg viewBox="0 0 16 16">
                <path d="M3.5 1.5h6l3 3v10h-9z" />
              </svg>
            )}
          </span>
          <span className="file-tree-name">{row.name}</span>
          {row.note ? <span className="file-tree-note">{row.note}</span> : null}
        </li>
      ))}
    </ul>
  );
}

function SemanticPixelGrid({ sample }: { sample: SceneData }) {
  // An 8×8 window across the edge of the dividing nucleus (class 2).
  const blob = sample.scene.blobs[1];
  const x0 = Math.round(blob.cx - blob.radius - 3);
  const y0 = Math.round(blob.cy - 4);
  return (
    <div className="pixel-value-grid" aria-label="Mask pixel values">
      {Array.from({ length: 8 }, (_, row) =>
        Array.from({ length: 8 }, (_, column) => {
          const value = sample.semantic[(y0 + row) * SIZE + x0 + column];
          return (
            <span key={`${row}-${column}`} className={`class-${value}`}>
              {value}
            </span>
          );
        }),
      )}
    </div>
  );
}

function LayoutDetail({ layout, samples }: { layout: LayoutId; samples: SceneData[] }) {
  const first = samples[0];
  const polygons = first.scene.blobs.map((blob) => blobPolygon(blob, 10));

  if (layout === "paired") {
    return (
      <div className="layout-detail">
        <strong>masks/img_0001.png: pixel value = class index</strong>
        <SemanticPixelGrid sample={first} />
        <span className="field-note">0 background · 1 nucleus · 2 dividing nucleus. Same H×W as the image.</span>
      </div>
    );
  }

  if (layout === "dsb") {
    return (
      <div className="layout-detail">
        <strong>stage1_train_labels.csv: the same masks as RLE text</strong>
        <pre className="code-block">
          {highlight(
            `ImageId,EncodedPixels\n00071198d059…,"6908 1 7161 8 7417 8 7672 9 …"\n00071198d059…,"36269 7 36523 11 36778 13 …"`,
          )}
        </pre>
        <span className="field-note">
          start/length pairs, 1-indexed, column-major. prepare_bbbc038.py decodes stage1_solution.csv into the same
          per-nucleus PNGs for the test set.
        </span>
      </div>
    );
  }

  if (layout === "coco") {
    const instance1 = Uint8Array.from(first.labels, (value) => (value === 2 ? 1 : 0));
    const counts = cocoRleCounts(instance1, SIZE, SIZE);
    const polygon = polygons[0].slice(0, 4).flatMap(([x, y]) => [round(x), round(y)]);
    const xs = polygons[0].map(([x]) => x);
    const ys = polygons[0].map(([, y]) => y);
    const bbox = [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)].map(
      (value) => round(value),
    );
    const json = `{
  "images": [{"id": 1, "file_name": "img_0001.png",
              "width": ${SIZE}, "height": ${SIZE}}],
  "categories": [{"id": 1, "name": "nucleus"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "segmentation": [[${polygon.slice(0, 4).join(", ")}, …]],
     "bbox": [${bbox.join(", ")}], "area": ${first.labels.filter((value) => value === 1).length}},
    {"id": 2, "image_id": 1, "category_id": 1,
     "segmentation": {"size": [${SIZE}, ${SIZE}],
                      "counts": [${counts.slice(0, 3).join(", ")}, …]}}
  ]
}`;
    return (
      <div className="layout-detail">
        <strong>instances_train.json: polygon or RLE per object</strong>
        <pre className="code-block">{highlight(json)}</pre>
      </div>
    );
  }

  const lines = polygons.slice(0, 3).map(
    (polygon) =>
      `0 ${polygon
        .slice(0, 5)
        .map(([x, y]) => `${(x / SIZE).toFixed(3)} ${(y / SIZE).toFixed(3)}`)
        .join(" ")} …`,
  );
  return (
    <div className="layout-detail">
      <strong>labels/img_0001.txt: one object per line</strong>
      <pre className="code-block">
        {highlight(lines.join("\n"))}
      </pre>
      <span className="field-note">class x1 y1 x2 y2 … with coordinates normalized to 0–1</span>
      <pre className="code-block">{highlight("path: yolo\ntrain: images/train\nval: images/val\nnames:\n  0: nucleus")}</pre>
    </div>
  );
}

function MaskEncodings({ samples }: { samples: SceneData[] }) {
  const first = samples[0];
  const tiles = useMemo(
    () => ({
      binary: binaryRgba(Uint8Array.from(first.labels, (value) => (value > 0 ? 1 : 0))),
      semantic: semanticRgba(first.semantic),
      instance: labelRgba(first.labels),
    }),
    [first],
  );
  const rle = useMemo(
    () => rleEncode(Uint8Array.from(first.labels, (value) => (value === 1 ? 1 : 0)), SIZE, SIZE),
    [first],
  );

  return (
    <div className="mask-encodings">
      <span className="section-kicker">One image, four encodings</span>
      <div className="mask-encodings-row">
        <figure className="pipeline-tile">
          <PixelCanvas width={SIZE} height={SIZE} pixels={tiles.binary} label="Binary mask" />
          <figcaption>
            <strong>Binary</strong>
            <span>{"{0, 1}"}</span>
          </figcaption>
        </figure>
        <figure className="pipeline-tile">
          <PixelCanvas width={SIZE} height={SIZE} pixels={tiles.semantic} label="Semantic class-index mask" />
          <figcaption>
            <strong>Semantic</strong>
            <span>{"{0 … C} class"}</span>
          </figcaption>
        </figure>
        <figure className="pipeline-tile">
          <PixelCanvas width={SIZE} height={SIZE} pixels={tiles.instance} label="Instance label map" />
          <figcaption>
            <strong>Instance</strong>
            <span>{"{0 … N} object id"}</span>
          </figcaption>
        </figure>
        <figure className="pipeline-tile is-text">
          <code>{`${rle.slice(0, 10).join(" ")} …`}</code>
          <figcaption>
            <strong>RLE</strong>
            <span>{`object 1 · ${rle.length / 2} runs`}</span>
          </figcaption>
        </figure>
      </div>
    </div>
  );
}

export function DatasetLayouts() {
  const samples = useSamples();
  const [layout, setLayout] = useState<LayoutId>("dsb");
  const active = LAYOUTS.find((item) => item.id === layout) ?? LAYOUTS[0];

  return (
    <div className="dataset-layouts">
      <div className="pipeline-tabs" role="tablist" aria-label="Dataset layouts">
        {LAYOUTS.map((item) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={item.id === layout}
            className={`inline-toggle-button${item.id === layout ? " is-active" : ""}`}
            onClick={() => setLayout(item.id)}
          >
            {item.label}
          </button>
        ))}
        <span className="pipeline-tabs-caption">{active.caption}</span>
      </div>
      <div className="dataset-layouts-body">
        <FileTree rows={TREES[layout]} samples={samples} />
        <div className="dataset-layouts-side">
          <LayoutDetail layout={layout} samples={samples} />
          <MaskEncodings samples={samples} />
        </div>
      </div>
    </div>
  );
}

/** The split file from prepare_bbbc038.py, trimmed. */
export function SplitJsonNote() {
  return (
    <pre className="code-block split-json">
      {highlight(`// split.json: seeded, written once
{"seed": 2018,
 "val_fraction": 0.15,
 "train": ["00071198d059…", …570],
 "val":   ["003cee89357d…", …100],
 "test":  ["0114f484a16c…", …65]}`)}
    </pre>
  );
}
