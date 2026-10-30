import { useMemo, useState, type PointerEvent } from "react";

import { binaryRgba, instanceColor, labelRgba, microscopyRgba, rgbCss } from "../../utils/maskRender";
import { rasterizePolygon, rleEncode, type Point } from "../../utils/segMetrics";
import { blobPolygon, createScene } from "../../utils/syntheticCells";
import { PixelCanvas } from "./PixelCanvas";

const SIZE = 160;
const CLOSE_DISTANCE = 6;
const RLE_PREVIEW_CHARS = 74;

function svgPoint(event: PointerEvent<SVGSVGElement>): Point {
  const bounds = event.currentTarget.getBoundingClientRect();
  return [((event.clientX - bounds.left) / bounds.width) * SIZE, ((event.clientY - bounds.top) / bounds.height) * SIZE];
}

function pointsAttribute(points: Point[]) {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}

export function AnnotationWidget() {
  const scene = useMemo(() => createScene(2018, SIZE, SIZE, 6, [11, 20]), []);
  const imagePixels = useMemo(() => microscopyRgba(scene.image), [scene]);
  const [polygons, setPolygons] = useState<Point[][]>([]);
  const [draft, setDraft] = useState<Point[]>([]);
  const [cursor, setCursor] = useState<Point | null>(null);

  const labels = useMemo(() => {
    const map = new Int32Array(SIZE * SIZE);
    polygons.forEach((polygon, index) => {
      const single = rasterizePolygon(polygon, SIZE, SIZE);
      single.forEach((value, pixel) => {
        if (value) map[pixel] = index + 1;
      });
    });
    return map;
  }, [polygons]);
  const binaryPixels = useMemo(() => binaryRgba(Uint8Array.from(labels, (value) => (value > 0 ? 1 : 0))), [labels]);
  const labelPixels = useMemo(() => labelRgba(labels), [labels]);

  const lastRle = useMemo(() => {
    if (polygons.length === 0) {
      return null;
    }
    const id = polygons.length;
    const mask = Uint8Array.from(labels, (value) => (value === id ? 1 : 0));
    const runs = rleEncode(mask, SIZE, SIZE);
    return { id, text: runs.join(" "), runs: runs.length / 2, area: mask.reduce((sum, value) => sum + value, 0) };
  }, [labels, polygons.length]);

  const nearFirst =
    draft.length >= 3 && cursor !== null && Math.hypot(cursor[0] - draft[0][0], cursor[1] - draft[0][1]) < CLOSE_DISTANCE;

  const closeDraft = () => {
    if (draft.length >= 3) {
      setPolygons((current) => [...current, draft]);
    }
    setDraft([]);
  };

  const handlePointerDown = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) {
      return;
    }
    const point = svgPoint(event);
    if (draft.length >= 3 && Math.hypot(point[0] - draft[0][0], point[1] - draft[0][1]) < CLOSE_DISTANCE) {
      closeDraft();
      return;
    }
    setDraft((current) => [...current, point]);
  };

  // Model-assisted: outline every nucleus not yet labelled, as a SAM3-style pseudo-label would.
  const autoAnnotate = () => {
    const missing = scene.blobs.filter((blob) => labels[Math.floor(blob.cy) * SIZE + Math.floor(blob.cx)] === 0);
    setPolygons((current) => [...current, ...missing.map((blob) => blobPolygon(blob, 18, 1.04))]);
    setDraft([]);
  };

  return (
    <div className="annotation-widget">
      <div className="annotation-stage">
        <PixelCanvas width={SIZE} height={SIZE} pixels={imagePixels} smooth label="Synthetic microscopy image" />
        <svg
          className={`annotation-overlay${nearFirst ? " is-closing" : ""}`}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          onPointerDown={handlePointerDown}
          onPointerMove={(event) => setCursor(svgPoint(event))}
          onPointerLeave={() => setCursor(null)}
          onDoubleClick={closeDraft}
        >
          {polygons.map((polygon, index) => (
            <polygon
              key={index}
              points={pointsAttribute(polygon)}
              fill={rgbCss(instanceColor(index + 1), 0.28)}
              stroke={rgbCss(instanceColor(index + 1))}
              strokeWidth={1.1}
            />
          ))}
          {draft.length > 0 ? (
            <polyline
              points={pointsAttribute(cursor ? [...draft, cursor] : draft)}
              fill="none"
              className="annotation-draft-line"
            />
          ) : null}
          {draft.map(([x, y], index) => (
            <circle
              key={index}
              cx={x}
              cy={y}
              r={index === 0 && nearFirst ? 3.4 : 1.9}
              className={index === 0 ? "annotation-vertex is-first" : "annotation-vertex"}
            />
          ))}
        </svg>
        <span className="annotation-hint">
          {draft.length === 0
            ? "Click around a nucleus to place vertices"
            : draft.length < 3
              ? `${draft.length} point${draft.length === 1 ? "" : "s"} — keep going`
              : "Click the first point (or double-click) to close"}
        </span>
      </div>

      <div className="annotation-outputs">
        <div className="annotation-toolbar">
          <button type="button" className="secondary-button" onClick={closeDraft} disabled={draft.length < 3}>
            Close polygon
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => (draft.length ? setDraft((current) => current.slice(0, -1)) : setPolygons((current) => current.slice(0, -1)))}
            disabled={draft.length === 0 && polygons.length === 0}
          >
            Undo
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => {
              setPolygons([]);
              setDraft([]);
            }}
            disabled={draft.length === 0 && polygons.length === 0}
          >
            Clear
          </button>
          <button type="button" className="secondary-button is-assist" onClick={autoAnnotate}>
            Model-assist
          </button>
        </div>

        <div className="annotation-tiles">
          <figure className="pipeline-tile">
            <PixelCanvas width={SIZE} height={SIZE} pixels={binaryPixels} label="Binary mask" />
            <figcaption>
              <strong>Binary mask</strong>
              <span>0 = background, 1 = nucleus</span>
            </figcaption>
          </figure>
          <figure className="pipeline-tile">
            <PixelCanvas width={SIZE} height={SIZE} pixels={labelPixels} label="Instance label map" />
            <figcaption>
              <strong>Instance IDs</strong>
              <span>{polygons.length ? `1 … ${polygons.length}, one integer per object` : "one integer per object"}</span>
            </figcaption>
          </figure>
        </div>

        <div className="annotation-rle">
          <strong>
            RLE{" "}
            {lastRle ? (
              <span style={{ color: rgbCss(instanceColor(lastRle.id)) }}>{`object #${lastRle.id}`}</span>
            ) : null}
          </strong>
          <code>
            {lastRle
              ? `${lastRle.text.slice(0, RLE_PREVIEW_CHARS)}${lastRle.text.length > RLE_PREVIEW_CHARS ? " …" : ""}`
              : "start length start length …"}
          </code>
          <span className="field-note">
            {lastRle
              ? `${lastRle.runs} runs encode ${lastRle.area} pixels of a ${SIZE}×${SIZE} image`
              : "Close a polygon to encode it"}
          </span>
        </div>
      </div>
    </div>
  );
}
