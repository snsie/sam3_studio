import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";

export interface LineChartSeries {
  key: string;
  label: string;
  color: string;
  values: Array<number | null>;
  dashed?: boolean;
  /** Index of a point to emphasise (e.g. the selected checkpoint). */
  markIndex?: number | null;
}

interface NucleiLineChartProps {
  title: string;
  subtitle?: string;
  xValues: number[];
  series: LineChartSeries[];
  formatX: (value: number) => string;
  formatY: (value: number) => string;
  yDomain?: [number, number];
  xLabel: string;
}

const DEFAULT_WIDTH = 560;
const HEIGHT = 230;
const MARGIN = { top: 14, right: 64, bottom: 34, left: 44 };
const LABEL_GAP = 13;

function niceTicks(min: number, max: number, count = 4): number[] {
  if (max <= min) {
    return [min];
  }
  const rawStep = (max - min) / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((value) => value >= rawStep) ?? rawStep;
  const ticks: number[] = [];
  for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-6; value += step) {
    ticks.push(Number(value.toFixed(6)));
  }
  return ticks;
}

function spreadLabels(positions: Array<{ key: string; y: number }>) {
  const sorted = [...positions].sort((left, right) => left.y - right.y);
  for (let index = 1; index < sorted.length; index += 1) {
    if (sorted[index].y - sorted[index - 1].y < LABEL_GAP) {
      sorted[index].y = sorted[index - 1].y + LABEL_GAP;
    }
  }
  return new Map(sorted.map((item) => [item.key, item.y]));
}

export function NucleiLineChart({
  title,
  subtitle,
  xValues,
  series,
  formatX,
  formatY,
  yDomain,
  xLabel,
}: NucleiLineChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  // Draw in real pixels so text and marks keep their size at any container width.
  const [WIDTH, setWidth] = useState(DEFAULT_WIDTH);

  useEffect(() => {
    const element = plotRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(280, Math.round(entry.contentRect.width)));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const geometry = useMemo(() => {
    const allValues = series.flatMap((item) => item.values.filter((value): value is number => value !== null));
    const yMin = yDomain ? yDomain[0] : Math.min(0, ...allValues);
    const yMax = yDomain ? yDomain[1] : Math.max(...allValues, yMin + 1e-6) * 1.08;
    const xMin = xValues[0] ?? 0;
    const xMax = xValues[xValues.length - 1] ?? 1;
    const plotWidth = WIDTH - MARGIN.left - MARGIN.right;
    const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom;
    const x = (value: number) =>
      MARGIN.left + (xMax === xMin ? plotWidth / 2 : ((value - xMin) / (xMax - xMin)) * plotWidth);
    const y = (value: number) => MARGIN.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;
    return { x, y, yTicks: niceTicks(yMin, yMax), plotHeight, plotWidth };
  }, [WIDTH, series, xValues, yDomain]);

  const endLabels = useMemo(() => {
    const positions = series
      .map((item) => {
        const lastIndex = item.values.map((value) => value !== null).lastIndexOf(true);
        return lastIndex < 0 ? null : { key: item.key, y: geometry.y(item.values[lastIndex] as number) };
      })
      .filter((item): item is { key: string; y: number } => item !== null);
    return spreadLabels(positions);
  }, [geometry, series]);

  const handlePointerMove = (event: PointerEvent<SVGRectElement>) => {
    const svg = svgRef.current;
    if (!svg || xValues.length === 0) {
      return;
    }
    const bounds = svg.getBoundingClientRect();
    const viewX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    let nearest = 0;
    xValues.forEach((value, index) => {
      if (Math.abs(geometry.x(value) - viewX) < Math.abs(geometry.x(xValues[nearest]) - viewX)) {
        nearest = index;
      }
    });
    setHoverIndex(nearest);
  };

  const tooltipLeft = hoverIndex === null ? 0 : (geometry.x(xValues[hoverIndex]) / WIDTH) * 100;

  return (
    <figure className="nuclei-chart">
      <figcaption className="nuclei-chart-heading">
        <strong>{title}</strong>
        {subtitle ? <span>{subtitle}</span> : null}
      </figcaption>
      <ul className="nuclei-chart-legend">
        {series.map((item) => (
          <li key={item.key}>
            <span
              className={`nuclei-legend-swatch${item.dashed ? " is-dashed" : ""}`}
              style={{ borderColor: item.color }}
              aria-hidden="true"
            />
            {item.label}
          </li>
        ))}
      </ul>
      <div className="nuclei-chart-plot" ref={plotRef}>
        <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={title}>
          {geometry.yTicks.map((tick) => (
            <g key={tick}>
              <line
                className="nuclei-chart-gridline"
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={geometry.y(tick)}
                y2={geometry.y(tick)}
              />
              <text className="nuclei-chart-tick" x={MARGIN.left - 8} y={geometry.y(tick) + 4} textAnchor="end">
                {formatY(tick)}
              </text>
            </g>
          ))}
          {xValues.map((value) => (
            <text
              key={value}
              className="nuclei-chart-tick"
              x={geometry.x(value)}
              y={HEIGHT - MARGIN.bottom + 16}
              textAnchor="middle"
            >
              {formatX(value)}
            </text>
          ))}
          <text className="nuclei-chart-axis-label" x={MARGIN.left + geometry.plotWidth / 2} y={HEIGHT - 4} textAnchor="middle">
            {xLabel}
          </text>

          {hoverIndex !== null ? (
            <line
              className="nuclei-chart-crosshair"
              x1={geometry.x(xValues[hoverIndex])}
              x2={geometry.x(xValues[hoverIndex])}
              y1={MARGIN.top}
              y2={MARGIN.top + geometry.plotHeight}
            />
          ) : null}

          {series.map((item) => {
            const points = item.values
              .map((value, index) => (value === null ? null : `${geometry.x(xValues[index])},${geometry.y(value)}`))
              .filter((point): point is string => point !== null);
            const lastIndex = item.values.map((value) => value !== null).lastIndexOf(true);
            return (
              <g key={item.key}>
                <polyline
                  points={points.join(" ")}
                  fill="none"
                  stroke={item.color}
                  strokeWidth={2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  strokeDasharray={item.dashed ? "5 4" : undefined}
                />
                {item.values.map((value, index) =>
                  value === null ? null : (
                    <circle
                      key={index}
                      className="nuclei-chart-marker"
                      cx={geometry.x(xValues[index])}
                      cy={geometry.y(value)}
                      r={index === item.markIndex ? 6 : hoverIndex === index ? 5 : 4}
                      fill={index === item.markIndex ? item.color : "var(--chart-surface)"}
                      stroke={item.color}
                      strokeWidth={2}
                    />
                  ),
                )}
                {lastIndex >= 0 ? (
                  <text
                    className="nuclei-chart-end-label"
                    x={geometry.x(xValues[lastIndex]) + 9}
                    y={(endLabels.get(item.key) ?? 0) + 4}
                  >
                    {formatY(item.values[lastIndex] as number)}
                  </text>
                ) : null}
              </g>
            );
          })}
          <rect
            x={MARGIN.left - 10}
            y={MARGIN.top}
            width={geometry.plotWidth + 20}
            height={geometry.plotHeight}
            fill="transparent"
            onPointerMove={handlePointerMove}
            onPointerLeave={() => setHoverIndex(null)}
          />
        </svg>
        {hoverIndex !== null ? (
          <div
            className={`nuclei-chart-tooltip${tooltipLeft > 60 ? " is-left" : ""}`}
            style={{ left: `${tooltipLeft}%` }}
          >
            <strong>{`${xLabel} ${formatX(xValues[hoverIndex])}`}</strong>
            {series.map((item) => {
              const value = item.values[hoverIndex];
              return (
                <span key={item.key}>
                  <i style={{ background: item.color }} aria-hidden="true" />
                  {item.label}
                  <b>{value === null ? "–" : formatY(value)}</b>
                </span>
              );
            })}
          </div>
        ) : null}
      </div>
    </figure>
  );
}
