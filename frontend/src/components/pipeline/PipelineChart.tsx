import { useEffect, useMemo, useRef, useState } from "react";

export interface ChartPoint {
  x: number;
  y: number;
}

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
  points: ChartPoint[];
  dashed?: boolean;
  /** Hide the series from the legend (e.g. reference lines). */
  hideLegend?: boolean;
}

export interface ChartMarker {
  key: string;
  x: number;
  y: number;
  color: string;
  label?: string;
}

interface PipelineChartProps {
  title: string;
  subtitle?: string;
  series: ChartSeries[];
  markers?: ChartMarker[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  formatX?: (value: number) => string;
  formatY?: (value: number) => string;
  height?: number;
  /** Draw the plot square (equal x/y scale), e.g. for Dice vs IoU. */
  square?: boolean;
}

const MARGIN = { top: 12, right: 18, bottom: 38, left: 48 };

// Same tick heuristic as NucleiLineChart, for a numeric x axis with many points.
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

/** Line chart with a numeric x axis; reuses the Nuclei Demo chart styling. */
export function PipelineChart({
  title,
  subtitle,
  series,
  markers = [],
  xDomain,
  yDomain,
  xLabel,
  formatX = (value) => String(value),
  formatY = (value) => value.toFixed(2),
  height = 200,
  square = false,
}: PipelineChartProps) {
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(420);

  useEffect(() => {
    const element = plotRef.current;
    if (!element) {
      return;
    }
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) {
        setWidth(Math.max(220, Math.round(entry.contentRect.width)));
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const svgWidth = square ? Math.min(width, height - MARGIN.top - MARGIN.bottom + MARGIN.left + MARGIN.right) : width;

  const geometry = useMemo(() => {
    const plotWidth = svgWidth - MARGIN.left - MARGIN.right;
    const plotHeight = height - MARGIN.top - MARGIN.bottom;
    const x = (value: number) => MARGIN.left + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * plotWidth;
    const y = (value: number) =>
      MARGIN.top + plotHeight - ((Math.min(Math.max(value, yDomain[0]), yDomain[1]) - yDomain[0]) / (yDomain[1] - yDomain[0])) * plotHeight;
    return { x, y, plotWidth, plotHeight, xTicks: niceTicks(xDomain[0], xDomain[1]), yTicks: niceTicks(yDomain[0], yDomain[1]) };
  }, [svgWidth, height, xDomain, yDomain]);

  return (
    <figure className={`nuclei-chart pipeline-chart${square ? " is-square" : ""}`}>
      <figcaption className="nuclei-chart-heading">
        <strong>{title}</strong>
        {subtitle ? <span>{subtitle}</span> : null}
      </figcaption>
      <ul className="nuclei-chart-legend">
        {series
          .filter((item) => !item.hideLegend)
          .map((item) => (
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
        <svg viewBox={`0 0 ${svgWidth} ${height}`} width={svgWidth} height={height} role="img" aria-label={title}>
          {geometry.yTicks.map((tick) => (
            <g key={`y${tick}`}>
              <line
                className="nuclei-chart-gridline"
                x1={MARGIN.left}
                x2={svgWidth - MARGIN.right}
                y1={geometry.y(tick)}
                y2={geometry.y(tick)}
              />
              <text className="nuclei-chart-tick" x={MARGIN.left - 8} y={geometry.y(tick) + 4} textAnchor="end">
                {formatY(tick)}
              </text>
            </g>
          ))}
          {geometry.xTicks.map((tick) => (
            <text
              key={`x${tick}`}
              className="nuclei-chart-tick"
              x={geometry.x(tick)}
              y={height - MARGIN.bottom + 17}
              textAnchor="middle"
            >
              {formatX(tick)}
            </text>
          ))}
          <text
            className="nuclei-chart-axis-label"
            x={MARGIN.left + geometry.plotWidth / 2}
            y={height - 4}
            textAnchor="middle"
          >
            {xLabel}
          </text>

          {series.map((item) => (
            <polyline
              key={item.key}
              points={item.points.map((point) => `${geometry.x(point.x)},${geometry.y(point.y)}`).join(" ")}
              fill="none"
              stroke={item.color}
              strokeWidth={2.25}
              strokeLinejoin="round"
              strokeLinecap="round"
              strokeDasharray={item.dashed ? "5 4" : undefined}
            />
          ))}
          {markers.map((marker) => (
            <g key={marker.key}>
              <circle
                cx={geometry.x(marker.x)}
                cy={geometry.y(marker.y)}
                r={6}
                fill={marker.color}
                stroke="var(--chart-surface)"
                strokeWidth={2}
              />
              {marker.label ? (
                <text
                  className="nuclei-chart-end-label"
                  x={geometry.x(marker.x) + (geometry.x(marker.x) > svgWidth - 110 ? -10 : 10)}
                  y={geometry.y(marker.y) - 8}
                  textAnchor={geometry.x(marker.x) > svgWidth - 110 ? "end" : "start"}
                >
                  {marker.label}
                </text>
              ) : null}
            </g>
          ))}
        </svg>
      </div>
    </figure>
  );
}
