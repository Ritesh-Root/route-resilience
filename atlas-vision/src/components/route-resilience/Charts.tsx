import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

const tooltipStyle = {
  background: "#FFFFFF",
  border: "1px solid #EBE3D5",
  borderRadius: 12,
  color: "#1F1B16",
  fontSize: 12,
  boxShadow: "0 8px 24px -8px rgba(31,27,22,0.18)",
};

/**
 * Internal row shape the chart actually renders. `removed` is a percentage of
 * edges removed (0..100); `efficiency` / `giant` are percentages (0..100).
 */
type CurveRow = { removed: number; efficiency: number; giant: number };

/**
 * The two shapes `fetchResilienceCurve` can resolve to:
 *  - backend contract: parallel arrays, fractions in 0..1
 *  - legacy mock fallback (RESILIENCE_CURVE): already-normalised CurveRow[]
 */
type BackendCurve = {
  removedFraction: number[];
  efficiency: number[];
  giantComponent: number[];
};
export type ResilienceCurveData = BackendCurve | CurveRow[] | null | undefined;

/**
 * Normalise whichever shape the API layer hands us into CurveRow[].
 * Backend fractions (0..1) are scaled to percentages so both code paths share
 * one axis domain; the mock path is already in 0..100 and passes through.
 */
function toCurveRows(data: ResilienceCurveData): CurveRow[] {
  if (!data) return [];
  if (Array.isArray(data)) {
    // Legacy mock fallback: already { removed, efficiency, giant } in 0..100.
    return data;
  }
  const { removedFraction = [], efficiency = [], giantComponent = [] } = data;
  const n = Math.min(
    removedFraction.length,
    efficiency.length,
    giantComponent.length,
  );
  // Defensive clamp: inputs are expected in 0..1, but a stray >1 (or <0) value
  // must not render past 100% (or below 0%).
  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));
  const rows: CurveRow[] = [];
  for (let i = 0; i < n; i++) {
    rows.push({
      removed: +(clamp01(removedFraction[i]) * 100).toFixed(1),
      efficiency: +(clamp01(efficiency[i]) * 100).toFixed(1),
      giant: +(clamp01(giantComponent[i]) * 100).toFixed(1),
    });
  }
  return rows;
}

/** Centered placeholder used for loading / empty / error states. */
function ChartPlaceholder({ message }: { message: string }) {
  return (
    <div className="flex h-44 w-full items-center justify-center text-center text-xs font-medium text-[#7A6F65]">
      {message}
    </div>
  );
}

export function ResilienceChart({
  data,
  isLoading = false,
  isError = false,
}: {
  data: ResilienceCurveData;
  isLoading?: boolean;
  isError?: boolean;
}) {
  if (isLoading) return <ChartPlaceholder message="Loading resilience curve…" />;
  if (isError)
    return <ChartPlaceholder message="Couldn’t load resilience curve." />;

  const rows = toCurveRows(data);
  if (rows.length === 0)
    return <ChartPlaceholder message="No resilience data available." />;

  return (
    <div className="h-44 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={rows} margin={{ top: 6, right: 6, left: -16, bottom: 6 }}>
          <defs>
            <linearGradient id="effGrad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#FF7A4E" stopOpacity={0.45} />
              <stop offset="100%" stopColor="#FF7A4E" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="giantGrad" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#2D8C7D" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#2D8C7D" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#EBE3D5" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="removed"
            tick={{ fill: "#7A6F65", fontSize: 10 }}
            stroke="#EBE3D5"
          />
          <YAxis tick={{ fill: "#7A6F65", fontSize: 10 }} stroke="#EBE3D5" />
          <Tooltip
            contentStyle={tooltipStyle}
            cursor={{ stroke: "#FF7A4E", strokeOpacity: 0.4 }}
          />
          <Area
            type="monotone"
            dataKey="efficiency"
            name="Global efficiency"
            stroke="#FF7A4E"
            fill="url(#effGrad)"
            strokeWidth={2.2}
          />
          <Area
            type="monotone"
            dataKey="giant"
            name="Giant component"
            stroke="#2D8C7D"
            fill="url(#giantGrad)"
            strokeWidth={2.2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CriticalityHistogram({
  data,
  isLoading = false,
  isError = false,
}: {
  data: { bucket: string; count: number }[] | null | undefined;
  isLoading?: boolean;
  isError?: boolean;
}) {
  if (isLoading)
    return <ChartPlaceholder message="Loading criticality distribution…" />;
  if (isError)
    return <ChartPlaceholder message="Couldn’t load criticality distribution." />;
  if (!data || data.length === 0)
    return <ChartPlaceholder message="No criticality data available." />;

  const maxCount = Math.max(...data.map((d) => d.count), 1);
  const maxSegments = 12;

  const palette = [
    { fill: "#B5D6B2", dim: "#B5D6B240" },
    { fill: "#DDE4A9", dim: "#DDE4A940" },
    { fill: "#FEE291", dim: "#FEE29140" },
    { fill: "#FFBD82", dim: "#FFBD8240" },
    { fill: "#FF9A7B", dim: "#FF9A7B40" },
  ];

  const yTicks = [maxCount, Math.round(maxCount * 0.75), Math.round(maxCount * 0.5), Math.round(maxCount * 0.25), 0];

  return (
    <div className="relative h-44 w-full">
      {/* Y-axis labels */}
      <div className="absolute left-0 top-0 bottom-8 w-6 flex flex-col justify-between text-[10px] font-medium text-[#7A6F65] text-right pr-1">
        {yTicks.map((t) => (
          <span key={t}>{t}</span>
        ))}
      </div>

      {/* Grid lines */}
      <div className="absolute left-7 right-0 top-0 bottom-8 flex flex-col justify-between pointer-events-none">
        {yTicks.map((t, i) => (
          <div
            key={t}
            className={`w-full border-t ${i === yTicks.length - 1 ? "border-[#D8CFC4]" : "border-dashed border-[#EBE3D5]"}`}
          />
        ))}
      </div>

      {/* Bars + X-axis in one shared column structure so widths align perfectly */}
      <div className="absolute left-7 right-0 top-0 bottom-0 flex justify-around px-2">
        {data.map((d, i) => {
          const color = palette[i] ?? palette[palette.length - 1];
          const segments = Math.round((d.count / maxCount) * maxSegments);
          return (
            <div key={d.bucket} className="group flex flex-col items-center w-12 h-full">
              {/* Bar area — fills all space above x-axis */}
              <div className="flex-1 w-full flex flex-col-reverse justify-start gap-[3px] pb-2">
                {Array.from({ length: maxSegments }).map((_, segIdx) => {
                  const isActive = segIdx < segments;
                  return (
                    <div
                      key={segIdx}
                      className="w-full rounded-full transition-all duration-200"
                      style={{
                        flex: 1,
                        minHeight: 0,
                        backgroundColor: isActive ? color.fill : color.dim,
                        opacity: isActive ? 1 : 0.35,
                      }}
                    />
                  );
                })}
              </div>

              {/* Tooltip */}
              <div className="absolute -top-1 text-[10px] font-semibold text-[#7A6F65] opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                {d.count}
              </div>

              {/* X-axis label */}
              <div className="h-8 flex items-center justify-center text-[10px] font-medium text-[#7A6F65] w-full text-center shrink-0">
                <span className="transition-colors group-hover:text-[#1F1B16] cursor-default">
                  {d.bucket}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
