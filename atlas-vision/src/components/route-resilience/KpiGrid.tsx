import { resilienceColor } from "@/lib/criticality";
import type { Metrics } from "@/lib/types";

function Kpi({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "coral" | "mint" | "amber" | "default";
}) {
  const toneClass =
    tone === "coral"
      ? "text-coral"
      : tone === "mint"
        ? "text-mint"
        : tone === "amber"
          ? "text-amber"
          : "text-foreground";
  return (
    <div className="rounded-2xl border border-card-hairline bg-sand p-4">
      <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      <p className={`mt-1 text-xl font-bold tabular-nums ${toneClass}`}>{value}</p>
      {hint && <p className="text-[11px] text-muted-foreground">{hint}</p>}
    </div>
  );
}

// Placeholder shown when a backend metric is null/undefined/NaN.
const MISSING = "—";

/** Format a 0..1 ratio with fixed decimals, tolerating missing values. */
function fmtRatio(n: number | null | undefined, digits = 2): string {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits) : MISSING;
}

/** Format a 0..1 value as a whole percentage, tolerating missing values. */
function fmtPct(n: number | null | undefined): string {
  return typeof n === "number" && Number.isFinite(n)
    ? `${(n * 100).toFixed(0)}%`
    : MISSING;
}

export function KpiGrid({ metrics }: { metrics?: Metrics | null }) {
  // The parent renders a skeleton while loading, but guard anyway so a
  // partial/empty payload from the backend never crashes the panel.
  if (!metrics) {
    return (
      <div className="rounded-2xl border border-card-hairline bg-sand p-4 text-sm text-muted-foreground">
        No metrics available.
      </div>
    );
  }

  const rawRi = metrics.resilienceIndex;
  const hasRi = typeof rawRi === "number" && Number.isFinite(rawRi);
  // Clamp to 0..100 for the gauge geometry; fall back to 0 when absent.
  const ri = hasRi ? Math.max(0, Math.min(100, rawRi)) : 0;
  const color = resilienceColor(ri);
  const label = !hasRi
    ? "Unknown"
    : ri >= 80
      ? "Resilient"
      : ri >= 60
        ? "Moderate"
        : "Fragile";

  return (
    <div className="space-y-4">
      {/* Hero Resilience Index */}
      <div className="relative overflow-hidden rounded-[24px] border border-card-hairline bg-card p-6">
        <div className="relative z-10">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Resilience index
            </p>
          </div>
          <div className="mt-2 flex items-baseline gap-3">
            <span className="text-5xl font-extrabold tabular-nums" style={{ color }}>
              {hasRi ? ri : MISSING}
            </span>
            <span className="text-sm font-semibold text-muted-foreground">
              / 100 · {label}
            </span>
          </div>
          <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-sand">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${ri}%`, background: color }}
            />
          </div>
        </div>
        {/* decorative coral glow */}
        <div className="pointer-events-none absolute -bottom-16 -right-16 h-44 w-44 rounded-full bg-coral/10 blur-3xl" />
        <svg
          className="pointer-events-none absolute -right-2 top-3 opacity-25"
          width="64"
          height="64"
          viewBox="0 0 100 100"
        >
          <path
            d="M10,90 Q50,10 90,90"
            stroke="#FF7A4E"
            fill="none"
            strokeWidth="2"
            strokeDasharray="4 4"
          />
        </svg>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Kpi label="IoU" value={fmtRatio(metrics.iou)} hint="Mask overlap" />
        <Kpi label="Dice" value={fmtRatio(metrics.dice)} hint="F1 similarity" />
        <Kpi
          label="Occlusion recall"
          value={fmtPct(metrics.occlusionRecall)}
          hint="Roads under cover"
          tone="mint"
        />
        <Kpi
          label="Connectivity"
          value={fmtRatio(metrics.connectivityRatio)}
          hint="Graph cohesion"
        />
        <Kpi
          label="APLS"
          value={fmtRatio(metrics.apls)}
          hint="Topological accuracy"
        />
        <Kpi
          label="Status"
          value={!hasRi ? MISSING : ri >= 70 ? "Stable" : "Stressed"}
          hint="Live system"
          tone={!hasRi ? "default" : ri >= 70 ? "mint" : "coral"}
        />
      </div>
    </div>
  );
}
