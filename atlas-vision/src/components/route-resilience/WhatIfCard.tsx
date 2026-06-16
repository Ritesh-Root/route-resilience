import { useMemo } from "react";
import { Power, RotateCcw, Zap, AlertTriangle, Loader2, Route } from "lucide-react";
import type { RoadNetwork, SimulationResult, Metrics } from "@/lib/types";
import { criticalityColor } from "@/lib/criticality";

export function WhatIfCard({
  network,
  metrics,
  disabledIds,
  result,
  loading = false,
  error = null,
  onToggle,
  onReset,
}: {
  network: RoadNetwork;
  metrics: Metrics;
  disabledIds: string[];
  /** Backend SimulationResult for the current disabled set; null until simulated. */
  result: SimulationResult | null;
  /** True while a /api/simulate request is in flight (optional; parent-driven). */
  loading?: boolean;
  /** Non-null when the last simulate call failed (optional; parent-driven). */
  error?: string | null;
  onToggle: (id: string) => void;
  onReset: () => void;
}) {
  const topEdges = useMemo(
    () =>
      [...network.features]
        .sort((a, b) => b.properties.criticality - a.properties.criticality)
        .slice(0, 8),
    [network],
  );

  // Total nodes are simulated too; surface a count so the summary subtitle is honest.
  const disabledNodeCount = result?.disabledNodeIds?.length ?? 0;

  // Resilience after the disable set. Until a simulation resolves we show the
  // baseline RI so the tile is never blank.
  const ri = result?.resilienceIndexAfter ?? metrics.resilienceIndex;
  const delta = ri - metrics.resilienceIndex;

  return (
    <div className="space-y-4">
      {/* Async status banner for the in-flight / failed simulate call. */}
      {disabledIds.length > 0 && loading && (
        <div className="flex items-center gap-2 rounded-2xl border border-card-hairline bg-sand px-4 py-2.5 text-[11px] font-semibold text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-coral" />
          Simulating impact of {disabledIds.length} disabled edge
          {disabledIds.length === 1 ? "" : "s"}…
        </div>
      )}
      {error && (
        <div className="flex items-center gap-2 rounded-2xl border border-coral/40 bg-coral-soft px-4 py-2.5 text-[11px] font-semibold text-coral">
          <AlertTriangle className="h-3.5 w-3.5" />
          {error}
        </div>
      )}

      {/* Summary tiles */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-2xl border border-card-hairline bg-sand p-3">
          <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            RI after
          </p>
          <p className="mt-0.5 text-lg font-bold tabular-nums">{ri}</p>
          <p
            className="text-[11px] font-bold tabular-nums"
            style={{ color: delta < 0 ? "#FF7A4E" : "#7A6F65" }}
          >
            {delta === 0 ? "—" : (delta > 0 ? "+" : "") + delta}
          </p>
        </div>
        <div className="rounded-2xl border border-card-hairline bg-sand p-3">
          <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            Travel time
          </p>
          <p className="mt-0.5 text-lg font-bold tabular-nums text-coral">
            +{result?.avgTravelTimeIncreasePct ?? 0}%
          </p>
          <p className="text-[11px] text-muted-foreground">city average</p>
        </div>
        <div className="rounded-2xl border border-card-hairline bg-sand p-3">
          <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            Cut-off zones
          </p>
          <p className="mt-0.5 text-lg font-bold tabular-nums">
            {result?.newlyDisconnectedZones ?? 0}
          </p>
          <p className="text-[11px] text-muted-foreground">newly isolated</p>
        </div>
      </div>

      {/* Broken-routes detail: only meaningful once a simulation has resolved. */}
      {disabledIds.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-card-hairline bg-sand px-4 py-3 text-[11px] text-muted-foreground">
          Disable one or more critical edges below to simulate route impact.
        </div>
      ) : result ? (
        <div className="flex items-center justify-between rounded-2xl border border-card-hairline bg-sand px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="grid h-7 w-7 place-items-center rounded-lg bg-coral-soft text-coral">
              <Route className="h-3.5 w-3.5" />
            </div>
            <div>
              <p className="text-sm font-bold tabular-nums">
                {result.brokenRoutesSampled}
                <span className="text-muted-foreground"> / {result.sampledRoutes}</span>
              </p>
              <p className="text-[11px] text-muted-foreground">
                sampled commutes broken
                {disabledNodeCount > 0
                  ? ` · ${disabledNodeCount} node${disabledNodeCount === 1 ? "" : "s"} disabled`
                  : ""}
              </p>
            </div>
          </div>
          <p
            className="text-[11px] font-bold tabular-nums"
            style={{ color: result.sampledRoutes > 0 ? "#FF7A4E" : "#7A6F65" }}
          >
            {result.sampledRoutes > 0
              ? `${Math.round((result.brokenRoutesSampled / result.sampledRoutes) * 100)}% cut`
              : "—"}
          </p>
        </div>
      ) : null}

      {/* Top edges */}
      <div className="rounded-2xl border border-card-hairline bg-card">
        <div className="flex items-center justify-between border-b border-card-hairline px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="grid h-7 w-7 place-items-center rounded-lg bg-coral-soft text-coral">
              <AlertTriangle className="h-3.5 w-3.5" />
            </div>
            <p className="text-sm font-bold">Top critical edges</p>
          </div>
          <button
            onClick={onReset}
            className="inline-flex items-center gap-1 rounded-full border border-card-hairline bg-sand px-3 py-1 text-[11px] font-semibold text-muted-foreground transition hover:border-coral/40 hover:text-coral"
          >
            <RotateCcw className="h-3 w-3" /> Reset
          </button>
        </div>
        <ul className="max-h-48 overflow-y-auto p-2">
          {topEdges.map((e) => {
            const disabled = disabledIds.includes(e.properties.id);
            const color = criticalityColor(e.properties.criticality);
            return (
              <li
                key={e.properties.id}
                className={[
                  "flex items-center justify-between rounded-xl px-2.5 py-2 transition",
                  disabled ? "opacity-50" : "hover:bg-sand",
                ].join(" ")}
              >
                <div className="flex min-w-0 items-center gap-2.5">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{
                      background: color,
                      boxShadow: `0 0 0 3px ${color}22`,
                    }}
                  />
                  <span className="truncate font-mono text-xs font-semibold">
                    {e.properties.id}
                  </span>
                  <span className="text-[11px] text-muted-foreground">
                    c={e.properties.criticality.toFixed(2)}
                  </span>
                </div>
                <button
                  onClick={() => onToggle(e.properties.id)}
                  className={[
                    "inline-flex items-center gap-1 rounded-full px-3 py-1 text-[11px] font-bold transition",
                    disabled
                      ? "bg-coral text-white shadow-sm shadow-coral/30 hover:bg-coral/90"
                      : "border border-card-hairline bg-card text-muted-foreground hover:border-coral/40 hover:text-coral",
                  ].join(" ")}
                >
                  <Power className="h-3 w-3" />
                  {disabled ? "Enabled" : "Disable"}
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <button
        onClick={onReset}
        disabled={disabledIds.length === 0}
        className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-foreground py-3.5 text-sm font-bold text-background shadow-lg transition active:scale-[0.98] disabled:cursor-not-allowed disabled:bg-card-hairline disabled:text-muted-foreground disabled:shadow-none"
      >
        <Zap className="h-4 w-4" />
        {disabledIds.length === 0
          ? "Disable an edge to simulate"
          : `Reset ${disabledIds.length} disabled edge${disabledIds.length === 1 ? "" : "s"}`}
      </button>
    </div>
  );
}
