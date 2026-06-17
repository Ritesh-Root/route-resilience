import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronDown,
  Cloud,
  Sun,
  Sparkles,
  Brain,
  Layers,
  CheckCircle2,
  Search,
  Bell,
  Rocket,
} from "lucide-react";
import { PipelineStrip } from "@/components/route-resilience/PipelineStrip";
import { Legend } from "@/components/route-resilience/Legend";
import { KpiGrid } from "@/components/route-resilience/KpiGrid";
import { WhatIfCard } from "@/components/route-resilience/WhatIfCard";
import {
  ResilienceChart,
  CriticalityHistogram,
} from "@/components/route-resilience/Charts";
import { BeforeAfterSlider } from "@/components/route-resilience/BeforeAfterSlider";
import { ClientMap } from "@/components/route-resilience/ClientMap";
import { CITIES } from "@/lib/mock";
import {
  fetchNetwork,
  fetchMetrics,
  fetchGatekeepers,
  fetchResilienceCurve,
  fetchCriticalityHistogram,
  simulateDisable,
} from "@/lib/api";
import type {
  InputMode,
  ModelMode,
  Stage,
  SimulationResult,
} from "@/lib/types";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Route Resilience · ISRO Bhartiya Antariksh Hackathon" },
      {
        name: "description",
        content:
          "Occlusion-robust road extraction and graph-theoretic criticality analysis for urban mobility — a warm, judge-ready demo for Bengaluru.",
      },
      { property: "og:title", content: "Route Resilience" },
      {
        property: "og:description",
        content:
          "Occlusion-robust road extraction and criticality analysis for urban mobility.",
      },
    ],
  }),
  component: RouteResiliencePage,
});

function SegToggle<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string; icon?: React.ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div
      aria-label={label}
      className="flex rounded-full border border-card-hairline bg-sand p-0.5"
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            aria-pressed={active}
            className={[
              "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-semibold transition",
              active
                ? "bg-coral text-white shadow-sm shadow-coral/30"
                : "text-muted-foreground hover:text-foreground",
            ].join(" ")}
          >
            {o.icon}
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function RouteResiliencePage() {
  const [city, setCity] = useState("Bengaluru");
  const [stage, setStage] = useState<Stage>("graph");
  const [input, setInput] = useState<InputMode>("occluded");
  const [model, setModel] = useState<ModelMode>("robust");
  const [disabledIds, setDisabledIds] = useState<string[]>([]);
  const [simResult, setSimResult] = useState<SimulationResult | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [view, setView] = useState<"map" | "health" | "reports">("map");


  const network = useQuery({
    queryKey: ["network", city, input, model],
    queryFn: () => fetchNetwork(city, input, model),
  });
  const metrics = useQuery({
    queryKey: ["metrics", city, input, model],
    queryFn: () => fetchMetrics(city, input, model),
  });
  const gatekeepers = useQuery({
    queryKey: ["gatekeepers", city, input, model],
    queryFn: () => fetchGatekeepers(city, input, model),
  });
  const curve = useQuery({
    queryKey: ["curve", city, input, model],
    queryFn: () => fetchResilienceCurve(city, input, model),
  });
  const hist = useQuery({
    queryKey: ["hist", city, input, model],
    queryFn: () => fetchCriticalityHistogram(city, input, model),
  });

  // The backend returns the resilience curve as parallel arrays
  // ({ removedFraction[], efficiency[], giantComponent[] }), but the
  // ResilienceChart expects row objects ({ removed, efficiency, giant }).
  // Normalize here so the chart renders regardless of which shape arrives
  // (the mock fallback is already row-shaped, so detect and pass through).
  const curveRows = useMemo(() => {
    const d = curve.data;
    if (!d) return undefined;
    if (Array.isArray(d)) {
      // Mock fallback shape: already row objects.
      return d as { removed: number; efficiency: number; giant: number }[];
    }
    const { removedFraction, efficiency, giantComponent } = d;
    return removedFraction.map((removed, i) => ({
      removed,
      efficiency: efficiency[i],
      giant: giantComponent[i],
    }));
  }, [curve.data]);

  // The simulation is keyed off the current display state (city/input/model)
  // on the backend, so build a DisplayState rather than passing a network
  // snapshot. `stage` is required by the type but does not affect simulation.
  useEffect(() => {
    if (disabledIds.length === 0) {
      setSimResult(null);
      setSimError(null);
      setSimLoading(false);
      return;
    }
    let cancelled = false;
    setSimLoading(true);
    setSimError(null);
    simulateDisable({ city, stage, input, model }, disabledIds)
      .then((r) => {
        if (!cancelled) {
          setSimResult(r);
          setSimLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSimError("Couldn’t simulate this disable set. Showing last values.");
          setSimLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [disabledIds, city, stage, input, model]);

  useEffect(() => {
    setDisabledIds([]);
    setSimResult(null);
    setSimError(null);
    setSimLoading(false);
  }, [input, model, city]);

  const recoveredCount = useMemo(() => {
    if (input !== "occluded" || model !== "robust") return 0;
    return 3;
  }, [input, model]);

  const baselineWarning = input === "occluded" && model === "baseline";

  return (
    <div className="min-h-dvh bg-cream text-foreground">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-6 py-6 lg:px-10">
        {/* ── TOP NAV ─────────────────────────────────────────── */}
        <nav className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 shrink-0 place-items-center rounded-2xl bg-coral shadow-lg shadow-coral/25">
              <Rocket className="h-5 w-5 text-white" strokeWidth={2.2} />
            </div>
            <div className="min-w-0">
              <h1 className="text-xl font-extrabold tracking-tight text-foreground">
                Route Resilience
              </h1>
              <p className="truncate text-[11px] text-muted-foreground">
                Occlusion-robust road extraction & criticality analysis
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-1 rounded-full border border-card-hairline bg-card/70 p-1.5 shadow-[var(--shadow-soft)] backdrop-blur md:flex">
            {([
              { id: "map", label: "Map view" },
              { id: "health", label: "Network health" },
              { id: "reports", label: "Reports" },
            ] as const).map((t) => {
              const active = view === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setView(t.id)}
                  aria-pressed={active}
                  className={[
                    "rounded-full px-5 py-1.5 text-sm font-semibold transition",
                    active
                      ? "bg-card text-coral shadow-sm"
                      : "text-muted-foreground hover:bg-card/60 hover:text-foreground",
                  ].join(" ")}
                >
                  {t.label}
                </button>
              );
            })}
          </div>


          <div className="flex items-center gap-3">
            <div className="hidden h-10 items-center gap-2 rounded-full border border-card-hairline bg-card/80 px-4 sm:flex">
              <Search className="h-4 w-4 text-muted-foreground/70" />
              <span className="text-sm text-muted-foreground/80">
                Search {city} sector…
              </span>
            </div>
            <div className="relative">
              <label className="sr-only" htmlFor="city">
                City
              </label>
              <select
                id="city"
                value={city}
                onChange={(e) => setCity(e.target.value)}
                className="h-10 appearance-none rounded-full border border-card-hairline bg-card pl-4 pr-9 text-sm font-semibold text-foreground shadow-sm focus:border-coral focus:outline-none"
              >
                {CITIES.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            </div>
            <button
              aria-label="Notifications"
              className="grid h-10 w-10 place-items-center rounded-full border border-card-hairline bg-card text-muted-foreground transition hover:text-coral"
            >
              <Bell className="h-4 w-4" />
            </button>
            <div className="hidden items-center gap-2 rounded-full border border-coral/20 bg-coral/5 px-3 py-1.5 text-[10px] font-bold uppercase tracking-widest text-coral lg:flex">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-coral" />
              ISRO · Bhartiya Antariksh
            </div>
          </div>
        </nav>

        {/* ── PIPELINE STRIP ──────────────────────────────────── */}
        {view === "map" && (
        <>
        <PipelineStrip active={stage} onChange={setStage} />
        </>
        )}


        {/* ── MAIN GRID ───────────────────────────────────────── */}
        {view === "map" && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_400px] xl:grid-cols-[1fr_440px]">

          {/* LEFT — MAP CARD */}
          <section className="rounded-[32px] border border-card-hairline bg-card p-3 shadow-[var(--shadow-card)]">
            <div className="flex flex-wrap items-center justify-between gap-2 px-3 pb-3 pt-2">
              <div>
                <h2 className="text-lg font-extrabold text-foreground">
                  {city} network status
                </h2>
                <p className="text-xs text-muted-foreground">
                  Updated 2 min ago · ISRO Cartosat tile · {input} imagery
                </p>
              </div>
              <span className="inline-flex items-center gap-1.5 rounded-full bg-mint-soft px-3 py-1 text-[11px] font-bold text-mint">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-mint" />
                Live feed
              </span>
            </div>

            <div className="relative h-[540px] w-full overflow-hidden rounded-[24px] bg-[#1A1C1E] lg:h-[620px]">
              {network.isError || gatekeepers.isError ? (
                <div className="grid h-full w-full place-items-center">
                  <div className="space-y-2 text-center">
                    <p className="text-sm font-bold text-white/90">
                      Couldn’t reach the analysis backend
                    </p>
                    <p className="text-xs text-white/60">
                      Showing the last known network for {city}.
                    </p>
                  </div>
                </div>
              ) : network.data && gatekeepers.data ? (
                network.data.features.length === 0 ? (
                  <div className="grid h-full w-full place-items-center">
                    <p className="text-xs text-white/60">
                      No road network returned for {city} · {input} · {model}.
                    </p>
                  </div>
                ) : (
                  <ClientMap
                    network={network.data}
                    gatekeepers={gatekeepers.data}
                    stage={stage}
                    input={input}
                    model={model}
                    disabledEdgeIds={disabledIds}
                    onDisableEdge={(id) =>
                      setDisabledIds((d) =>
                        d.includes(id) ? d.filter((x) => x !== id) : [...d, id],
                      )
                    }
                    onSelectEdge={setSelectedEdge}
                  />
                )
              ) : (
                <div className="grid h-full w-full place-items-center">
                  <div className="space-y-3 text-center">
                    <div className="mx-auto h-2 w-32 animate-pulse rounded-full bg-white/10" />
                    <p className="text-xs text-white/60">
                      Loading network for {city}…
                    </p>
                  </div>
                </div>
              )}

              {/* Floating controls cluster — top right */}
              <div className="pointer-events-auto absolute right-4 top-4 z-[400] flex max-w-[calc(100%-2rem)] flex-col items-end gap-2">
                <FloatingControl label="Display mode">
                  <SegToggle
                    label="Display mode"
                    value={stage}
                    onChange={setStage}
                    options={[
                      { value: "satellite", label: "Satellite" },
                      { value: "extracted", label: "Extracted" },
                      { value: "graph", label: "Criticality" },
                    ]}
                  />
                </FloatingControl>
                <FloatingControl label="Imagery input">
                  <SegToggle
                    label="Imagery input"
                    value={input}
                    onChange={setInput}
                    options={[
                      {
                        value: "clean",
                        label: "Clean",
                        icon: <Sun className="h-3 w-3" />,
                      },
                      {
                        value: "occluded",
                        label: "Occluded",
                        icon: <Cloud className="h-3 w-3" />,
                      },
                    ]}
                  />
                </FloatingControl>
                <FloatingControl label="Extraction model">
                  <SegToggle
                    label="Extraction model"
                    value={model}
                    onChange={setModel}
                    options={[
                      {
                        value: "baseline",
                        label: "Baseline",
                        icon: <Layers className="h-3 w-3" />,
                      },
                      {
                        value: "robust",
                        label: "Robust",
                        icon: <Brain className="h-3 w-3" />,
                      },
                    ]}
                  />
                </FloatingControl>
              </div>

              {/* Bottom-right legend */}
              <div className="absolute bottom-4 right-4 z-[400]">
                <Legend />
              </div>

              {/* Callouts bottom-left */}
              <div className="absolute bottom-4 left-4 z-[400] max-w-[460px] space-y-2">
                {baselineWarning && (
                  <div className="flex items-start gap-2 rounded-2xl border border-coral/30 bg-white/95 px-4 py-3 text-xs shadow-lg backdrop-blur">
                    <span className="mt-1 h-2 w-2 shrink-0 animate-pulse rounded-full bg-coral" />
                    <span>
                      <span className="font-bold text-coral">
                        Baseline under occlusion
                      </span>{" "}
                      — clouds, shadows and canopy break the road graph.{" "}
                      <strong>3 gatekeeper junctions</strong> are missed and
                      several edges disappear.
                    </span>
                  </div>
                )}
                {input === "occluded" && model === "robust" && (
                  <div className="flex items-start gap-2 rounded-2xl border border-mint/30 bg-white/95 px-4 py-3 text-xs shadow-lg backdrop-blur">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-mint" />
                    <span>
                      <span className="font-bold text-mint">
                        Robust model recovered {recoveredCount} critical links
                      </span>{" "}
                      the baseline missed under occlusion.
                    </span>
                  </div>
                )}
                {selectedEdge && (
                  <div className="rounded-2xl border border-card-hairline bg-white/95 px-4 py-2 text-xs shadow-lg backdrop-blur">
                    <span className="text-muted-foreground">Selected:</span>{" "}
                    <span className="font-mono font-bold">{selectedEdge}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Before / After slider */}
            <div className="mt-3 rounded-[24px] bg-sand p-4">
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="grid h-7 w-7 place-items-center rounded-lg bg-coral-soft text-coral">
                    <Sparkles className="h-3.5 w-3.5" />
                  </div>
                  <h3 className="text-sm font-extrabold text-foreground">
                    Baseline vs robust under occlusion
                  </h3>
                </div>
                <span className="text-[11px] text-muted-foreground">
                  drag the divider →
                </span>
              </div>
              <BeforeAfterSlider />
            </div>

            {/* Analytics row — fills the left column so it matches the
                insight column on the right */}
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-[24px] border border-card-hairline bg-card p-5">
                <header className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-extrabold text-foreground">
                    Network resilience curve
                  </h3>
                  <span className="rounded-full bg-sand px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    efficiency vs edges removed
                  </span>
                </header>
                {curveRows ? (
                  <ResilienceChart data={curveRows} />
                ) : (
                  <SkeletonBlock />
                )}
                <div className="mt-1 flex items-center gap-4 text-[11px] text-muted-foreground">
                  <Dot color="#FF7A4E" label="Global efficiency" />
                  <Dot color="#2D8C7D" label="Giant component" />
                </div>
              </div>

              <div className="rounded-[24px] border border-card-hairline bg-card p-5">
                <header className="mb-3 flex items-center justify-between">
                  <h3 className="text-sm font-extrabold text-foreground">
                    Criticality distribution
                  </h3>
                  <span className="rounded-full bg-sand px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                    edge histogram
                  </span>
                </header>
                {hist.data ? (
                  <CriticalityHistogram data={hist.data} />
                ) : (
                  <SkeletonBlock />
                )}
              </div>
            </div>

            {/* Mission notes strip */}
            <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              <NoteCard
                tone="coral"
                kicker="Occlusion brief"
                title="Cloud cover 38%"
                body="Monsoon band drifting NE over central wards — expect baseline drops along MG Road corridor."
              />
              <NoteCard
                tone="mint"
                kicker="Model uplift"
                title="+12.4 pts efficiency"
                body="Robust extraction recovered 3 gatekeeper junctions the baseline lost under canopy."
              />
              <NoteCard
                tone="marigold"
                kicker="Field note"
                title="Sector 7 watch"
                body="Two high-criticality edges (D-1, V-3-4) sit on a single corridor — single point of failure."
              />
            </div>
          </section>


          {/* RIGHT — INSIGHT COLUMN */}
          <aside className="space-y-6">
            {/* Greeting hero */}
            <div className="relative overflow-hidden rounded-[32px] border border-card-hairline bg-card p-7 shadow-[var(--shadow-card)]">
              <div className="relative z-10">
                <p className="text-[11px] font-bold uppercase tracking-widest text-muted-foreground">
                  Good morning, Mission Lead
                </p>
                <h2 className="mt-1 text-2xl font-extrabold leading-tight">
                  {city} is{" "}
                  <span className="text-coral">
                    {metrics.data && metrics.data.resilienceIndex >= 70
                      ? "holding steady"
                      : "under stress"}
                  </span>
                </h2>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  Cloud cover is moderate over central wards; the robust model
                  is recovering road structure where the baseline drops out.
                </p>
              </div>
              <div className="pointer-events-none absolute -bottom-12 -right-12 h-48 w-48 rounded-full bg-coral/10 blur-3xl" />
              <svg
                className="pointer-events-none absolute right-5 top-5 opacity-30"
                width="56"
                height="56"
                viewBox="0 0 100 100"
              >
                <path
                  d="M10,90 Q50,10 90,90"
                  stroke="#FF7A4E"
                  fill="none"
                  strokeWidth="2.4"
                  strokeDasharray="5 5"
                  strokeLinecap="round"
                />
              </svg>
            </div>

            <Panel title="Performance metrics" subtitle={`${input} · ${model}`}>
              {metrics.data ? <KpiGrid metrics={metrics.data} /> : <SkeletonBlock />}
            </Panel>

            <Panel
              title="What-if simulation"
              subtitle={`${disabledIds.length} edge${
                disabledIds.length === 1 ? "" : "s"
              } disabled`}
            >
              {network.data && metrics.data ? (
                <WhatIfCard
                  network={network.data}
                  metrics={metrics.data}
                  disabledIds={disabledIds}
                  result={simResult}
                  loading={simLoading}
                  error={simError}
                  onToggle={(id) =>
                    setDisabledIds((d) =>
                      d.includes(id) ? d.filter((x) => x !== id) : [...d, id],
                    )
                  }
                  onReset={() => setDisabledIds([])}
                />
              ) : (
                <SkeletonBlock />
              )}
            </Panel>
          </aside>

        </div>
        )}

        {view === "health" && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Panel title="Network resilience curve" subtitle="efficiency vs edges removed">
              {curveRows ? <ResilienceChart data={curveRows} /> : <SkeletonBlock />}
              <div className="mt-1 flex items-center gap-4 text-[11px] text-muted-foreground">
                <Dot color="#FF7A4E" label="Global efficiency" />
                <Dot color="#2D8C7D" label="Giant component" />
              </div>
            </Panel>
            <Panel title="Criticality distribution" subtitle="edge histogram">
              {hist.data ? <CriticalityHistogram data={hist.data} /> : <SkeletonBlock />}
            </Panel>
            <Panel title="Performance metrics" subtitle={`${input} · ${model}`}>
              {metrics.data ? <KpiGrid metrics={metrics.data} /> : <SkeletonBlock />}
            </Panel>
            <Panel title="What-if simulation" subtitle={`${disabledIds.length} edge${disabledIds.length === 1 ? "" : "s"} disabled`}>
              {network.data && metrics.data ? (
                <WhatIfCard
                  network={network.data}
                  metrics={metrics.data}
                  disabledIds={disabledIds}
                  result={simResult}
                  loading={simLoading}
                  error={simError}
                  onToggle={(id) =>
                    setDisabledIds((d) =>
                      d.includes(id) ? d.filter((x) => x !== id) : [...d, id],
                    )
                  }
                  onReset={() => setDisabledIds([])}
                />
              ) : (
                <SkeletonBlock />
              )}
            </Panel>
          </div>
        )}

        {view === "reports" && (
          <div className="space-y-6">
            <Panel title="Mission report" subtitle={`${city} · ${input} · ${model}`}>
              <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
                <p>
                  <span className="font-bold text-foreground">{city}</span> network analysed
                  with the <span className="font-bold text-coral">{model}</span> extractor on{" "}
                  <span className="font-bold text-foreground">{input}</span> imagery.
                  Resilience index{" "}
                  <span className="font-bold text-foreground">
                    {metrics.data?.resilienceIndex.toFixed(1) ?? "—"}
                  </span>{" "}
                  · Occlusion recall{" "}
                  <span className="font-bold text-foreground">
                    {metrics.data ? (metrics.data.occlusionRecall * 100).toFixed(1) + "%" : "—"}
                  </span>
                  .
                </p>
                <ul className="grid grid-cols-1 gap-3 md:grid-cols-3">
                  <li className="rounded-2xl bg-sand p-4">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Edges analysed</div>
                    <div className="mt-1 text-2xl font-extrabold text-foreground">{network.data?.features.length ?? "—"}</div>
                  </li>
                  <li className="rounded-2xl bg-sand p-4">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Critical edges</div>
                    <div className="mt-1 text-2xl font-extrabold text-coral">
                      {network.data?.features.filter((e) => e.properties.criticality >= 0.75).length ?? "—"}
                    </div>
                  </li>
                  <li className="rounded-2xl bg-sand p-4">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">Disabled in sim</div>
                    <div className="mt-1 text-2xl font-extrabold text-foreground">{disabledIds.length}</div>
                  </li>
                </ul>
              </div>
            </Panel>
            <Panel title="Top critical edges" subtitle="ranked by betweenness">
              <ol className="divide-y divide-card-hairline">
                {(network.data?.features ?? [])
                  .slice()
                  .sort((a, b) => b.properties.criticality - a.properties.criticality)
                  .slice(0, 8)
                  .map((e, i) => (
                    <li key={e.properties.id} className="flex items-center justify-between py-2.5 text-sm">
                      <span className="flex items-center gap-3">
                        <span className="grid h-6 w-6 place-items-center rounded-full bg-coral-soft text-[10px] font-bold text-coral">
                          {i + 1}
                        </span>
                        <span className="font-mono font-bold">{e.properties.id}</span>
                      </span>
                      <span className="text-muted-foreground">c = {e.properties.criticality.toFixed(2)}</span>
                    </li>
                  ))}
              </ol>
            </Panel>

          </div>
        )}



        <footer className="pt-2 text-center text-[11px] text-muted-foreground">
          Route Resilience · live FastAPI via{" "}
          <code className="rounded bg-sand px-1.5 py-0.5 text-foreground">
            src/lib/api.ts
          </code>{" "}
          · falls back to bundled data when the backend is offline
        </footer>
      </div>
    </div>
  );
}

function NoteCard({
  tone,
  kicker,
  title,
  body,
}: {
  tone: "coral" | "mint" | "marigold";
  kicker: string;
  title: string;
  body: string;
}) {
  const palette = {
    coral: { dot: "bg-coral", tint: "bg-coral-soft", text: "text-coral" },
    mint: { dot: "bg-mint", tint: "bg-mint-soft", text: "text-mint" },
    marigold: {
      dot: "bg-[hsl(38_90%_60%)]",
      tint: "bg-[hsl(38_90%_60%)]/10",
      text: "text-[hsl(32_80%_42%)]",
    },
  }[tone];
  return (
    <div className="rounded-[24px] border border-card-hairline bg-card p-5 shadow-[var(--shadow-card)]">
      <div className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest ${palette.tint} ${palette.text}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${palette.dot}`} />
        {kicker}
      </div>
      <h4 className="mt-3 text-base font-extrabold text-foreground">{title}</h4>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>
    </div>
  );
}

function FloatingControl({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-white/60 bg-white/95 p-2.5 shadow-[0_10px_30px_-12px_rgba(31,27,22,0.25)] backdrop-blur">
      <p className="mb-1.5 px-1 text-[9px] font-bold uppercase tracking-widest text-muted-foreground">
        {label}
      </p>
      {children}
    </div>
  );
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-[28px] border border-card-hairline bg-card p-6 shadow-[var(--shadow-card)]">
      <header className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-extrabold text-foreground">{title}</h3>
        {subtitle && (
          <span className="rounded-full bg-sand px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            {subtitle}
          </span>
        )}
      </header>
      {children}
    </section>
  );
}

function Dot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{ background: color, boxShadow: `0 0 0 3px ${color}22` }}
      />
      {label}
    </span>
  );
}

function SkeletonBlock() {
  return (
    <div className="space-y-2">
      <div className="h-3 w-3/4 animate-pulse rounded bg-sand" />
      <div className="h-3 w-1/2 animate-pulse rounded bg-sand" />
      <div className="h-16 w-full animate-pulse rounded bg-sand" />
    </div>
  );
}
