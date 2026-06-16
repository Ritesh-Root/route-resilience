import { Satellite, Workflow, Network, ChevronRight, CheckCircle2 } from "lucide-react";
import type { Stage } from "@/lib/types";

const STAGES: {
  id: Stage;
  title: string;
  subtitle: string;
  Icon: typeof Satellite;
  tint: string;
  iconTint: string;
}[] = [
  {
    id: "satellite",
    title: "Satellite",
    subtitle: "Cartosat / Sentinel-2",
    Icon: Satellite,
    tint: "bg-mint-soft",
    iconTint: "text-mint",
  },
  {
    id: "extracted",
    title: "Extraction",
    subtitle: "Occlusion-robust segmentation",
    Icon: Workflow,
    tint: "bg-amber-soft",
    iconTint: "text-amber",
  },
  {
    id: "graph",
    title: "Criticality graph",
    subtitle: "Betweenness · APLS · resilience",
    Icon: Network,
    tint: "bg-coral-soft",
    iconTint: "text-coral",
  },
];

export function PipelineStrip({
  active,
  onChange,
}: {
  active: Stage;
  onChange: (s: Stage) => void;
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {STAGES.map((s, i) => {
        const isActive = active === s.id;
        return (
          <button
            key={s.id}
            onClick={() => onChange(s.id)}
            aria-pressed={isActive}
            className={[
              "group relative flex items-center gap-4 rounded-2xl bg-card p-4 text-left transition-all",
              "border shadow-[var(--shadow-soft)] hover:-translate-y-0.5 hover:shadow-[var(--shadow-card)]",
              isActive ? "border-coral/40 ring-2 ring-coral/15" : "border-card-hairline",
            ].join(" ")}
          >
            <div
              className={`grid h-11 w-11 shrink-0 place-items-center rounded-full ${s.tint} ${s.iconTint}`}
            >
              <s.Icon className="h-5 w-5" strokeWidth={2.2} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                  Stage {i + 1}
                </p>
                {isActive && (
                  <span className="rounded-full bg-coral/10 px-2 py-0.5 text-[10px] font-bold text-coral">
                    Active
                  </span>
                )}
              </div>
              <p className="truncate text-sm font-bold text-foreground">{s.title}</p>
              <p className="truncate text-[11px] text-muted-foreground">{s.subtitle}</p>
            </div>
            {i < STAGES.length - 1 && (
              <ChevronRight className="hidden h-4 w-4 shrink-0 text-card-hairline lg:block" />
            )}
          </button>
        );
      })}
      {/* 4th slot — system health pill */}
      <div className="hidden items-center gap-4 rounded-2xl border border-card-hairline bg-card p-4 shadow-[var(--shadow-soft)] sm:flex">
        <div className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-mint-soft text-mint">
          <CheckCircle2 className="h-5 w-5" strokeWidth={2.2} />
        </div>
        <div className="min-w-0">
          <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
            System
          </p>
          <p className="text-sm font-bold text-foreground">Sync healthy</p>
        </div>
      </div>
    </div>
  );
}
