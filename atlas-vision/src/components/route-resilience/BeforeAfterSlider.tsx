import { useRef, useState, useEffect } from "react";
import { ArrowLeftRight } from "lucide-react";
import { fetchMetrics, fetchNetwork } from "@/lib/api";
import type { InputMode, ModelMode } from "@/lib/types";

/**
 * Before/After slider comparing two road-network reconstructions side by side.
 *
 * The decorative SVG "masks" (fragmented baseline vs. recovered robust) are kept
 * as the visual, but the labels + caption strip are now driven by REAL backend
 * data fetched via the corrected lib layer:
 *   - left  side: {leftInput,  leftModel}  (default occluded + baseline)
 *   - right side: {rightInput, rightModel} (default occluded + robust)
 *
 * For each side we pull /api/metrics (resilienceIndex) and /api/network (edge
 * count) so the captions reflect the live demo story (baseline fragments under
 * occlusion -> ~79; robust recovers -> 100). On failure the lib layer already
 * falls back to bundled mock data, so this component shows an explicit error
 * banner only if BOTH sides reject (e.g. fetch threw before any fallback).
 */

interface SideData {
  resilienceIndex: number;
  edges: number;
}

interface SideConfig {
  input: InputMode;
  model: ModelMode;
}

export function BeforeAfterSlider({
  city = "Bengaluru",
  leftLabel = "Baseline",
  rightLabel = "Robust",
  left = { input: "occluded", model: "baseline" },
  right = { input: "occluded", model: "robust" },
}: {
  city?: string;
  leftLabel?: string;
  rightLabel?: string;
  left?: SideConfig;
  right?: SideConfig;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState(50);
  const [dragging, setDragging] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [leftData, setLeftData] = useState<SideData | null>(null);
  const [rightData, setRightData] = useState<SideData | null>(null);

  // Fetch real metrics + network edge counts for both sides whenever the
  // compared (city, input, model) selection changes.
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const loadSide = async (side: SideConfig): Promise<SideData> => {
      const [metrics, network] = await Promise.all([
        fetchMetrics(city, side.input, side.model),
        fetchNetwork(city, side.input, side.model),
      ]);
      return {
        resilienceIndex: metrics.resilienceIndex,
        edges: network.meta?.edges ?? network.features.length,
      };
    };

    Promise.all([loadSide(left), loadSide(right)])
      .then(([l, r]) => {
        if (cancelled) return;
        setLeftData(l);
        setRightData(r);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load comparison");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [city, left.input, left.model, right.input, right.model]);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      const rect = ref.current?.getBoundingClientRect();
      if (!rect) return;
      const clientX =
        "touches" in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
      const p = ((clientX - rect.left) / rect.width) * 100;
      setPos(Math.max(0, Math.min(100, p)));
    };
    const onUp = () => setDragging(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onMove);
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onMove);
      window.removeEventListener("touchend", onUp);
    };
  }, [dragging]);

  // Loading state — keep the rounded card frame + styling, show a pulse.
  if (loading) {
    return (
      <div className="relative grid h-48 w-full place-items-center overflow-hidden rounded-[20px] border border-card-hairline bg-sand">
        <div className="h-24 w-[88%] animate-pulse rounded-2xl bg-card-hairline/40" />
        <span className="absolute bottom-3 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
          loading comparison…
        </span>
      </div>
    );
  }

  // Error state — only when both sides failed before any mock fallback.
  if (error || !leftData || !rightData) {
    return (
      <div className="relative grid h-48 w-full place-items-center overflow-hidden rounded-[20px] border border-card-hairline bg-sand px-6 text-center">
        <div>
          <p className="text-xs font-bold uppercase tracking-widest text-coral">
            comparison unavailable
          </p>
          <p className="mt-1 text-[11px] text-muted-foreground">
            {error ?? "No network data returned for this selection."}
          </p>
        </div>
      </div>
    );
  }

  // Derived demo-story copy from real numbers.
  const leftBroken = leftData.resilienceIndex < rightData.resilienceIndex;
  const leftCaption = leftBroken
    ? `broken under cloud / canopy · RI ${leftData.resilienceIndex}`
    : `RI ${leftData.resilienceIndex} · ${leftData.edges} edges`;
  const rightCaption =
    rightData.resilienceIndex >= 100
      ? `full network recovered · RI ${rightData.resilienceIndex}`
      : `RI ${rightData.resilienceIndex} · ${rightData.edges} edges`;

  return (
    <div
      ref={ref}
      className="relative h-48 w-full select-none overflow-hidden rounded-[20px] border border-card-hairline bg-sand"
    >
      <BaselineMask />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${pos}%)` }}>
        <RobustMask />
      </div>

      {/* caption strip pinned to bottom, above the road art — now real data */}
      <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-sand via-sand/95 to-transparent px-4 pb-2 pt-6 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
        <span>{leftCaption}</span>
        <span className="text-[hsl(165_45%_36%)]">{rightCaption}</span>
      </div>

      <div className="pointer-events-none absolute left-3 top-3 rounded-full bg-white/95 px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest text-muted-foreground shadow-sm">
        {leftLabel}
      </div>
      <div className="pointer-events-none absolute right-3 top-3 rounded-full bg-coral px-2.5 py-1 text-[10px] font-bold uppercase tracking-widest text-white shadow-sm">
        {rightLabel}
      </div>

      <div
        className="pointer-events-none absolute top-0 bottom-0 w-px bg-white shadow-[0_0_8px_rgba(255,122,78,0.45)]"
        style={{ left: `${pos}%` }}
      />
      <button
        aria-label="Drag to compare"
        onMouseDown={() => setDragging(true)}
        onTouchStart={() => setDragging(true)}
        className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 grid h-9 w-9 cursor-ew-resize place-items-center rounded-full border-2 border-coral bg-white shadow-lg"
        style={{ left: `${pos}%` }}
      >
        <ArrowLeftRight className="h-3.5 w-3.5 text-coral" />
      </button>
    </div>
  );
}

function BaselineMask() {
  return (
    <svg viewBox="0 0 400 180" className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
      <defs>
        <radialGradient id="cloudB" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#E8DFCC" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#E8DFCC" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="400" height="180" fill="#FAF7F2" />
      <g>
        <ellipse cx="60" cy="40" rx="55" ry="26" fill="url(#cloudB)" />
        <ellipse cx="170" cy="70" rx="70" ry="30" fill="url(#cloudB)" />
        <ellipse cx="300" cy="50" rx="60" ry="28" fill="url(#cloudB)" />
        <ellipse cx="120" cy="115" rx="65" ry="26" fill="url(#cloudB)" />
        <ellipse cx="290" cy="120" rx="75" ry="28" fill="url(#cloudB)" />
      </g>
      <g stroke="#FF7A4E" strokeWidth="2.4" strokeLinecap="round" fill="none">
        <path d="M 10 38 L 80 38 M 130 38 L 230 38 M 290 38 L 390 38" />
        <path d="M 10 82 L 60 82 M 140 82 L 210 82 M 260 82 L 320 82" />
        <path d="M 10 126 L 95 126 M 165 126 L 250 126 M 315 126 L 390 126" />
        <path d="M 70 8 L 70 58 M 70 100 L 70 146" />
        <path d="M 200 8 L 200 68 M 200 115 L 200 146" />
        <path d="M 330 8 L 330 72 M 330 110 L 330 146" />
      </g>
    </svg>
  );
}

function RobustMask() {
  return (
    <svg viewBox="0 0 400 180" className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
      <rect width="400" height="180" fill="#FAF7F2" />
      <g opacity="0.4">
        <ellipse cx="170" cy="70" rx="70" ry="30" fill="#E8DFCC" />
        <ellipse cx="290" cy="120" rx="75" ry="28" fill="#E8DFCC" />
      </g>
      <g stroke="#2D8C7D" strokeWidth="2.6" strokeLinecap="round" fill="none">
        <path d="M 10 38 L 390 38" />
        <path d="M 10 82 L 390 82" />
        <path d="M 10 126 L 390 126" />
        <path d="M 70 8 L 70 146" />
        <path d="M 200 8 L 200 146" />
        <path d="M 330 8 L 330 146" />
      </g>
      <path d="M 30 20 L 380 140" stroke="#FFC857" strokeWidth="1.8" fill="none" opacity="0.75" strokeLinecap="round" />
    </svg>
  );
}
