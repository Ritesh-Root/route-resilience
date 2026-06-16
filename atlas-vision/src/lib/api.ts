/**
 * Route Resilience API facade.
 *
 * Each function calls the real FastAPI backend (see BACKEND CONTRACT) using
 * `API_BASE` from "./config". All routes live under `${API_BASE}/api`.
 *
 * Resilience: if a request fails for any reason (backend down, network error,
 * non-2xx), we fall back to the bundled MOCK data so the demo never hard-crashes.
 */

import {
  NETWORKS,
  METRICS,
  GATEKEEPERS,
  BASELINE_OCCLUDED_DROPPED_GATEKEEPERS,
  RESILIENCE_CURVE,
  CRITICALITY_HIST,
} from "./mock";
import type {
  InputMode,
  ModelMode,
  Metrics,
  RoadNetwork,
  GatekeeperNode,
  SimulationResult,
  DisplayState,
} from "./types";
import { API_BASE } from "./config";

const API = `${API_BASE}/api`;

function key(input: InputMode, model: ModelMode) {
  return `${input}-${model}` as keyof typeof NETWORKS;
}

/**
 * Thin fetch wrapper: builds a URL under `${API_BASE}/api`, parses JSON, and
 * throws on non-2xx so callers can fall back to mock data in a single catch.
 */
async function getJSON<T>(
  path: string,
  params?: Record<string, string | number | undefined>,
): Promise<T> {
  const url = new URL(`${API}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return (await res.json()) as T;
}

export async function fetchNetwork(
  city: string,
  input: InputMode,
  model: ModelMode,
): Promise<RoadNetwork> {
  try {
    // Backend returns a GeoJSON FeatureCollection (plus a top-level `.meta`).
    // RoadNetwork is structurally compatible; extra fields are harmless.
    return await getJSON<RoadNetwork>("/network", { city, input, model });
  } catch (e) {
    console.warn("[api] fetchNetwork falling back to mock:", e);
    return NETWORKS[key(input, model)];
  }
}

export async function fetchMetrics(
  city: string,
  input: InputMode,
  model: ModelMode,
): Promise<Metrics> {
  try {
    return await getJSON<Metrics>("/metrics", { city, input, model });
  } catch (e) {
    console.warn("[api] fetchMetrics falling back to mock:", e);
    return METRICS[key(input, model)];
  }
}

export async function fetchGatekeepers(
  city: string,
  input: InputMode,
  model: ModelMode,
  topK?: number,
): Promise<GatekeeperNode[]> {
  try {
    // Backend nodes carry `isArticulation` in addition to GatekeeperNode's
    // fields; it is preserved at runtime even though the TS type omits it.
    return await getJSON<GatekeeperNode[]>("/gatekeepers", {
      city,
      input,
      model,
      top_k: topK,
    });
  } catch (e) {
    console.warn("[api] fetchGatekeepers falling back to mock:", e);
    if (input === "occluded" && model === "baseline") {
      return GATEKEEPERS.map((g) =>
        BASELINE_OCCLUDED_DROPPED_GATEKEEPERS.includes(g.id)
          ? { ...g, betweenness: 0 } // marker for "dropped"
          : g,
      );
    }
    return GATEKEEPERS;
  }
}

/**
 * GET /api/resilience-curve -> {removedFraction[], efficiency[], giantComponent[]}.
 * Returned as-is (parallel arrays). The mock fallback uses the legacy
 * row-shaped array; callers that need the demo curve should consume whichever
 * shape they receive (see followups).
 */
export async function fetchResilienceCurve(
  city = "Bengaluru",
  input: InputMode = "occluded",
  model: ModelMode = "baseline",
) {
  try {
    return await getJSON<{
      removedFraction: number[];
      efficiency: number[];
      giantComponent: number[];
    }>("/resilience-curve", { city, input, model });
  } catch (e) {
    console.warn("[api] fetchResilienceCurve falling back to mock:", e);
    return RESILIENCE_CURVE;
  }
}

/**
 * The backend exposes no histogram endpoint, so we compute it client-side from
 * the network's per-edge `criticality` (0..1) values, bucketed into fifths to
 * match the mock shape `{ bucket, count }[]`.
 *
 * Kept callable with zero args (current call site) by fetching the network for
 * the given/default display state.
 */
export async function fetchCriticalityHistogram(
  city = "Bengaluru",
  input: InputMode = "occluded",
  model: ModelMode = "baseline",
) {
  try {
    const net = await fetchNetwork(city, input, model);
    const buckets = [
      { bucket: "0.0–0.2", count: 0 },
      { bucket: "0.2–0.4", count: 0 },
      { bucket: "0.4–0.6", count: 0 },
      { bucket: "0.6–0.8", count: 0 },
      { bucket: "0.8–1.0", count: 0 },
    ];
    for (const f of net.features) {
      const c = Math.max(0, Math.min(0.999999, f.properties.criticality));
      buckets[Math.floor(c * 5)].count++;
    }
    return buckets;
  } catch (e) {
    console.warn("[api] fetchCriticalityHistogram falling back to mock:", e);
    return CRITICALITY_HIST;
  }
}

/**
 * POST /api/simulate -> SimulationResult.
 *
 * Signature changed from the mock `(network, baseResilience, disabledEdgeIds)`
 * to `(display, disabledEdgeIds, disabledNodeIds?)` because the backend keys the
 * simulation off {city, model, input} rather than a client network snapshot.
 * See followups — the call site in routes/index.tsx must be updated.
 *
 * On error we synthesize a result from the network's criticality (old mock
 * heuristic) so the panel keeps working offline.
 */
export async function simulateDisable(
  display: DisplayState,
  disabledEdgeIds: string[],
  disabledNodeIds: string[] = [],
): Promise<SimulationResult> {
  try {
    const res = await fetch(`${API}/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city: display.city,
        model: display.model,
        input: display.input,
        disabledEdgeIds,
        disabledNodeIds,
      }),
    });
    if (!res.ok) throw new Error(`POST /simulate -> ${res.status}`);
    return (await res.json()) as SimulationResult;
  } catch (e) {
    console.warn("[api] simulateDisable falling back to mock:", e);
    const network = NETWORKS[key(display.input, display.model)];
    const baseResilience = METRICS[key(display.input, display.model)].resilienceIndex;
    const crits = disabledEdgeIds.map((id) => {
      const f = network.features.find((f) => f.properties.id === id);
      return f?.properties.criticality ?? 0;
    });
    const totalCrit = crits.reduce((a, b) => a + b, 0);
    const resilienceIndexAfter = Math.max(
      8,
      Math.round(baseResilience - totalCrit * 22),
    );
    const avgTravelTimeIncreasePct = +(totalCrit * 14).toFixed(1);
    const newlyDisconnectedZones = Math.min(5, Math.floor(totalCrit * 1.8));
    return {
      disabledEdgeIds,
      disabledNodeIds,
      resilienceIndexAfter,
      avgTravelTimeIncreasePct,
      newlyDisconnectedZones,
      brokenRoutesSampled: Math.round(totalCrit * 6),
      sampledRoutes: 50,
    };
  }
}
