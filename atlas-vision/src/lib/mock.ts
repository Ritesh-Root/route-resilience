import type {
  RoadNetwork,
  GatekeeperNode,
  Metrics,
  RoadFeature,
} from "./types";

// Bengaluru center
export const BENGALURU_CENTER: [number, number] = [12.9716, 77.5946];

// Deterministic pseudo-random
function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Pick a road class from criticality so high-criticality edges read as
// arterials/highways and low-criticality edges as residential streets.
// Matches the backend `roadClass` property on edge features.
type RoadClass = "highway" | "primary" | "secondary" | "residential";
function roadClassFor(criticality: number): RoadClass {
  if (criticality >= 0.8) return "highway";
  if (criticality >= 0.6) return "primary";
  if (criticality >= 0.35) return "secondary";
  return "residential";
}

function makeGrid(seed: number, drop: number[] = []): RoadNetwork {
  const rand = rng(seed);
  const features: RoadFeature[] = [];
  const lat0 = 12.9716;
  const lng0 = 77.5946;
  const step = 0.012;
  const rows = 7;
  const cols = 7;

  let idx = 0;
  // horizontal edges
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols - 1; c++) {
      const id = `H-${r}-${c}`;
      idx++;
      if (drop.includes(idx)) continue;
      const lat = lat0 + (r - rows / 2) * step + (rand() - 0.5) * 0.0008;
      const lng1 = lng0 + (c - cols / 2) * step;
      const lng2 = lng0 + (c + 1 - cols / 2) * step;
      // Add slight curve
      const mid: [number, number] = [
        lng1 + (lng2 - lng1) * 0.5,
        lat + (rand() - 0.5) * 0.0012,
      ];
      const coords: [number, number][] = [
        [lng1, lat],
        mid,
        [lng2, lat + (rand() - 0.5) * 0.0006],
      ];
      // Higher criticality toward center
      const distFromCenter =
        Math.abs(r - rows / 2) + Math.abs(c + 0.5 - cols / 2);
      const baseCrit = Math.max(0, 1 - distFromCenter / 6) + (rand() - 0.5) * 0.2;
      const crit = Math.max(0.05, Math.min(0.98, baseCrit));
      features.push({
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords },
        properties: {
          id,
          criticality: crit,
          travelTimeSec: 60 + Math.round(rand() * 120),
          lengthM: Math.round(900 + rand() * 600),
          roadClass: roadClassFor(crit),
          isBridge: rand() > 0.92,
        },
      });
    }
  }
  // vertical edges
  for (let r = 0; r < rows - 1; r++) {
    for (let c = 0; c < cols; c++) {
      const id = `V-${r}-${c}`;
      idx++;
      if (drop.includes(idx)) continue;
      const lng = lng0 + (c - cols / 2) * step + (rand() - 0.5) * 0.0008;
      const lat1 = lat0 + (r - rows / 2) * step;
      const lat2 = lat0 + (r + 1 - rows / 2) * step;
      const mid: [number, number] = [
        lng + (rand() - 0.5) * 0.0012,
        lat1 + (lat2 - lat1) * 0.5,
      ];
      const coords: [number, number][] = [
        [lng, lat1],
        mid,
        [lng + (rand() - 0.5) * 0.0006, lat2],
      ];
      const distFromCenter =
        Math.abs(r + 0.5 - rows / 2) + Math.abs(c - cols / 2);
      const baseCrit = Math.max(0, 1 - distFromCenter / 6) + (rand() - 0.5) * 0.2;
      const crit = Math.max(0.05, Math.min(0.98, baseCrit));
      features.push({
        type: "Feature",
        geometry: { type: "LineString", coordinates: coords },
        properties: {
          id,
          criticality: crit,
          travelTimeSec: 70 + Math.round(rand() * 140),
          lengthM: Math.round(900 + rand() * 600),
          roadClass: roadClassFor(crit),
          isBridge: rand() > 0.92,
        },
      });
    }
  }

  // a few diagonals for character
  for (let i = 0; i < 4; i++) {
    const id = `D-${i}`;
    idx++;
    if (drop.includes(idx)) continue;
    const r = 1 + Math.floor(rand() * (rows - 2));
    const c = 1 + Math.floor(rand() * (cols - 2));
    const a: [number, number] = [
      lng0 + (c - cols / 2) * step,
      lat0 + (r - rows / 2) * step,
    ];
    const b: [number, number] = [
      lng0 + (c + 1 - cols / 2) * step,
      lat0 + (r + 1 - rows / 2) * step,
    ];
    features.push({
      type: "Feature",
      geometry: { type: "LineString", coordinates: [a, b] },
      properties: {
        id,
        criticality: 0.7 + rand() * 0.25,
        travelTimeSec: 110 + Math.round(rand() * 120),
        lengthM: 1400 + Math.round(rand() * 500),
        roadClass: "highway",
        isBridge: rand() > 0.5,
      },
    });
  }

  return { type: "FeatureCollection", features };
}

// Two variants for the same area
export const NETWORKS: Record<
  "clean-baseline" | "clean-robust" | "occluded-baseline" | "occluded-robust",
  RoadNetwork
> = {
  "clean-baseline": makeGrid(11),
  "clean-robust": makeGrid(11),
  // baseline under occlusion drops a chunk of edges (broken extraction)
  "occluded-baseline": makeGrid(11, [3, 8, 14, 17, 22, 27, 31, 36, 41, 48, 55, 60, 66, 72, 79, 85]),
  // robust under occlusion recovers nearly all
  "occluded-robust": makeGrid(11, [55]),
};

export const GATEKEEPERS: GatekeeperNode[] = [
  { id: "GK-01", lat: 12.9716, lng: 77.5946, betweenness: 0.94, isArticulation: true, label: "Majestic Junction" },
  { id: "GK-02", lat: 12.9836, lng: 77.6066, betweenness: 0.88, isArticulation: true, label: "Cubbon Park Gate" },
  { id: "GK-03", lat: 12.9596, lng: 77.5826, betweenness: 0.84, isArticulation: false, label: "K. R. Market" },
  { id: "GK-04", lat: 12.9476, lng: 77.6066, betweenness: 0.81, isArticulation: true, label: "Lalbagh North" },
  { id: "GK-05", lat: 12.9836, lng: 77.5826, betweenness: 0.79, isArticulation: true, label: "Vidhana Soudha" },
  { id: "GK-06", lat: 12.9596, lng: 77.6186, betweenness: 0.74, isArticulation: false, label: "Richmond Circle" },
];

// Gatekeepers that the baseline model drops under occlusion
export const BASELINE_OCCLUDED_DROPPED_GATEKEEPERS = ["GK-02", "GK-04", "GK-05"];

export const METRICS: Record<
  "clean-baseline" | "clean-robust" | "occluded-baseline" | "occluded-robust",
  Metrics
> = {
  // Real DeepGlobe val metrics from the trained D-LinkNet models (Kaggle run).
  // resilienceIndex mirrors the live backend story (intact 100; occluded
  // baseline fragments to ~79).
  "clean-baseline": {
    iou: 0.528, dice: 0.691, occlusionRecall: 0.671, resilienceIndex: 100,
  },
  "clean-robust": {
    iou: 0.526, dice: 0.689, occlusionRecall: 0.665, resilienceIndex: 100,
  },
  "occluded-baseline": {
    iou: 0.439, dice: 0.610, occlusionRecall: 0.528, resilienceIndex: 79,
  },
  "occluded-robust": {
    iou: 0.511, dice: 0.676, occlusionRecall: 0.648, resilienceIndex: 100,
  },
};

// Resilience curve: efficiency & giant component vs edges removed
export const RESILIENCE_CURVE = Array.from({ length: 21 }, (_, i) => {
  const removed = i * 2;
  const efficiency = Math.max(0.05, 1 - Math.pow(removed / 50, 1.4));
  const giant = Math.max(0.08, 1 - Math.pow(removed / 45, 1.7));
  return {
    removed,
    efficiency: +(efficiency * 100).toFixed(1),
    giant: +(giant * 100).toFixed(1),
  };
});

// Histogram of criticality buckets
export const CRITICALITY_HIST = [
  { bucket: "0.0–0.2", count: 18 },
  { bucket: "0.2–0.4", count: 24 },
  { bucket: "0.4–0.6", count: 19 },
  { bucket: "0.6–0.8", count: 12 },
  { bucket: "0.8–1.0", count: 7 },
];

// Sample origin->destination routes used by the simulation result.
// `sampledRoutes` mirrors the backend SimulationResult.sampledRoutes shape;
// when a route's path crosses a disabled edge it is reported in
// `brokenRoutesSampled`. These give the demo concrete "broken commute"
// examples without needing a real shortest-path solve.
export interface SampledRoute {
  id: string;
  fromLabel: string;
  toLabel: string;
  edgeIds: string[]; // edges along the route path
}

export const SAMPLED_ROUTES: SampledRoute[] = [
  { id: "RT-01", fromLabel: "Majestic Junction", toLabel: "Lalbagh North", edgeIds: ["V-2-3", "V-3-3", "H-4-3"] },
  { id: "RT-02", fromLabel: "Cubbon Park Gate", toLabel: "K. R. Market", edgeIds: ["H-2-4", "V-2-2", "H-3-1"] },
  { id: "RT-03", fromLabel: "Vidhana Soudha", toLabel: "Richmond Circle", edgeIds: ["H-2-1", "V-2-4", "H-3-5"] },
  { id: "RT-04", fromLabel: "Majestic Junction", toLabel: "Cubbon Park Gate", edgeIds: ["H-3-3", "V-3-4", "H-2-4"] },
];

export const CITIES = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Chennai"];
