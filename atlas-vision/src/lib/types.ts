export interface RoadFeatureProperties {
  id: string;
  criticality: number; // 0..1
  travelTimeSec: number;
  lengthM: number;
  roadClass: string;
  isBridge: boolean;
}

export interface RoadFeature {
  type: "Feature";
  geometry: { type: "LineString"; coordinates: [number, number][] };
  properties: RoadFeatureProperties;
}

export interface RoadNetworkMeta {
  city: string;
  input: InputMode;
  model: ModelMode;
  edges: number;
  nodes: number;
}

export interface RoadNetwork {
  type: "FeatureCollection";
  features: RoadFeature[];
  meta?: RoadNetworkMeta;
}

export interface GatekeeperNode {
  id: string;
  lat: number;
  lng: number;
  betweenness: number;
  isArticulation: boolean;
  label: string;
}

export interface Metrics {
  iou: number;
  dice: number;
  occlusionRecall: number;
  resilienceIndex: number; // 0..100
}

export interface SimulationResult {
  disabledEdgeIds: string[];
  disabledNodeIds: string[];
  resilienceIndexAfter: number;
  avgTravelTimeIncreasePct: number;
  newlyDisconnectedZones: number;
  brokenRoutesSampled: number;
  sampledRoutes: number;
}

export type Stage = "satellite" | "extracted" | "graph";
export type InputMode = "clean" | "occluded";
export type ModelMode = "baseline" | "robust";

export interface DisplayState {
  city: string;
  stage: Stage;
  input: InputMode;
  model: ModelMode;
}
