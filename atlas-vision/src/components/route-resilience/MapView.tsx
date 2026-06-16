import { useEffect, useMemo, useRef } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import type { GeoJSON as LeafletGeoJSON } from "leaflet";
import { criticalityColor } from "@/lib/criticality";
import { BENGALURU_CENTER } from "@/lib/mock";
import { AlertTriangle, Power } from "lucide-react";
import type {
  RoadNetwork,
  RoadFeatureProperties,
  GatekeeperNode,
  Stage,
  InputMode,
  ModelMode,
} from "@/lib/types";

const TILE_URLS: Record<Stage, string> = {
  satellite:
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  extracted:
    "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
  graph:
    "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
};

function FlyOnChange({ deps }: { deps: unknown[] }) {
  const map = useMap();
  useEffect(() => {
    map.invalidateSize();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return null;
}

function gatekeeperIcon({
  dim,
  articulation,
}: {
  dim: boolean;
  articulation: boolean;
}) {
  // Articulation points (cut vertices) are the highest-stakes gatekeepers:
  // removing one splits the graph. Render them larger so they stand out.
  const cls = ["gatekeeper-marker", dim && "dim", articulation && "articulation"]
    .filter(Boolean)
    .join(" ");
  const size = articulation ? 22 : 16;
  return L.divIcon({
    html: `<div class="${cls}"></div>`,
    className: "",
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

export function MapView({
  network,
  gatekeepers,
  stage,
  input,
  model,
  disabledEdgeIds,
  onDisableEdge,
  onSelectEdge,
}: {
  network: RoadNetwork;
  gatekeepers: GatekeeperNode[];
  stage: Stage;
  input: InputMode;
  model: ModelMode;
  disabledEdgeIds: string[];
  onDisableEdge: (id: string) => void;
  onSelectEdge: (id: string | null) => void;
}) {
  const layerRef = useRef<LeafletGeoJSON | null>(null);

  // Re-render key for GeoJSON when network/disabled changes
  const geoKey = useMemo(
    () =>
      `${network.features.length}-${disabledEdgeIds.join(",")}-${stage}-${input}-${model}`,
    [network, disabledEdgeIds, stage, input, model],
  );

  // For the "extracted" stage show roads thicker & uniform amber; for "graph" use criticality colors
  const tileOpacity =
    stage === "satellite" ? 1 : stage === "extracted" ? 0.35 : 0.55;

  // Empty-state: backend returned a FeatureCollection with no edges (e.g. a
  // fully fragmented extraction). Only meaningful on stages that draw roads.
  const hasEdges = (network?.features?.length ?? 0) > 0;
  const showEmpty = stage !== "satellite" && !hasEdges;

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={BENGALURU_CENTER}
        zoom={13}
        scrollWheelZoom
        className="h-full w-full"
        preferCanvas
        zoomControl
      >
        <FlyOnChange deps={[stage, input, model]} />
        <TileLayer
          key={stage}
          url={TILE_URLS[stage]}
          attribution='&copy; OSM &copy; CARTO &copy; Esri'
          opacity={tileOpacity}
        />

        {stage !== "satellite" && hasEdges && (
        <GeoJSON
          key={geoKey}
          ref={(r) => {
            layerRef.current = r as unknown as LeafletGeoJSON;
          }}
          data={network as unknown as GeoJSON.GeoJsonObject}
          style={(feature) => {
            // Backend edge props (GET /api/network). `criticality` is 0..1 and
            // drives both the heat color and the stroke weight; clamp defensively
            // in case the backend ever emits an out-of-range value.
            const p = feature?.properties as RoadFeatureProperties | undefined;
            const id = p?.id ?? "";
            const crit = Math.max(0, Math.min(1, p?.criticality ?? 0));
            const disabled = disabledEdgeIds.includes(id);
            if (stage === "extracted") {
              return {
                color: disabled ? "#5a5246" : "#FF7A4E",
                weight: disabled ? 1.5 : 3,
                opacity: disabled ? 0.3 : 0.95,
              };
            }
            return {
              color: disabled ? "#5a5246" : criticalityColor(crit),
              weight: disabled ? 1.5 : 1.5 + crit * 3.5,
              opacity: disabled ? 0.3 : 0.95,
            };
          }}
          onEachFeature={(feature, layer) => {
            // Real backend properties: {id, criticality, travelTimeSec, lengthM,
            // roadClass, isBridge}. Coordinates are GeoJSON [lng,lat]; the
            // <GeoJSON> layer reprojects them to Leaflet [lat,lng] automatically.
            const p = feature.properties as RoadFeatureProperties;
            const crit = Math.max(0, Math.min(1, p.criticality ?? 0));
            layer.on({
              mouseover: (e) => {
                (e.target as L.Path).setStyle({ weight: 6, opacity: 1 });
              },
              mouseout: (e) => {
                const disabled = disabledEdgeIds.includes(p.id);
                (e.target as L.Path).setStyle({
                  weight: disabled ? 1.5 : 1.5 + crit * 3.5,
                  opacity: disabled ? 0.3 : 0.95,
                });
              },
              click: () => onSelectEdge(p.id),
            });
            layer.bindPopup(
              `<div style="font-family:'Plus Jakarta Sans',sans-serif;min-width:200px">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#7A6F65;font-weight:700">edge${p.roadClass ? ` · ${p.roadClass}` : ""}</div>
                <div style="font-size:15px;font-weight:700;color:#1F1B16;margin-top:2px">${p.id}</div>
                <div style="margin-top:10px;display:grid;grid-template-columns:auto auto;gap:6px 12px;font-size:11px">
                  <span style="color:#7A6F65">Criticality</span><span style="color:#1F1B16;font-weight:600">${crit.toFixed(2)}</span>
                  <span style="color:#7A6F65">Travel time</span><span style="color:#1F1B16;font-weight:600">${p.travelTimeSec}s</span>
                  <span style="color:#7A6F65">Length</span><span style="color:#1F1B16;font-weight:600">${p.lengthM} m</span>
                  <span style="color:#7A6F65">Road class</span><span style="color:#1F1B16;font-weight:600">${p.roadClass ?? "—"}</span>
                  <span style="color:#7A6F65">Bridge</span><span style="color:#1F1B16;font-weight:600">${p.isBridge ? "yes" : "no"}</span>
                </div>
                <button id="disable-${p.id}" style="margin-top:12px;width:100%;padding:8px;background:#FF7A4E;border:none;border-radius:9999px;color:#fff;font-weight:700;cursor:pointer;font-size:11px;font-family:inherit">
                  ${disabledEdgeIds.includes(p.id) ? "Re-enable edge" : "Disable edge"}
                </button>
              </div>`,
            );
            layer.on("popupopen", () => {
              const btn = document.getElementById(`disable-${p.id}`);
              btn?.addEventListener("click", () => {
                onDisableEdge(p.id);
                (layer as L.Path).closePopup();
              });
            });
          }}
        />
      )}

      {stage === "graph" &&
        gatekeepers.map((g) => {
          // betweenness === 0 is the backend/mock marker for a gatekeeper the
          // baseline model missed under occlusion (it is "dropped"/dimmed).
          const dim = g.betweenness === 0;
          // Articulation points (cut vertices) are the most critical nodes:
          // their removal disconnects the graph — emphasize them.
          const articulation = !!g.isArticulation;
          return (
            <Marker
              key={g.id}
              position={[g.lat, g.lng]}
              icon={gatekeeperIcon({ dim, articulation })}
            >
              <Popup>
                <div style={{ minWidth: 180 }}>
                  <div
                    style={{
                      fontSize: 10,
                      textTransform: "uppercase",
                      letterSpacing: ".08em",
                      color: "#7A6F65",
                      fontWeight: 700,
                    }}
                  >
                    {articulation ? "Articulation point" : "Gatekeeper node"}
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: "#1F1B16", marginTop: 2 }}>
                    {g.label}
                  </div>
                  <div style={{ fontSize: 11, color: "#7A6F65", marginTop: 4 }}>
                    {g.id} · betweenness{" "}
                    {dim ? "n/a (missed)" : g.betweenness.toFixed(2)}
                  </div>
                  {articulation && (
                    <div
                      style={{
                        fontSize: 11,
                        color: "#FF7A4E",
                        fontWeight: 700,
                        marginTop: 4,
                      }}
                    >
                      Removing this node disconnects the network.
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {/* Empty-state overlay: no road edges to render for this view. */}
      {showEmpty && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="pointer-events-auto flex items-center gap-2 rounded-full bg-black/60 px-4 py-2 text-xs font-semibold text-white/80 backdrop-blur">
            <AlertTriangle className="h-3.5 w-3.5 text-[#FF7A4E]" />
            No road network for this view yet.
          </div>
        </div>
      )}
    </div>
  );
}

// Floating overlay legend pieces (icons re-exported for parent)
export const MapIcons = { AlertTriangle, Power };
