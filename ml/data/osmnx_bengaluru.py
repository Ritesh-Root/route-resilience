"""Pull the real Bengaluru drive network with OSMnx -> backend node/edge schema.

This is the *validation / demo* graph source. The backend currently ships a
synthetic stand-in (see ``backend/app/services/network_factory.py``); this module
lets you swap in a real OpenStreetMap-derived graph that already speaks the
backend's schema, so ``graph_build.graph_to_geojson`` / ``criticality`` /
``gatekeeper_nodes`` all work unchanged.

Schema produced (matches backend/app/services/network_factory.py + graph_build.py):
    node attrs : lat (float), lng (float)
    edge attrs : id (str), length_m (float), road_class ("arterial"|"local"),
                 travel_time (float, seconds)
    node id    : stable string "n{osmid}" so it serialises cleanly to JSON.

Heavy / optional deps (osmnx, networkx) are imported lazily *inside* functions so
this file always imports cleanly even on a machine without them — same defensive
posture as ``backend/app/services/occlusion.py`` keeping the API image lean.

Run (downloads from the Overpass API; needs network access + ``osmnx`` installed)::

    python ml/data/osmnx_bengaluru.py

Pinned deps (add to ml/requirements.txt — do NOT pip install here):
    osmnx==1.9.4
    networkx==3.3
    # osmnx pulls geopandas / shapely / rtree transitively

NOTE: This calls the public Overpass API (a few MB of vector data, *not* the
multi-GB raster datasets). It does no model training and needs no GPU.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import only for type-checkers; never at runtime
    import networkx as nx

# Default place + cache location. OSMnx geocodes this string via Nominatim.
PLACE = "Bengaluru, Karnataka, India"
CITY = "Bengaluru"

# Match SPEED_KMH in backend/app/services/network_factory.py so travel times are
# consistent between the synthetic and the real graph.
SPEED_KMH: dict[str, float] = {"arterial": 50.0, "local": 30.0}

# OSM highway tags we treat as "arterial"; everything else drivable -> "local".
# (Keeps the two-class scheme the backend + front-end already render.)
_ARTERIAL_HIGHWAYS: frozenset[str] = frozenset(
    {"motorway", "trunk", "primary", "secondary",
     "motorway_link", "trunk_link", "primary_link", "secondary_link"}
)


def _require_osmnx() -> Any:
    """Import osmnx lazily with a clear, actionable error if it is missing."""
    try:
        import osmnx as ox  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "osmnx is required for ml/data/osmnx_bengaluru.py. "
            "Install the pinned deps (e.g. `pip install osmnx==1.9.4 networkx==3.3`) "
            "and ensure network access to the Overpass API."
        ) from exc
    return ox


def _require_networkx() -> Any:
    """Import networkx lazily (osmnx already depends on it, but be explicit)."""
    try:
        import networkx as nx  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "networkx is required. Install with `pip install networkx==3.3`."
        ) from exc
    return nx


def _highway_to_class(highway: Any) -> str:
    """Map an OSM ``highway`` tag (str or list) to the backend road_class."""
    # OSM occasionally stores a list of tags on a single way; take the first.
    if isinstance(highway, (list, tuple)):
        highway = highway[0] if highway else ""
    return "arterial" if str(highway) in _ARTERIAL_HIGHWAYS else "local"


def _edge_travel_time(length_m: float, road_class: str) -> float:
    """Seconds to traverse ``length_m`` at the class's nominal speed."""
    speed_ms = SPEED_KMH[road_class] * 1000.0 / 3600.0
    return round(length_m / speed_ms, 1) if speed_ms > 0 else 0.0


def fetch_drive_graph(place: str = PLACE) -> Any:
    """Download the drivable street network for ``place`` as a raw OSMnx MultiDiGraph.

    Returns the OSMnx graph untouched (nodes keyed by integer OSM ids, geometry in
    EPSG:4326 lat/lng). Conversion to the backend schema happens in
    :func:`to_backend_graph`.
    """
    ox = _require_osmnx()
    # network_type="drive" => roads cars can use (the demo is about urban mobility).
    # simplify=True collapses interstitial nodes so edges are road segments, not
    # every OSM vertex — closer to the backend's intersection-centric model.
    return ox.graph_from_place(place, network_type="drive", simplify=True)


def to_backend_graph(osm_graph: Any) -> "nx.Graph":
    """Convert an OSMnx MultiDiGraph to an undirected backend-schema ``nx.Graph``.

    - Collapses the directed multigraph to a simple undirected graph (the backend
      criticality/efficiency code operates on undirected graphs).
    - Renames node ids to ``"n{osmid}"`` strings and attaches lat/lng.
    - Computes ``length_m`` (prefers OSM's ``length``, else haversine), assigns
      ``road_class`` from the ``highway`` tag, and derives ``travel_time``.
    """
    nx = _require_networkx()

    G: "nx.Graph" = nx.Graph()

    # --- nodes: OSMnx stores y=lat, x=lng on each node ---
    def _node_id(osmid: Any) -> str:
        return f"n{osmid}"

    for osmid, data in osm_graph.nodes(data=True):
        G.add_node(
            _node_id(osmid),
            lat=round(float(data["y"]), 6),
            lng=round(float(data["x"]), 6),
        )

    # --- edges: dedupe parallel/reverse edges (MultiDiGraph -> simple Graph) ---
    for u, v, data in osm_graph.edges(data=True):
        su, sv = _node_id(u), _node_id(v)
        if su == sv:
            continue  # drop self-loops (roundabouts can introduce them)
        if G.has_edge(su, sv):
            continue  # keep the first occurrence; undirected, no parallels

        road_class = _highway_to_class(data.get("highway", ""))

        length_m = data.get("length")
        if length_m is None:
            # Fall back to great-circle distance using the same formula as the
            # backend (backend/app/services/geo.py haversine_m).
            length_m = _haversine_m(
                G.nodes[su]["lat"], G.nodes[su]["lng"],
                G.nodes[sv]["lat"], G.nodes[sv]["lng"],
            )
        length_m = round(float(length_m), 1)

        G.add_edge(
            su, sv,
            id=f"e_{su}__{sv}",
            length_m=length_m,
            road_class=road_class,
            travel_time=_edge_travel_time(length_m, road_class),
        )

    return G


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (mirrors backend/app/services/geo.py)."""
    import math

    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def build_bengaluru_network(place: str = PLACE) -> "nx.Graph":
    """End-to-end: download Bengaluru drive net and return a backend-schema graph.

    TODO(backend): to use this as the live demo graph, call this from
    ``network_factory.build_base_network`` (or cache its GeoJSON) and run the
    existing ``_annotate`` step to attach criticality/betweenness/isBridge before
    serialising via ``graph_build.graph_to_geojson``.
    """
    return to_backend_graph(fetch_drive_graph(place))


if __name__ == "__main__":  # pragma: no cover - manual/demo entrypoint
    # Prints node/edge counts so you can sanity-check the pull without the backend.
    print(f"Fetching drive network for: {PLACE!r} (via Overpass API)...")
    graph = build_bengaluru_network()
    print(f"city   : {CITY}")
    print(f"nodes  : {graph.number_of_nodes()}")
    print(f"edges  : {graph.number_of_edges()}")
    arterial = sum(
        1 for _, _, d in graph.edges(data=True) if d["road_class"] == "arterial"
    )
    print(f"  arterial edges: {arterial}")
    print(f"  local edges   : {graph.number_of_edges() - arterial}")
