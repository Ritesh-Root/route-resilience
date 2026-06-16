"""Skeleton (1-px road centreline) -> NetworkX graph (VECTORIZATION side).

This is the bridge between the segmentation model and the criticality engine.
The pipeline is::

    satellite chip --(CoaNet/U-Net)--> road probability mask
        --(threshold + skimage.morphology.skeletonize)--> 1-px skeleton
        --[THIS MODULE]--> NetworkX graph
        --> backend/app/services/criticality.py (UNCHANGED)

The graph this module emits matches the schema the backend already relies on, so
``criticality.py``, ``graph_build.py`` and the gatekeeper/simulation logic work
verbatim against the *real* extracted graph:

    node attrs : ``lat`` (float), ``lng`` (float)      # pixel -> geo, see below
    edge attrs : ``id`` (unique str), ``length_m`` (float, metres),
                 ``road_class`` ("arterial" | "local"), ``travel_time`` (s)

``criticality`` (edges), ``is_bridge`` (edges) and ``betweenness`` (nodes) are
NOT set here — they are derived analytics added downstream by
``network_factory._annotate``. We optionally call them via :func:`annotate` for a
self-contained graph, but the canonical place is the backend.

Algorithm (pure numpy core, no OpenCV/torch needed):
  1. Find junction pixels (skeleton pixels with >= 3 neighbours) and endpoints
     (exactly 1 neighbour) using an 8-neighbour count via a 3x3 convolution.
  2. Trace each edge by walking the skeleton from one node pixel to the next,
     accumulating the polyline of pixel coordinates.
  3. Collapse the polyline to two graph endpoints, measure its real-world length
     (geo-aware via the haversine of consecutive vertices, falling back to pixel
     length * metres-per-pixel), classify road_class by length/straightness, and
     attach a travel-time estimate from a per-class speed.

Heavy / optional deps are guarded so this file ALWAYS imports on a bare
interpreter (matches ml/data/tiling.py):
  - ``skimage`` only for the convenience mask->skeleton helper (lazy import).
  - ``rasterio`` only if you pass a GeoTIFF for the affine transform (lazy).
The graph-tracing core needs only numpy + networkx, both pinned in
``ml/requirements-ml.txt``.

Pixel -> geo: provide ONE of
  * ``transform`` : an affine (rasterio.Affine or 6-tuple a,b,c,d,e,f) mapping
    (col, row) -> (x=lng, y=lat). This is the GIS-correct path.
  * ``bbox``      : a {lat_min,lat_max,lng_min,lng_max} (same shape as
    ``network_factory.BBOX``) — chip pixels are linearly mapped into the box.
If neither is given, raw pixel coordinates are stored as lng=col, lat=row, which
keeps the graph topologically correct (criticality is geometry-agnostic) but the
map overlay will be in pixel space — fine for a quick demo, wrong for real geo.

Usage (NO dataset download, NO training, NO torch):

    # 1) From a saved skeleton (.npy boolean/0-1 array), demo bbox = Bengaluru:
    python ml/vectorize/skeleton_to_graph.py --skeleton path/to/skeleton.npy \
        --bbox-bengaluru --out graph.graphml

    # 2) From a probability/binary MASK (.npy); we skeletonize it (needs skimage):
    python ml/vectorize/skeleton_to_graph.py --mask path/to/mask.npy \
        --threshold 0.5 --bbox-bengaluru --out graph.graphml

    # 3) Dependency-free correctness check (numpy + networkx only):
    python ml/vectorize/skeleton_to_graph.py --self-test

TODO(you): wire :func:`skeleton_to_graph` (or :func:`mask_to_graph`) into the
backend ``network_factory.build_real_network()`` — return the graph, call
``_annotate`` on it, and the rest of the stack is unchanged.
"""
from __future__ import annotations

import argparse
import math
from typing import Iterable, Sequence

import networkx as nx
import numpy as np

# --------------------------------------------------------------------------- #
# Constants — kept in lock-step with backend/app/services/network_factory.py so
# the extracted graph is interchangeable with the synthetic demo graph.
# --------------------------------------------------------------------------- #
SPEED_KMH: dict[str, float] = {"arterial": 50.0, "local": 30.0}

# Bengaluru bbox copied from network_factory.BBOX for the --bbox-bengaluru demo.
BENGALURU_BBOX: dict[str, float] = {
    "lat_min": 12.90, "lat_max": 13.04, "lng_min": 77.52, "lng_max": 77.66,
}

# A road is promoted to "arterial" when it is long AND fairly straight (a major
# through-road), else "local". Tunable; mirrors the arterial/local split the
# demo network uses for travel-time weighting.
ARTERIAL_MIN_LENGTH_M: float = 400.0
ARTERIAL_MAX_TORTUOSITY: float = 1.15  # path_len / straight_line_len; 1.0 = straight


# --------------------------------------------------------------------------- #
# Geo helpers (haversine duplicated from backend geo.py — this module must stay
# importable WITHOUT the backend package on sys.path).
# --------------------------------------------------------------------------- #
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lng points, in metres.

    Mirrors ``backend/app/services/geo.haversine_m`` (kept local so the ML side
    has no import dependency on the FastAPI app).
    """
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Pixel -> geo mapping. Either an affine transform or a linear bbox mapping.
# --------------------------------------------------------------------------- #
class PixelToGeo:
    """Maps (row, col) skeleton pixels to (lat, lng).

    Construct via :meth:`from_affine` (GIS-correct, from a GeoTIFF) or
    :meth:`from_bbox` (linear stretch of an image into a lat/lng box, for demos).
    With no mapping, :meth:`identity` stores pixel coordinates directly.
    """

    def __init__(self, fn, is_geo: bool = True):
        # is_geo=False marks the identity (pixel-space) mapping, so length
        # functions use pixel distance * metres_per_pixel instead of haversine.
        self._fn = fn
        self.is_geo = is_geo

    def __call__(self, row: float, col: float) -> tuple[float, float]:
        """Return ``(lat, lng)`` for a (possibly fractional) pixel coordinate."""
        return self._fn(row, col)

    @classmethod
    def identity(cls) -> "PixelToGeo":
        """No geo-referencing: lat=row, lng=col (topology-correct, pixel space)."""
        return cls(lambda row, col: (float(row), float(col)), is_geo=False)

    @classmethod
    def from_bbox(cls, bbox: dict[str, float], height: int, width: int) -> "PixelToGeo":
        """Linearly map an (height x width) image into a lat/lng bounding box.

        Row 0 is the TOP of the image -> ``lat_max`` (north), matching how raster
        rows increase downward (south). Column 0 -> ``lng_min`` (west).
        """
        if height < 2 or width < 2:
            raise ValueError("from_bbox needs an image at least 2x2 px")
        lat_max, lat_min = bbox["lat_max"], bbox["lat_min"]
        lng_min, lng_max = bbox["lng_min"], bbox["lng_max"]

        def fn(row: float, col: float) -> tuple[float, float]:
            fy = row / (height - 1)   # 0 at top .. 1 at bottom
            fx = col / (width - 1)    # 0 at left .. 1 at right
            lat = lat_max - fy * (lat_max - lat_min)  # top row -> north
            lng = lng_min + fx * (lng_max - lng_min)
            return (lat, lng)

        return cls(fn)

    @classmethod
    def from_affine(cls, transform: Sequence[float] | object) -> "PixelToGeo":
        """Map via an affine transform giving (x=lng, y=lat) from (col, row).

        ``transform`` may be a rasterio ``Affine`` (callable as ``t * (col, row)``)
        or a 6-tuple ``(a, b, c, d, e, f)`` with
        ``x = a*col + b*row + c`` and ``y = d*col + e*row + f``.
        """
        # rasterio.Affine supports ``transform * (col, row) -> (x, y)``.
        if hasattr(transform, "__mul__") and not isinstance(transform, (tuple, list)):
            def fn(row: float, col: float) -> tuple[float, float]:
                x, y = transform * (col, row)  # type: ignore[operator]
                return (float(y), float(x))   # (lat, lng)
            return cls(fn)

        a, b, c, d, e, f = transform  # type: ignore[misc]

        def fn(row: float, col: float) -> tuple[float, float]:
            x = a * col + b * row + c
            y = d * col + e * row + f
            return (float(y), float(x))  # (lat, lng)

        return cls(fn)


# --------------------------------------------------------------------------- #
# Pure-numpy skeleton topology: neighbour counts -> junctions / endpoints.
# --------------------------------------------------------------------------- #
# 8-connected neighbour offsets (row, col), excluding the centre.
_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _as_bool_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Coerce input to a 2-D boolean skeleton (any non-zero pixel is 'on')."""
    arr = np.asarray(skeleton)
    if arr.ndim != 2:
        raise ValueError(f"skeleton must be 2-D (H, W); got shape {arr.shape}")
    return arr.astype(bool)


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Count 8-connected 'on' neighbours for every pixel (0 off skeleton).

    Pure numpy: sum eight shifted copies of the boolean image. The result is only
    meaningful on skeleton pixels; off-skeleton pixels are forced to 0.
    """
    s = skel.astype(np.uint8)
    padded = np.pad(s, 1, mode="constant", constant_values=0)
    acc = np.zeros_like(s, dtype=np.uint8)
    h, w = s.shape
    for dr, dc in _NEIGHBOURS:
        acc += padded[1 + dr: 1 + dr + h, 1 + dc: 1 + dc + w]
    acc[~skel] = 0
    return acc


def node_pixels(skel: np.ndarray) -> set[tuple[int, int]]:
    """Skeleton pixels that are graph nodes: endpoints (deg 1) or junctions (deg>=3).

    Degree-2 pixels are interior points of an edge and are NOT nodes; they are
    consumed while tracing. Isolated pixels (deg 0) are ignored as noise.

    NOTE: 8-connectivity inflates the apparent degree near a junction (diagonal
    neighbours of neighbours get counted), so several adjacent pixels around one
    real intersection all read as deg>=3. :func:`cluster_nodes` merges those into
    a single graph node, which is what tracing actually uses.
    """
    deg = neighbour_count(skel)
    rr, cc = np.where(skel)
    nodes: set[tuple[int, int]] = set()
    for r, c in zip(rr.tolist(), cc.tolist()):
        d = int(deg[r, c])
        if d == 1 or d >= 3:
            nodes.add((r, c))
    return nodes


def cluster_nodes(
    node_px: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], tuple[int, int]], set[tuple[int, int]]]:
    """Merge 8-adjacent node pixels into one representative each (junction blobs).

    Returns ``(pixel -> representative, set_of_representatives)``. The representative
    is the cluster's centroid pixel (nearest member to the mean), giving a stable,
    well-placed graph node for a thick/diagonal junction. Endpoints are usually
    singleton clusters and map to themselves.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {p: p for p in node_px}

    def find(p: tuple[int, int]) -> tuple[int, int]:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (r, c) in node_px:
        for dr, dc in _NEIGHBOURS:
            nb = (r + dr, c + dc)
            if nb in parent:
                union((r, c), nb)

    clusters: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in node_px:
        clusters.setdefault(find(p), []).append(p)

    rep_of: dict[tuple[int, int], tuple[int, int]] = {}
    reps: set[tuple[int, int]] = set()
    for members in clusters.values():
        mean_r = sum(p[0] for p in members) / len(members)
        mean_c = sum(p[1] for p in members) / len(members)
        rep = min(members, key=lambda p: (p[0] - mean_r) ** 2 + (p[1] - mean_c) ** 2)
        reps.add(rep)
        for p in members:
            rep_of[p] = rep
    return rep_of, reps


def _on_neighbours(skel: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    """List in-bounds, 'on' 8-neighbours of pixel (r, c)."""
    h, w = skel.shape
    out: list[tuple[int, int]] = []
    for dr, dc in _NEIGHBOURS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and skel[nr, nc]:
            out.append((nr, nc))
    return out


def _trace_edge(
    skel: np.ndarray,
    start: tuple[int, int],
    first_step: tuple[int, int],
    node_px: set[tuple[int, int]],
    visited_edge_px: set[frozenset],
) -> list[tuple[int, int]] | None:
    """Walk from a node pixel ``start`` along degree-2 pixels to the next node pixel.

    Returns the polyline ``[start, ..., end_node_pixel]`` (both ends inclusive), or
    ``None`` if this first step was already walked from the other end (dedupe).
    ``node_px`` is the set of ALL node pixels (pre-clustering); reaching any of them
    that is not the start pixel ends the edge. Step keys make each edge walk once.
    """
    step_key = frozenset((start, first_step))
    if step_key in visited_edge_px:
        return None
    visited_edge_px.add(step_key)

    # If the very first step lands on another node pixel, this is a unit-length edge.
    if first_step in node_px:
        return [start, first_step]

    prev, cur = start, first_step
    poly: list[tuple[int, int]] = [start, cur]
    max_steps = int(skel.sum()) + 2
    for _ in range(max_steps):
        nbrs = [n for n in _on_neighbours(skel, *cur) if n != prev]
        # Prefer continuing along non-node (degree-2) pixels; a node neighbour ends it.
        node_nbrs = [n for n in nbrs if n in node_px]
        if node_nbrs:
            nxt = node_nbrs[0]
            visited_edge_px.add(frozenset((cur, nxt)))
            poly.append(nxt)
            return poly  # reached the far node — edge complete
        if not nbrs:
            return poly  # dangling tail (shouldn't normally happen)
        nxt = nbrs[0]
        visited_edge_px.add(frozenset((cur, nxt)))
        poly.append(nxt)
        prev, cur = cur, nxt
    return poly  # safety: return what we have rather than loop forever


def trace_edges(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Trace every skeleton edge as a polyline of pixel coords between two nodes.

    Adjacent junction pixels are first clustered into single nodes (:func:`cluster_nodes`)
    so a thick / diagonally-counted intersection becomes ONE graph node. Polylines
    whose two ends fall in the same cluster (a tiny loop around the junction) are
    dropped. Handles: normal edges, pure loops (no junction) by seeding a node, and
    ignores isolated pixels.
    """
    node_px = node_pixels(skel)
    rep_of, _reps = cluster_nodes(node_px)
    visited: set[frozenset] = set()
    raw: list[list[tuple[int, int]]] = []

    for (nr, nc) in node_px:
        for step in _on_neighbours(skel, nr, nc):
            poly = _trace_edge(skel, (nr, nc), step, node_px, visited)
            if poly is not None and len(poly) >= 2:
                raw.append(poly)

    # Map endpoints to their cluster representatives; drop intra-cluster stubs.
    polylines: list[list[tuple[int, int]]] = []
    for poly in raw:
        a, b = rep_of.get(poly[0], poly[0]), rep_of.get(poly[-1], poly[-1])
        if a == b:
            continue  # both ends are the same junction cluster — spurious stub
        polylines.append([a, *poly[1:-1], b])

    # Pure loops have no degree-1/3 node; detect any untraced skeleton component
    # and break it open at an arbitrary pixel so it still becomes an edge.
    rr, cc = np.where(skel)
    all_px = set(zip(rr.tolist(), cc.tolist()))
    traced_px = {p for poly in raw for p in poly}
    leftover = all_px - traced_px
    deg = neighbour_count(skel)
    while leftover:
        seed = next(iter(leftover))
        if int(deg[seed]) != 2:
            leftover.discard(seed)  # not a clean degree-2 loop pixel; drop as noise
            continue
        seed_neighbours = _on_neighbours(skel, *seed)
        poly = _trace_edge(skel, seed, seed_neighbours[0], {seed}, visited)
        if poly is not None and len(poly) >= 2:
            polylines.append(poly)
            traced_px.update(poly)
        leftover -= traced_px
        leftover.discard(seed)
    return polylines


# --------------------------------------------------------------------------- #
# Geometry of a traced polyline: real length, straight-line span, tortuosity.
# --------------------------------------------------------------------------- #
def _polyline_length_m(
    poly: list[tuple[int, int]],
    p2g: PixelToGeo,
    metres_per_pixel: float | None,
) -> float:
    """Sum the geo length of a pixel polyline.

    If ``p2g`` produces real lat/lng (bbox or affine), uses haversine between
    consecutive vertices (curvature-aware). For identity (pixel-space) mapping,
    falls back to Euclidean pixel distance * ``metres_per_pixel`` (default 1.0).
    """
    if len(poly) < 2:
        return 0.0
    if p2g.is_geo:
        geo = [p2g(r, c) for r, c in poly]
        return sum(
            haversine_m(lat1, lng1, lat2, lng2)
            for (lat1, lng1), (lat2, lng2) in zip(geo, geo[1:])
        )
    # Pure pixel-space (identity mapping): Euclidean px distance scaled to metres.
    mpp = 1.0 if metres_per_pixel is None else metres_per_pixel
    return sum(math.hypot(r2 - r1, c2 - c1)
               for (r1, c1), (r2, c2) in zip(poly, poly[1:])) * mpp


def _straight_line_m(
    poly: list[tuple[int, int]],
    p2g: PixelToGeo,
    metres_per_pixel: float | None,
) -> float:
    """Geo distance between the two endpoints of the polyline (for tortuosity)."""
    (r0, c0), (r1, c1) = poly[0], poly[-1]
    if p2g.is_geo:
        lat0, lng0 = p2g(r0, c0)
        lat1, lng1 = p2g(r1, c1)
        return haversine_m(lat0, lng0, lat1, lng1)
    mpp = 1.0 if metres_per_pixel is None else metres_per_pixel
    return math.hypot(r1 - r0, c1 - c0) * mpp


def classify_road(length_m: float, tortuosity: float) -> str:
    """Heuristic road class: long + straight -> 'arterial', else 'local'.

    Mirrors the two-tier (arterial/local) speed model in network_factory so the
    travel-time weighting is consistent with the demo graph.
    """
    if length_m >= ARTERIAL_MIN_LENGTH_M and tortuosity <= ARTERIAL_MAX_TORTUOSITY:
        return "arterial"
    return "local"


def travel_time_s(length_m: float, road_class: str) -> float:
    """Free-flow travel time (seconds) = length / class speed."""
    speed_ms = SPEED_KMH[road_class] * 1000.0 / 3600.0
    return length_m / speed_ms if speed_ms > 0 else 0.0


# --------------------------------------------------------------------------- #
# Assembly: polylines -> NetworkX graph matching the backend schema.
# --------------------------------------------------------------------------- #
def _pixel_node_id(r: int, c: int) -> str:
    """Stable string node id from a pixel coordinate (matches backend str ids)."""
    return f"px_{r}_{c}"


def build_graph(
    polylines: Iterable[list[tuple[int, int]]],
    p2g: PixelToGeo,
    metres_per_pixel: float | None = None,
) -> nx.Graph:
    """Assemble traced polylines into the backend-compatible NetworkX graph.

    Nodes carry ``lat``/``lng``; edges carry ``id``/``length_m``/``road_class``/
    ``travel_time`` — exactly what ``criticality.py`` and ``graph_build.py`` read.
    Parallel/duplicate segments between the same node pair are collapsed (the
    backend graph is a simple ``nx.Graph``), keeping the longer polyline.
    """
    G = nx.Graph()
    for poly in polylines:
        if len(poly) < 2:
            continue
        (r0, c0), (r1, c1) = poly[0], poly[-1]
        u, v = _pixel_node_id(r0, c0), _pixel_node_id(r1, c1)
        if u == v:
            continue  # self-loop (degenerate ring collapsed to a point) — skip

        lat_u, lng_u = p2g(r0, c0)
        lat_v, lng_v = p2g(r1, c1)
        G.add_node(u, lat=round(lat_u, 6), lng=round(lng_u, 6))
        G.add_node(v, lat=round(lat_v, 6), lng=round(lng_v, 6))

        length = _polyline_length_m(poly, p2g, metres_per_pixel)
        straight = _straight_line_m(poly, p2g, metres_per_pixel)
        tortuosity = (length / straight) if straight > 0 else 1.0
        road_class = classify_road(length, tortuosity)

        # Collapse parallel edges: keep whichever segment is longer (likelier the
        # true road; the short one is usually a skeleton artefact / shortcut).
        if G.has_edge(u, v) and G[u][v].get("length_m", 0.0) >= length:
            continue
        G.add_edge(
            u, v,
            id=f"e_{u}__{v}",
            length_m=round(length, 1),
            road_class=road_class,
            travel_time=round(travel_time_s(length, road_class), 1),
        )
    return G


def annotate(G: nx.Graph) -> nx.Graph:
    """Attach criticality/is_bridge (edges) + betweenness (nodes), in place.

    OPTIONAL convenience that reproduces ``network_factory._annotate`` so a graph
    built here is self-contained for offline inspection. In production the backend
    owns this step; the values are recomputed there after fragmentation/occlusion
    views are derived. Returns ``G`` for chaining.
    """
    if G.number_of_edges() == 0:
        return G
    eb = nx.edge_betweenness_centrality(G, weight="travel_time", normalized=True)
    nb = nx.betweenness_centrality(G, weight="travel_time", normalized=True)
    bridge_set = {frozenset(e) for e in nx.bridges(G)}
    max_eb = max(eb.values()) if eb else 1.0
    for u, v, d in G.edges(data=True):
        raw = eb.get((u, v), eb.get((v, u), 0.0))
        d["criticality"] = round((raw / max_eb) if max_eb > 0 else 0.0, 4)
        d["is_bridge"] = frozenset((u, v)) in bridge_set
    max_nb = max(nb.values()) if nb else 1.0
    for n, val in nb.items():
        G.nodes[n]["betweenness"] = round((val / max_nb) if max_nb > 0 else 0.0, 4)
    return G


# --------------------------------------------------------------------------- #
# Top-level entry points.
# --------------------------------------------------------------------------- #
def skeleton_to_graph(
    skeleton: np.ndarray,
    *,
    bbox: dict[str, float] | None = None,
    transform: Sequence[float] | object | None = None,
    metres_per_pixel: float | None = None,
    do_annotate: bool = False,
) -> nx.Graph:
    """Convert a 1-px skeleton array into a backend-compatible NetworkX graph.

    Parameters
    ----------
    skeleton:
        2-D array; any non-zero pixel is treated as skeleton (road centreline).
    bbox:
        Optional ``{lat_min,lat_max,lng_min,lng_max}`` to linearly geo-reference
        the image (demo path). Mutually exclusive with ``transform``.
    transform:
        Optional affine ((col,row)->(lng,lat)); GIS-correct path for GeoTIFF chips.
    metres_per_pixel:
        Used only when neither ``bbox`` nor ``transform`` is given (identity /
        pixel-space mapping) to scale pixel lengths into metres. Defaults to 1.0.
    do_annotate:
        If True, also attach criticality/betweenness/is_bridge via :func:`annotate`.
    """
    if bbox is not None and transform is not None:
        raise ValueError("pass at most one of bbox / transform")
    skel = _as_bool_skeleton(skeleton)
    h, w = skel.shape

    if transform is not None:
        p2g = PixelToGeo.from_affine(transform)
    elif bbox is not None:
        p2g = PixelToGeo.from_bbox(bbox, h, w)
    else:
        p2g = PixelToGeo.identity()

    polylines = trace_edges(skel)
    G = build_graph(polylines, p2g, metres_per_pixel=metres_per_pixel)
    return annotate(G) if do_annotate else G


def mask_to_graph(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    bbox: dict[str, float] | None = None,
    transform: Sequence[float] | object | None = None,
    metres_per_pixel: float | None = None,
    do_annotate: bool = False,
) -> nx.Graph:
    """Threshold + skeletonize a road probability/binary mask, then build a graph.

    Skeletonization uses ``skimage.morphology.skeletonize`` (lazy-imported and
    guarded). If scikit-image is absent, raises a clear, actionable error.
    """
    arr = np.asarray(mask)
    binary = arr >= threshold if arr.dtype.kind == "f" else arr.astype(bool)
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "mask_to_graph needs scikit-image for skeletonize "
            "(pip install scikit-image==0.24.0 — see ml/requirements-ml.txt). "
            "Alternatively skeletonize the mask yourself and call "
            "skeleton_to_graph() with the result."
        ) from exc
    skel = skeletonize(binary)
    return skeleton_to_graph(
        skel, bbox=bbox, transform=transform,
        metres_per_pixel=metres_per_pixel, do_annotate=do_annotate,
    )


# --------------------------------------------------------------------------- #
# Guarded I/O for the CLI (numpy .npy always works; rasterio is optional).
# --------------------------------------------------------------------------- #
def _load_array(path: str) -> np.ndarray:
    """Load a 2-D array from .npy, or a single-band raster via rasterio (lazy)."""
    if path.lower().endswith(".npy"):
        return np.load(path)
    try:
        import rasterio  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"Reading '{path}' needs rasterio (pip install rasterio==1.3.11), or "
            "convert it to a .npy array first. See ml/requirements-ml.txt."
        ) from exc
    with rasterio.open(path) as src:
        return src.read(1)  # first band


def _load_transform(path: str) -> object | None:
    """Read the affine transform of a GeoTIFF (None for .npy / no rasterio)."""
    if path.lower().endswith(".npy"):
        return None
    try:
        import rasterio  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - depends on env
        return None
    with rasterio.open(path) as src:
        return src.transform


def _write_graph(G: nx.Graph, path: str) -> None:
    """Persist the graph. GraphML/GEXF via networkx; .json as GeoJSON-ish edges."""
    lower = path.lower()
    if lower.endswith(".graphml"):
        nx.write_graphml(G, path)
    elif lower.endswith(".gexf"):
        nx.write_gexf(G, path)
    elif lower.endswith(".json"):
        import json
        # Lightweight node-link dump (round-trippable via nx.node_link_graph).
        data = nx.node_link_data(G, edges="links")
        with open(path, "w") as fh:
            json.dump(data, fh)
    else:
        raise ValueError(f"unsupported output extension: {path} "
                         "(use .graphml, .gexf or .json)")


# --------------------------------------------------------------------------- #
# Dependency-free self-test (numpy + networkx only — no skimage/rasterio/torch).
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Validate topology + schema on a hand-built skeleton; no heavy deps."""
    # A 'T'-junction skeleton on an 11x11 grid:
    #   - a horizontal bar (row 5, cols 1..9)
    #   - a vertical stem (rows 5..9, col 5)
    # Expected nodes: 2 horizontal endpoints, 1 vertical endpoint, 1 junction.
    skel = np.zeros((11, 11), dtype=bool)
    skel[5, 1:10] = True       # horizontal bar
    skel[5:10, 5] = True       # vertical stem (shares (5,5) with the bar -> junction)

    nodes = node_pixels(skel)
    assert (5, 1) in nodes and (5, 9) in nodes, "horizontal endpoints expected"
    assert (9, 5) in nodes, "vertical endpoint expected"
    assert (5, 5) in nodes, "junction at the T expected"
    # Degree-2 interior pixels must NOT be nodes.
    assert (5, 3) not in nodes and (7, 5) not in nodes, "interior pixels are not nodes"

    G = skeleton_to_graph(skel, bbox=BENGALURU_BBOX, do_annotate=True)
    # A clean T has exactly 3 edges meeting at the junction.
    assert G.number_of_edges() == 3, f"expected 3 edges, got {G.number_of_edges()}"
    assert G.number_of_nodes() == 4, f"expected 4 nodes, got {G.number_of_nodes()}"

    # Schema: every edge must carry the attributes criticality.py / graph_build.py read.
    for u, v, d in G.edges(data=True):
        for key in ("id", "length_m", "road_class", "travel_time"):
            assert key in d, f"edge missing '{key}'"
        assert d["road_class"] in SPEED_KMH, d["road_class"]
        assert d["length_m"] > 0 and d["travel_time"] > 0
        assert "criticality" in d and "is_bridge" in d  # added by annotate()
    for n, nd in G.nodes(data=True):
        assert "lat" in nd and "lng" in nd, "node missing geo coords"
        assert BENGALURU_BBOX["lat_min"] - 1e-6 <= nd["lat"] <= BENGALURU_BBOX["lat_max"] + 1e-6
        assert BENGALURU_BBOX["lng_min"] - 1e-6 <= nd["lng"] <= BENGALURU_BBOX["lng_max"] + 1e-6

    # The graph must feed the REAL criticality engine unchanged. Import is best
    # effort: only run if the backend package happens to be importable.
    try:
        import sys, os
        backend = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
        sys.path.insert(0, os.path.abspath(backend))
        from app.services import criticality as crit  # type: ignore
        eb = crit.edge_betweenness(G)
        assert len(eb) == G.number_of_edges()
        _ = crit.resilience_index(G, crit.global_efficiency_weighted(G))
        backend_note = "criticality.py consumed the graph OK"
    except Exception as exc:  # backend not on path in isolation — that's fine
        backend_note = f"(skipped backend integration check: {type(exc).__name__})"

    # Identity (pixel-space) mapping still produces a valid topological graph.
    Gpx = skeleton_to_graph(skel, metres_per_pixel=10.0)
    assert Gpx.number_of_edges() == 3 and Gpx.number_of_nodes() == 4

    print("[skeleton_to_graph] self-test OK:",
          dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(),
               road_classes=sorted({d["road_class"] for *_e, d in G.edges(data=True)})),
          "|", backend_note)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--skeleton", help="path to a 1-px skeleton (.npy or raster)")
    src.add_argument("--mask", help="path to a road mask (.npy/raster); we skeletonize it")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="binarisation threshold for a float --mask (default 0.5)")

    geo = ap.add_mutually_exclusive_group()
    geo.add_argument("--bbox-bengaluru", action="store_true",
                     help="geo-reference with the demo Bengaluru bbox")
    geo.add_argument("--geotiff-transform", action="store_true",
                     help="use the input raster's affine transform (needs rasterio)")
    ap.add_argument("--metres-per-pixel", type=float, default=None,
                    help="metres/pixel for identity (pixel-space) mapping; default 1.0")

    ap.add_argument("--out", help="output graph path (.graphml / .gexf / .json)")
    ap.add_argument("--annotate", action="store_true",
                    help="attach criticality/betweenness/is_bridge before writing")
    ap.add_argument("--self-test", action="store_true",
                    help="run dependency-free correctness checks and exit")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.skeleton and not args.mask:
        ap.error("provide --skeleton or --mask (or --self-test). This script never "
                 "downloads data or runs the model; point it at your own array.")

    in_path = args.skeleton or args.mask
    bbox = BENGALURU_BBOX if args.bbox_bengaluru else None
    transform = _load_transform(in_path) if args.geotiff_transform else None

    arr = _load_array(in_path)
    if args.mask:
        G = mask_to_graph(arr, threshold=args.threshold, bbox=bbox,
                          transform=transform, metres_per_pixel=args.metres_per_pixel,
                          do_annotate=args.annotate)
    else:
        G = skeleton_to_graph(arr, bbox=bbox, transform=transform,
                              metres_per_pixel=args.metres_per_pixel,
                              do_annotate=args.annotate)

    print(f"[skeleton_to_graph] built graph: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if args.out:
        _write_graph(G, args.out)
        print(f"[skeleton_to_graph] wrote -> {args.out}")
    else:
        print("[skeleton_to_graph] no --out given; not writing (add --out graph.graphml)")


if __name__ == "__main__":
    main()
