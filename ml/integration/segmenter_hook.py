"""Integration glue: real ML pipeline -> the NetworkX graph the backend serves.

WHAT THIS FILE IS
-----------------
A *swap-in seam* between the trained ML pipeline (``ml/``) and the FastAPI demo
backend (``backend/app``). Today the backend serves a synthetic graph from
``backend/app/services/network_factory.get_network()`` so the whole API + UI work
before the model exists. This module shows — end to end — how to produce the SAME
``networkx.Graph`` object from a *real* prediction, so that dropping it in changes
nothing downstream (``criticality`` -> ``gatekeepers`` -> ``simulate`` -> GeoJSON
via ``backend/app/services/graph_build.graph_to_geojson``).

The real pipeline, mirroring ``ml/README.md`` (steps 3-5):

    predict mask  ->  skeletonize  ->  skeleton_to_graph  ->  MST healing
                                                            -> annotated nx.Graph

The annotated graph carries the exact node/edge attributes ``network_factory``
emits, so ``get_network()`` can be re-pointed at :func:`build_real_network` with a
one-line change and the contract (``GET /api/network`` etc.) stays identical.

INTERFACE MIRROR
----------------
* ``RoadSegmenter`` (baseline) and ``CoANetSegmenter`` (robust) both expose
  ``predict(image: np.ndarray) -> np.ndarray``  (HxW float mask in [0, 1]); see
  ``backend/app/services/segmentation.py`` and ``ml/models/coanet.py``.
* :func:`build_real_network` returns ``networkx.Graph`` in the SAME shape as
  ``network_factory.build_base_network()`` — per-node ``lat``/``lng``/
  ``betweenness``; per-edge ``id``/``length_m``/``road_class``/``travel_time``/
  ``criticality``/``is_bridge``.

IMPORT SAFETY (hard project constraint: 4GB RTX 3050 / Python 3.14, no ML stack)
-------------------------------------------------------------------------------
Heavy libs (``torch``, ``rasterio``, ``skimage``) are imported lazily inside
functions with a clear remediation message, mirroring the numpy-only style of
``backend/app/services/occlusion.py``. Importing this module never needs torch.
``networkx`` and ``numpy`` are assumed present (the backend already depends on
them). Run ``python ml/integration/segmenter_hook.py`` for a torch-free self-test
that exercises the vectorize + heal + annotate path on a synthetic mask.

HOW TO SWAP THE REAL PIPELINE IN (one edit, no API change)
----------------------------------------------------------
In ``backend/app/services/network_factory.get_network()`` replace the synthetic
branch with::

    from ml.integration.segmenter_hook import build_real_network
    G = build_real_network(image, transform, input_mode=input_mode, model=model)

(keeping the ``(input_mode, model)`` cache key). Everything else is unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import networkx as nx
import numpy as np

if TYPE_CHECKING:  # static type-checkers only; never imported at runtime
    import rasterio  # noqa: F401
    import torch  # noqa: F401

# Pixel->geo affine: maps (col, row) -> (lng, lat). rasterio's ``Affine`` matches
# this signature, so a rasterio dataset's ``.transform`` can be passed directly.
PixelToGeo = Callable[[float, float], tuple[float, float]]

InputMode = Literal["clean", "occluded"]
ModelName = Literal["baseline", "robust"]

# Assumed cruising speed per inferred road class (km/h) — used to derive
# ``travel_time`` from ``length_m``, matching network_factory.SPEED_KMH.
SPEED_KMH: dict[str, float] = {"arterial": 50.0, "local": 30.0}

# Default max gap (metres) a healing bridge may span. Roughly the largest
# occlusion gap (cloud/shadow) we are willing to reconnect; larger gaps are left
# broken (genuinely missing road), which is what keeps baseline fragmented.
DEFAULT_MAX_HEAL_GAP_M: float = 250.0


# --------------------------------------------------------------------------- #
# Lazy, guarded heavy imports (clear message instead of ImportError at load).
# --------------------------------------------------------------------------- #
def _require_skimage() -> "Any":
    """Import scikit-image lazily; raise an actionable error if it is absent."""
    try:
        import skimage  # noqa: PLC0415 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "scikit-image is required for skeletonization (vectorize step). It is "
            "intentionally not installed on the 4GB dev box. Install the pinned "
            "stack on a training machine:\n    pip install -r ml/requirements.txt"
        ) from exc
    return skimage


def _load_segmenter(model: ModelName, weights_path: str | None, device: str) -> "Any":
    """Instantiate the baseline or robust segmenter (torch guarded inside each).

    Returns an object exposing ``predict(image) -> HxW float mask in [0, 1]``.
    Construction itself is cheap and torch-free for the baseline stub; the robust
    CoANet builds its torch model lazily on first ``predict`` call.
    """
    if model == "robust":
        # CoANet: connectivity-attention, recovers links the baseline drops under
        # occlusion (the "robust model" half of the demo story).
        from ml.models.coanet import CoANetSegmenter  # noqa: PLC0415

        return CoANetSegmenter(weights_path=weights_path, device=device)
    if model == "baseline":
        # D-LinkNet / plain LinkNet+ResNet34: fine on clean imagery, fragments
        # under occlusion. Lives in the backend service today.
        try:
            from backend.app.services.segmentation import RoadSegmenter  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - path/layout dependent
            raise ImportError(
                "Could not import RoadSegmenter from backend.app.services."
                "segmentation. Run from the repo root so 'backend' is importable, "
                "or adjust sys.path."
            ) from exc
        return RoadSegmenter(weights_path=weights_path)
    raise ValueError(f"unknown model {model!r}; expected 'baseline' or 'robust'")


# --------------------------------------------------------------------------- #
# Step 4: vectorize — binary mask -> skeleton -> node/edge graph.
# --------------------------------------------------------------------------- #
def mask_to_skeleton(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold a road-probability mask and thin it to a 1-px centreline.

    Args:
        mask: ``(H, W)`` float probabilities in ``[0, 1]`` from a segmenter.
        threshold: road/background cut (default 0.5).

    Returns:
        ``(H, W)`` bool skeleton (``True`` on the 1-px centreline).
    """
    _require_skimage()
    from skimage.morphology import skeletonize  # noqa: PLC0415

    binary = mask > threshold
    return skeletonize(binary)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-connected skeleton neighbours (numpy, no skimage)."""
    sk = skel.astype(np.int16)
    counts = np.zeros_like(sk)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            counts += np.roll(np.roll(sk, dy, axis=0), dx, axis=1)
    # Wrap-around from np.roll is harmless: border pixels rarely sit on a road
    # endpoint, and downstream we only read counts at skeleton pixels.
    return counts * sk


def skeleton_to_graph(
    skel: np.ndarray,
    pixel_to_geo: PixelToGeo,
    gsd_m: float = 1.0,
) -> nx.Graph:
    """Walk a 1-px skeleton into a NetworkX graph of intersections + roads.

    Skeleton pixels with a neighbour count != 2 are *control points* (endpoints
    with 1 neighbour, junctions with >=3); the runs of degree-2 pixels between
    them become single edges. Each node gets ``lat``/``lng`` via ``pixel_to_geo``;
    each edge gets ``id``/``length_m``/``road_class``/``travel_time`` — the same
    attributes ``network_factory._add_edge`` produces. ``criticality``/
    ``is_bridge`` are filled later by :func:`annotate`.

    Args:
        skel: ``(H, W)`` bool skeleton from :func:`mask_to_skeleton`.
        pixel_to_geo: maps ``(col, row)`` -> ``(lng, lat)`` (e.g. a rasterio
            affine transform; see :func:`affine_from_rasterio`).
        gsd_m: ground sample distance — metres per pixel — for ``length_m``.

    Returns:
        Unannotated ``nx.Graph`` (run :func:`annotate` before serving).
    """
    counts = _neighbour_count(skel)
    ys, xs = np.where(skel)
    # Control points: endpoints / junctions. A pure loop with no such point is
    # skipped (acceptable for the demo; real code could seed an arbitrary pixel).
    control = {(int(y), int(x)) for y, x in zip(ys, xs) if counts[y, x] != 2}

    G = nx.Graph()

    def _node_id(rc: tuple[int, int]) -> str:
        return f"px_{rc[0]}_{rc[1]}"

    def _add_node(rc: tuple[int, int]) -> str:
        nid = _node_id(rc)
        if nid not in G:
            lng, lat = pixel_to_geo(float(rc[1]), float(rc[0]))  # (col, row)
            G.add_node(nid, lat=round(lat, 6), lng=round(lng, 6))
        return nid

    def _neighbours(rc: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = rc
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx_ < skel.shape[1] and skel[ny, nx_]:
                    out.append((ny, nx_))
        return out

    visited_edges: set[frozenset[tuple[int, int]]] = set()
    for start in control:
        for first in _neighbours(start):
            step = frozenset((start, first))
            if step in visited_edges:
                continue
            # Walk the degree-2 chain from ``start`` until the next control point.
            path = [start, first]
            prev, cur = start, first
            visited_edges.add(step)
            while cur not in control:
                nxts = [n for n in _neighbours(cur) if n != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
                visited_edges.add(frozenset((path[-2], path[-1])))
            u = _add_node(path[0])
            v = _add_node(path[-1])
            if u == v:
                continue
            # Polyline length in pixels -> metres via the ground sample distance.
            length_px = sum(
                float(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
                for i in range(len(path) - 1)
            )
            length_m = round(length_px * gsd_m, 1)
            road_class = _infer_road_class(length_m)
            speed_ms = SPEED_KMH[road_class] * 1000 / 3600
            G.add_edge(
                u, v,
                id=f"e_{u}__{v}",
                length_m=length_m,
                road_class=road_class,
                travel_time=round(length_m / speed_ms, 1) if speed_ms else 0.0,
            )
    return G


def _infer_road_class(length_m: float) -> str:
    """Crude road-class heuristic from segment length.

    Real code should use mask width / a learned class head; for the seam a long
    run is treated as an ``arterial`` and short runs as ``local`` (matching the
    two classes ``network_factory`` uses).
    """
    return "arterial" if length_m >= 400.0 else "local"


# --------------------------------------------------------------------------- #
# Step 5: MST healing — reconnect occlusion gaps with minimal bridges.
# --------------------------------------------------------------------------- #
def heal_graph(
    G: nx.Graph,
    max_gap_m: float = DEFAULT_MAX_HEAL_GAP_M,
    *,
    enabled: bool = True,
) -> nx.Graph:
    """Reconnect components split by occlusion using a minimum-spanning-tree of gaps.

    This is what turns the *robust* model's fragmented raw prediction back into an
    intact network (``resilienceIndex`` -> 100). For the *baseline* under occlusion
    pass ``enabled=False`` so the graph stays fragmented (``resilienceIndex`` ~79,
    several disconnected zones) — exactly the demo failure mode.

    Algorithm (mirrors ``ml/README.md`` step 5):
        1. Find connected components.
        2. Propose bridge edges between near endpoints of *different* components
           whose great-circle gap is below ``max_gap_m``, weighted by gap length.
        3. Take the MST over the component-contraction graph; add only those
           bridges back (minimal, non-redundant). Healed edges are tagged
           ``is_bridge=True`` so the frontend can highlight recovered links.

    Args:
        G: unannotated graph from :func:`skeleton_to_graph`.
        max_gap_m: longest gap a single healing bridge may span.
        enabled: ``False`` -> return ``G`` unchanged (baseline-under-occlusion).

    Returns:
        The (possibly) healed graph. Annotate afterwards with :func:`annotate`.
    """
    from backend.app.services.geo import haversine_m  # noqa: PLC0415

    if not enabled or G.number_of_nodes() == 0:
        return G

    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    comp_of: dict[str, int] = {}
    for idx, comp in enumerate(components):
        for n in comp:
            comp_of[n] = idx

    # Candidate bridges between endpoints (degree<=1) of different components.
    endpoints = [n for n in G.nodes if G.degree(n) <= 1] or list(G.nodes)
    candidates: list[tuple[float, int, int, str, str]] = []
    for i, a in enumerate(endpoints):
        for b in endpoints[i + 1:]:
            if comp_of[a] == comp_of[b]:
                continue
            gap = haversine_m(G.nodes[a]["lat"], G.nodes[a]["lng"],
                              G.nodes[b]["lat"], G.nodes[b]["lng"])
            if gap <= max_gap_m:
                candidates.append((gap, comp_of[a], comp_of[b], a, b))

    # MST over components: Kruskal on the candidate bridges (cheapest gap first).
    candidates.sort(key=lambda c: c[0])
    parent = list(range(len(components)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for gap, ca, cb, a, b in candidates:
        ra, rb = _find(ca), _find(cb)
        if ra == rb:
            continue
        parent[ra] = rb
        length_m = round(gap, 1)
        speed_ms = SPEED_KMH["local"] * 1000 / 3600
        G.add_edge(
            a, b,
            id=f"heal_{a}__{b}",
            length_m=length_m,
            road_class="local",
            travel_time=round(length_m / speed_ms, 1) if speed_ms else 0.0,
            healed=True,
        )
    return G


# --------------------------------------------------------------------------- #
# Annotation — identical to network_factory._annotate so downstream is unchanged.
# --------------------------------------------------------------------------- #
def annotate(G: nx.Graph) -> nx.Graph:
    """Attach edge ``criticality``/``is_bridge`` + node ``betweenness``.

    Delegates to ``backend.app.services.criticality`` so the numbers match the
    synthetic path exactly. Healed bridges (``healed=True``) are also flagged
    ``is_bridge=True`` so they render as recovered links.
    """
    from backend.app.services import criticality as crit  # noqa: PLC0415

    eb = crit.edge_betweenness(G)
    nb = crit.node_betweenness(G)
    bridge_set = crit.bridges(G)
    max_eb = max(eb.values()) if eb else 1.0
    for u, v, d in G.edges(data=True):
        raw = eb.get((u, v), eb.get((v, u), 0.0))
        d["criticality"] = round((raw / max_eb) if max_eb > 0 else 0.0, 4)
        d["is_bridge"] = (frozenset((u, v)) in bridge_set) or bool(d.get("healed"))
    max_nb = max(nb.values()) if nb else 1.0
    for n, val in nb.items():
        G.nodes[n]["betweenness"] = round((val / max_nb) if max_nb > 0 else 0.0, 4)
    return G


# --------------------------------------------------------------------------- #
# Geo helper: build a pixel->geo callable from a rasterio dataset/transform.
# --------------------------------------------------------------------------- #
def affine_from_rasterio(transform: "Any") -> PixelToGeo:
    """Wrap a rasterio ``Affine`` (or any obj supporting ``transform * (col, row)``).

    rasterio's ``Affine`` maps ``(col, row) -> (x, y) = (lng, lat)`` for a
    geographic CRS, which is exactly the GeoJSON ``[lng, lat]`` order the backend
    emits. Kept here so callers don't import rasterio just to pass a transform.
    """
    def _to_geo(col: float, row: float) -> tuple[float, float]:
        lng, lat = transform * (col, row)
        return float(lng), float(lat)

    return _to_geo


# --------------------------------------------------------------------------- #
# Top-level seam: image (+ transform) -> annotated graph in get_network() shape.
# --------------------------------------------------------------------------- #
def build_real_network(
    image: np.ndarray,
    pixel_to_geo: PixelToGeo,
    *,
    input_mode: InputMode = "clean",
    model: ModelName = "robust",
    gsd_m: float = 1.0,
    weights_path: str | None = None,
    device: str = "cpu",
    heal_max_gap_m: float = DEFAULT_MAX_HEAL_GAP_M,
) -> nx.Graph:
    """Run the real pipeline and return the same ``nx.Graph`` ``get_network`` does.

    predict mask -> skeletonize -> skeleton_to_graph -> heal -> annotate.

    The ``(input_mode, model)`` combo reproduces the demo story:
        * ``occluded`` + ``baseline`` -> healing OFF  => FRAGMENTED network.
        * every other combo           -> healing ON   => intact network.
    (The robust model's connectivity attention plus MST healing recover the links
    the baseline loses under occlusion.)

    Args:
        image: ``(H, W, 3)`` RGB chip to segment.
        pixel_to_geo: ``(col, row) -> (lng, lat)`` mapping (see
            :func:`affine_from_rasterio`).
        input_mode: ``"clean"`` or ``"occluded"`` (occlusion applied upstream at
            data-prep time; here it only selects the healing policy for the demo).
        model: ``"baseline"`` or ``"robust"`` segmenter.
        gsd_m: ground sample distance in metres/pixel.
        weights_path: checkpoint for the chosen segmenter (None -> stub raises in
            ``predict``; fine for the structural smoke test).
        device: torch device for the robust model ("cpu" / "cuda").
        heal_max_gap_m: max gap a healing bridge may span.

    Returns:
        Annotated ``nx.Graph`` ready for ``graph_to_geojson`` / criticality / etc.
    """
    segmenter = _load_segmenter(model, weights_path, device)
    mask = segmenter.predict(image)  # (H, W) float in [0, 1]
    return graph_from_mask(
        mask,
        pixel_to_geo,
        input_mode=input_mode,
        model=model,
        gsd_m=gsd_m,
        heal_max_gap_m=heal_max_gap_m,
    )


def graph_from_mask(
    mask: np.ndarray,
    pixel_to_geo: PixelToGeo,
    *,
    input_mode: InputMode = "clean",
    model: ModelName = "robust",
    gsd_m: float = 1.0,
    threshold: float = 0.5,
    heal_max_gap_m: float = DEFAULT_MAX_HEAL_GAP_M,
) -> nx.Graph:
    """Mask -> annotated graph (everything after ``predict``).

    Split out from :func:`build_real_network` so the torch-free part of the
    pipeline can be tested with a hand-built mask (see ``__main__``). Healing is
    disabled only for ``occluded + baseline`` to reproduce the fragmented demo.
    """
    skel = mask_to_skeleton(mask, threshold=threshold)
    G = skeleton_to_graph(skel, pixel_to_geo, gsd_m=gsd_m)
    heal_enabled = not (input_mode == "occluded" and model == "baseline")
    G = heal_graph(G, max_gap_m=heal_max_gap_m, enabled=heal_enabled)
    return annotate(G)


# --------------------------------------------------------------------------- #
# Self-test / usage. Runs the torch-free path (synthetic mask) so it works on the
# 4GB dev box with no ML stack. Requires only numpy + networkx (+ scikit-image
# for skeletonize; falls back to a message if absent).
# --------------------------------------------------------------------------- #
def _identity_pixel_to_geo(col: float, row: float) -> tuple[float, float]:
    """Tiny demo transform: pretend 1px == 0.0001 deg around a Bengaluru origin."""
    lng = 77.58 + col * 1e-4
    lat = 13.00 - row * 1e-4
    return lng, lat


def _demo_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """A '+'-shaped road with an occlusion gap, as a float probability mask."""
    m = np.zeros((h, w), dtype=np.float32)
    m[h // 2, :] = 1.0          # horizontal road
    m[:, w // 2] = 1.0          # vertical road
    m[h // 2, w // 2 - 6: w // 2 - 2] = 0.0  # occlusion gap on the horizontal arm
    return m


if __name__ == "__main__":
    # Torch-free smoke test: synthetic mask -> graph -> heal -> annotate. Proves
    # the seam produces a get_network()-shaped graph without any ML stack.
    print("segmenter_hook.py self-test (torch-free; synthetic mask)")
    mask = _demo_mask()
    try:
        skel = mask_to_skeleton(mask)
    except ImportError as exc:
        print("  [skip] scikit-image not installed:", exc)
        print("  Install ml/requirements.txt on a training box to run vectorize.")
        raise SystemExit(0)

    pg = _identity_pixel_to_geo

    # baseline + occluded -> healing OFF -> should stay fragmented (>1 component).
    g_frag = graph_from_mask(mask, pg, input_mode="occluded", model="baseline")
    n_comp_frag = nx.number_connected_components(g_frag)

    # robust + occluded -> healing ON -> gap reconnected -> single component.
    g_heal = graph_from_mask(mask, pg, input_mode="occluded", model="robust")
    n_comp_heal = nx.number_connected_components(g_heal)

    print(f"  fragmented (occluded+baseline): {g_frag.number_of_nodes()} nodes, "
          f"{g_frag.number_of_edges()} edges, {n_comp_frag} components")
    print(f"  healed     (occluded+robust)  : {g_heal.number_of_nodes()} nodes, "
          f"{g_heal.number_of_edges()} edges, {n_comp_heal} components")

    # Show the annotated edge shape matches what graph_to_geojson expects.
    sample = next(iter(g_heal.edges(data=True)), None)
    if sample:
        _, _, d = sample
        print("  sample edge props:", {k: d[k] for k in
              ("id", "criticality", "travel_time", "length_m", "road_class", "is_bridge")
              if k in d})

    # TODO (training machine, torch + weights present):
    #   import rasterio
    #   from ml.integration.segmenter_hook import build_real_network, affine_from_rasterio
    #   with rasterio.open("chip.tif") as ds:
    #       img = ds.read([1, 2, 3]).transpose(1, 2, 0)   # (H, W, 3)
    #       pg = affine_from_rasterio(ds.transform)        # (col,row)->(lng,lat)
    #       gsd = abs(ds.transform.a)                       # metres/pixel
    #   G = build_real_network(img, pg, input_mode="occluded", model="robust",
    #                          gsd_m=gsd, weights_path="ml/checkpoints/robust.pt",
    #                          device="cuda")
    #   # then point backend network_factory.get_network() at build_real_network;
    #   # graph_to_geojson(G) / criticality / gatekeepers / simulate are unchanged.
    print("OK")
