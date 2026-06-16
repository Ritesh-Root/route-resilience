"""Tests for GET /api/network — the GeoJSON road graph the front-end consumes.

Run from the backend/ directory:  cd backend && python -m pytest

The `client` fixture comes from tests/conftest.py (a FastAPI TestClient).
"""
import pytest

# Bengaluru is the only city; exercise every (input, model) view of the network.
VIEWS = [
    ("clean", "robust"),
    ("clean", "baseline"),
    ("occluded", "robust"),
    ("occluded", "baseline"),
]

# Required edge properties and the Python types the front-end expects.
REQUIRED_EDGE_PROPS = {
    "id": (str, int),
    "criticality": (int, float),
    "travelTimeSec": (int, float),
    "lengthM": (int, float),
    "roadClass": str,
    "isBridge": bool,
}


def _get_network(client, input_mode, model):
    resp = client.get(
        "/api/network",
        params={"city": "Bengaluru", "input": input_mode, "model": model},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_is_feature_collection(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    assert fc["type"] == "FeatureCollection"
    assert isinstance(fc["features"], list)
    assert fc["features"], "expected at least one edge feature"


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_edge_properties_present_and_typed(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    for feat in fc["features"]:
        props = feat["properties"]
        for name, expected_type in REQUIRED_EDGE_PROPS.items():
            assert name in props, f"missing edge property {name!r}"
            # bool is a subclass of int, so guard isBridge explicitly.
            if name == "isBridge":
                assert isinstance(props[name], bool)
            else:
                assert isinstance(props[name], expected_type), (
                    f"{name}={props[name]!r} has wrong type"
                )


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_criticality_in_unit_range(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    for feat in fc["features"]:
        c = feat["properties"]["criticality"]
        assert 0.0 <= c <= 1.0, f"criticality out of [0,1]: {c}"


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_geometry_is_lng_lat_linestring(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    for feat in fc["features"]:
        geom = feat["geometry"]
        assert geom["type"] == "LineString"
        coords = geom["coordinates"]
        assert len(coords) >= 2, "LineString needs >= 2 positions"
        for lng, lat in coords:
            # Coords are [lng, lat]: longitude spans wider than latitude.
            assert -180.0 <= lng <= 180.0, f"lng out of range: {lng}"
            assert -90.0 <= lat <= 90.0, f"lat out of range: {lat}"
        # Bengaluru sits near lat ~12.97, lng ~77.59 — assert the ordering
        # is [lng, lat] (not the GeoJSON-violating [lat, lng]) on the first edge.
        first_lng, first_lat = coords[0]
        assert first_lng > first_lat, (
            "coords look swapped; expected [lng, lat] with lng > lat for Bengaluru"
        )


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_meta_counts_match_features(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    meta = fc["meta"]
    assert meta["city"] == "Bengaluru"
    assert meta["input"] == input_mode
    assert meta["model"] == model
    assert isinstance(meta["edges"], int) and meta["edges"] > 0
    assert isinstance(meta["nodes"], int) and meta["nodes"] > 0
    # meta.edges must equal the number of edge features actually returned.
    assert meta["edges"] == len(fc["features"])


@pytest.mark.parametrize("input_mode,model", VIEWS)
def test_edge_ids_unique(client, input_mode, model):
    fc = _get_network(client, input_mode, model)
    ids = [feat["properties"]["id"] for feat in fc["features"]]
    assert len(ids) == len(set(ids)), "duplicate edge ids"
