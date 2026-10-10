"""JSON round trips and schema handling."""

from __future__ import annotations

import json

import pytest
from ir_fixtures import ALL, residual_cnn

from tensorgami.ir import SCHEMA_VERSION, GraphError, from_dict, from_json, to_dict, to_json


@pytest.mark.parametrize("make", ALL.values(), ids=ALL.keys())
def test_round_trip_is_lossless(make):
    graph = make()
    assert from_json(to_json(graph)) == graph


def test_output_is_deterministic_and_versioned():
    text = to_json(residual_cnn())
    assert text == to_json(residual_cnn())
    data = json.loads(text)
    assert data["schema"] == "tensorgami.ir" and data["version"] == SCHEMA_VERSION
    conv = data["nodes"][1]
    assert conv["params"]["kernel_size"] == [3, 3]  # tuples become JSON lists
    assert from_dict(data).nodes[1].params["kernel_size"] == (3, 3)  # and back


def test_wrong_schema_or_version_is_rejected():
    data = to_dict(residual_cnn())
    with pytest.raises(GraphError, match="not a Tensorgami graph"):
        from_dict({**data, "schema": "other"})
    with pytest.raises(GraphError, match="unsupported graph version 99"):
        from_dict({**data, "version": 99})


def test_malformed_data_reports_the_problem():
    data = to_dict(residual_cnn())
    del data["nodes"][0]["op"]
    with pytest.raises(GraphError, match="malformed graph data: KeyError"):
        from_dict(data)
    data = to_dict(residual_cnn())
    data["nodes"][1]["category"] = "quantum"
    with pytest.raises(GraphError, match="malformed"):
        from_dict(data)
    with pytest.raises(GraphError, match="invalid JSON"):
        from_json("{not json")


def test_invalid_graph_is_rejected_on_load_unless_disabled():
    data = to_dict(residual_cnn())
    data["edges"].append({"source": "n1", "target": "ghost"})
    with pytest.raises(GraphError, match="ghost"):
        from_dict(data)
    assert len(from_dict(data, validate=False).edges) == len(data["edges"])


def test_bad_shape_values_are_rejected():
    data = to_dict(residual_cnn())
    data["nodes"][1]["outputs"] = [["a", 2]]
    with pytest.raises(GraphError, match="invalid shape"):
        from_dict(data)
