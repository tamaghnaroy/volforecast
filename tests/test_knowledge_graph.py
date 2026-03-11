"""
Tests for the volatility model knowledge graph.
"""

import pytest

from volforecast.knowledge.graph import VolatilityKnowledgeGraph
from volforecast.core.targets import VolatilityTarget


@pytest.fixture(scope="module")
def kg():
    return VolatilityKnowledgeGraph()


class TestKnowledgeGraph:
    def test_builds_without_error(self, kg):
        assert kg.G is not None
        assert kg.G.number_of_nodes() > 0

    def test_summary(self, kg):
        s = kg.summary()
        assert s["n_models"] >= 25
        assert s["n_families"] >= 5
        assert "GARCH" in s["families"]
        assert "HAR" in s["families"]
        assert "SV" in s["families"]
        assert "COMBO" in s["families"]
        assert "REALIZED_MEASURE" in s["families"]

    def test_get_garch_family(self, kg):
        garch_models = kg.get_family("GARCH")
        assert len(garch_models) >= 5
        names = [m["name"] for m in garch_models]
        assert "GARCH(1,1)" in names
        assert "Realized GARCH" in names

    def test_get_har_family(self, kg):
        har_models = kg.get_family("HAR")
        assert len(har_models) >= 4

    def test_get_models_for_target(self, kg):
        cond_var_models = kg.get_models_for_target(VolatilityTarget.CONDITIONAL_VARIANCE)
        assert len(cond_var_models) > 0
        # GARCH should target conditional variance
        assert any("GARCH" in m for m in cond_var_models)

    def test_get_models_for_iv(self, kg):
        iv_models = kg.get_models_for_target(VolatilityTarget.INTEGRATED_VARIANCE)
        assert len(iv_models) > 0
        assert any("HAR" in m for m in iv_models)

    def test_ancestors(self, kg):
        ancestors = kg.get_ancestors("GJR")
        assert any("GARCH" in a for a in ancestors)

    def test_to_dict(self, kg):
        d = kg.to_dict()
        assert "nodes" in d or "directed" in d
        assert isinstance(d, dict)

    def test_all_models_have_references(self, kg):
        for node_id, data in kg.G.nodes(data=True):
            if data.get("kind") == "model":
                assert data.get("reference"), f"Model {node_id} missing reference"

    def test_all_models_have_targets_edge(self, kg):
        for node_id, data in kg.G.nodes(data=True):
            if data.get("kind") == "model":
                targets_edges = [
                    e for e in kg.G.out_edges(node_id, data=True)
                    if e[2].get("relation") == "targets"
                ]
                assert len(targets_edges) >= 1, f"Model {node_id} missing targets edge"
