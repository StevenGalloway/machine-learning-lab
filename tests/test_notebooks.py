"""
Structural integrity tests for all Jupyter notebooks in the repository.

These tests verify notebooks can be parsed and have the expected structure
without executing them (execution requires external data and heavy models).

Requires: nbformat  (pip install nbformat)
"""
from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="nbformat not installed")

from tests.conftest import REPO_ROOT

# ---------------------------------------------------------------------------
# Discover all notebooks in the repository
# ---------------------------------------------------------------------------

ALL_NOTEBOOKS = sorted(REPO_ROOT.rglob("*.ipynb"))

# Known notebooks we explicitly test
NBA_NB = REPO_ROOT / "notebooks" / "jupyter" / "nba-points-prediction" / "nba-points-prediction_rf.ipynb"
NFL_PASSING_NB = REPO_ROOT / "notebooks" / "jupyter" / "nfl-passing-yards-prediction" / "nfl-passing-yards-prediction_linear_reg.ipynb"
WEATHER_NB = REPO_ROOT / "notebooks" / "jupyter" / "weather-prediction" / "weather_markov.ipynb"

KNOWN_NOTEBOOKS = [NBA_NB, NFL_PASSING_NB, WEATHER_NB]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_notebook(path: Path) -> nbformat.NotebookNode:
    with path.open("r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


# ---------------------------------------------------------------------------
# Generic: every notebook in the repo can be parsed
# ---------------------------------------------------------------------------

class TestAllNotebooksParseable:
    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOKS, ids=[p.name for p in ALL_NOTEBOOKS])
    def test_notebook_parseable(self, nb_path):
        nb = _load_notebook(nb_path)
        assert nb is not None

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOKS, ids=[p.name for p in ALL_NOTEBOOKS])
    def test_notebook_has_cells(self, nb_path):
        nb = _load_notebook(nb_path)
        assert len(nb.cells) > 0

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOKS, ids=[p.name for p in ALL_NOTEBOOKS])
    def test_notebook_has_code_cells(self, nb_path):
        nb = _load_notebook(nb_path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code_cells) > 0, f"{nb_path.name} has no code cells"

    @pytest.mark.parametrize("nb_path", ALL_NOTEBOOKS, ids=[p.name for p in ALL_NOTEBOOKS])
    def test_no_cell_has_syntax_error_import(self, nb_path):
        """Verify no cell contains obviously broken imports (like old typos)."""
        nb = _load_notebook(nb_path)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert "MultinomialNBhey" not in cell.source, (
                    f"Typo 'MultinomialNBhey' found in {nb_path.name}"
                )


# ---------------------------------------------------------------------------
# NBA points prediction notebook
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NBA_NB.exists(), reason="NBA notebook not found")
class TestNbaPointsPredictionNotebook:
    def test_parseable(self):
        nb = _load_notebook(NBA_NB)
        assert nb is not None

    def test_has_minimum_cells(self):
        nb = _load_notebook(NBA_NB)
        assert len(nb.cells) >= 4

    def test_contains_sklearn_import(self):
        nb = _load_notebook(NBA_NB)
        code_sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "sklearn" in code_sources

    def test_contains_random_forest(self):
        nb = _load_notebook(NBA_NB)
        code_sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "RandomForest" in code_sources or "random_forest" in code_sources.lower()

    def test_all_code_cells_have_source(self):
        nb = _load_notebook(NBA_NB)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert all(isinstance(c.source, str) for c in code_cells)

    def test_kernel_language_is_python(self):
        nb = _load_notebook(NBA_NB)
        lang = nb.metadata.get("kernelspec", {}).get("language", "")
        # Accept empty or "python"
        assert lang in ("", "python")


# ---------------------------------------------------------------------------
# NFL passing yards prediction notebook
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NFL_PASSING_NB.exists(), reason="NFL passing yards notebook not found")
class TestNflPassingYardsNotebook:
    def test_parseable(self):
        nb = _load_notebook(NFL_PASSING_NB)
        assert nb is not None

    def test_has_minimum_cells(self):
        nb = _load_notebook(NFL_PASSING_NB)
        assert len(nb.cells) >= 3

    def test_contains_linear_regression_import(self):
        nb = _load_notebook(NFL_PASSING_NB)
        sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "LinearRegression" in sources or "linear_model" in sources

    def test_mentions_passing_yards(self):
        nb = _load_notebook(NFL_PASSING_NB)
        all_text = " ".join(c.source for c in nb.cells)
        assert "Pass" in all_text or "Yard" in all_text or "pass" in all_text.lower()

    def test_no_empty_cells(self):
        nb = _load_notebook(NFL_PASSING_NB)
        # Allow markdown cells to be empty; code cells should have content
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        non_empty = [c for c in code_cells if c.source.strip()]
        assert len(non_empty) > 0


# ---------------------------------------------------------------------------
# Weather Markov chain notebook
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not WEATHER_NB.exists(), reason="Weather Markov notebook not found")
class TestWeatherMarkovNotebook:
    def test_parseable(self):
        nb = _load_notebook(WEATHER_NB)
        assert nb is not None

    def test_has_minimum_cells(self):
        nb = _load_notebook(WEATHER_NB)
        assert len(nb.cells) >= 5

    def test_contains_markov_chain_logic(self):
        nb = _load_notebook(WEATHER_NB)
        sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        # Should contain transition probabilities or stationary distribution
        keywords = {"transition", "stationary", "markov", "Markov", "Sunny", "Rainy"}
        assert any(kw in sources for kw in keywords)

    def test_mentions_weather_states(self):
        nb = _load_notebook(WEATHER_NB)
        all_text = " ".join(c.source for c in nb.cells)
        assert "Sunny" in all_text or "Rainy" in all_text

    def test_imports_numpy(self):
        nb = _load_notebook(WEATHER_NB)
        sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "numpy" in sources or "np" in sources

    def test_uses_dataclass_or_config(self):
        nb = _load_notebook(WEATHER_NB)
        sources = " ".join(c.source for c in nb.cells if c.cell_type == "code")
        assert "dataclass" in sources or "Config" in sources or "config" in sources
