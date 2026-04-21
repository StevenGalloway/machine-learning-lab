"""
Tests for notebooks/jupyter/weather-prediction/weather_markov.py
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from tests.conftest import load_module

_MOD = load_module(
    "weather_markov",
    "notebooks/jupyter/weather-prediction/weather_markov.py",
)

Config = _MOD.Config
validate_transition_probabilities = _MOD.validate_transition_probabilities
build_transition_matrix = _MOD.build_transition_matrix
compute_stationary_distribution = _MOD.compute_stationary_distribution
next_state = _MOD.next_state
simulate_markov_chain = _MOD.simulate_markov_chain
summarize_history = _MOD.summarize_history


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

STATES = ["Sunny", "Rainy", "Cloudy"]
TRANSITION_PROBS: Dict[str, List[float]] = {
    "Sunny":  [0.8, 0.1, 0.1],
    "Rainy":  [0.2, 0.6, 0.2],
    "Cloudy": [0.3, 0.3, 0.4],
}


# ---------------------------------------------------------------------------
# validate_transition_probabilities
# ---------------------------------------------------------------------------

class TestValidateTransitionProbabilities:
    def test_valid_probs_pass(self):
        validate_transition_probabilities(STATES, TRANSITION_PROBS)  # must not raise

    def test_raises_on_missing_state(self):
        incomplete = {k: v for k, v in TRANSITION_PROBS.items() if k != "Rainy"}
        with pytest.raises(ValueError, match="Missing transition probabilities"):
            validate_transition_probabilities(STATES, incomplete)

    def test_raises_on_wrong_number_of_probabilities(self):
        bad = {**TRANSITION_PROBS, "Sunny": [0.5, 0.5]}  # only 2 probs for 3 states
        with pytest.raises(ValueError, match="probabilities but expected"):
            validate_transition_probabilities(STATES, bad)

    def test_raises_on_negative_probability(self):
        bad = {**TRANSITION_PROBS, "Sunny": [0.9, -0.1, 0.2]}
        with pytest.raises(ValueError, match="negative probability"):
            validate_transition_probabilities(STATES, bad)

    def test_raises_when_probs_do_not_sum_to_one(self):
        bad = {**TRANSITION_PROBS, "Sunny": [0.5, 0.4, 0.0]}  # sums to 0.9
        with pytest.raises(ValueError, match="sum to"):
            validate_transition_probabilities(STATES, bad)

    def test_passes_with_near_one_sum(self):
        # floating-point sums like 0.9999999 should be accepted
        near_one = {**TRANSITION_PROBS, "Sunny": [0.8, 0.1, 0.1000001]}
        validate_transition_probabilities(STATES, near_one)  # must not raise


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

class TestBuildTransitionMatrix:
    def test_shape_is_n_by_n(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        assert tm.shape == (3, 3)

    def test_rows_sum_to_one(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        row_sums = tm.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

    def test_values_match_transition_probs(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        for i, state in enumerate(STATES):
            np.testing.assert_array_equal(tm[i], TRANSITION_PROBS[state])

    def test_dtype_is_float(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        assert tm.dtype == float


# ---------------------------------------------------------------------------
# compute_stationary_distribution
# ---------------------------------------------------------------------------

class TestComputeStationaryDistribution:
    def test_sums_to_one(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        sd = compute_stationary_distribution(tm)
        assert sd.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_non_negative(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        sd = compute_stationary_distribution(tm)
        assert (sd >= 0).all()

    def test_length_equals_num_states(self):
        tm = build_transition_matrix(STATES, TRANSITION_PROBS)
        sd = compute_stationary_distribution(tm)
        assert len(sd) == len(STATES)

    def test_identity_matrix_gives_uniform(self):
        # Identity transition matrix → any starting distribution is stationary
        n = 3
        # Uniform row-stochastic → stationary = uniform
        tm = np.ones((n, n), dtype=float) / n
        sd = compute_stationary_distribution(tm)
        np.testing.assert_allclose(sd, np.ones(n) / n, atol=1e-6)

    def test_absorbing_state(self):
        # If state 0 is absorbing (all probability stays there), stationary = [1, 0, 0]
        tm = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.3, 0.2],
            [0.4, 0.2, 0.4],
        ])
        sd = compute_stationary_distribution(tm)
        assert sd[0] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# next_state
# ---------------------------------------------------------------------------

class TestNextState:
    def test_returns_valid_state(self):
        for _ in range(50):
            s = next_state("Sunny", STATES, TRANSITION_PROBS)
            assert s in STATES

    def test_absorbing_state_stays_put(self):
        absorbing_probs = {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
        }
        # State A is absorbing → always returns A
        for _ in range(20):
            assert next_state("A", ["A", "B"], absorbing_probs) == "A"


# ---------------------------------------------------------------------------
# simulate_markov_chain
# ---------------------------------------------------------------------------

class TestSimulateMarkovChain:
    def _config(self, start="Sunny", steps=30, seed=42):
        return Config(
            states=STATES,
            transition_probs=TRANSITION_PROBS,
            start_state=start,
            num_steps=steps,
            seed=seed,
        )

    def test_history_length_is_steps_plus_one(self):
        history = simulate_markov_chain(self._config(steps=30))
        assert len(history) == 31  # start + 30 transitions

    def test_first_element_is_start_state(self):
        history = simulate_markov_chain(self._config(start="Rainy"))
        assert history[0] == "Rainy"

    def test_all_states_are_valid(self):
        history = simulate_markov_chain(self._config(steps=100))
        assert all(s in STATES for s in history)

    def test_reproducible_with_same_seed(self):
        h1 = simulate_markov_chain(self._config(seed=7))
        h2 = simulate_markov_chain(self._config(seed=7))
        assert h1 == h2

    def test_different_seeds_can_differ(self):
        h1 = simulate_markov_chain(self._config(seed=1, steps=50))
        h2 = simulate_markov_chain(self._config(seed=2, steps=50))
        # Very unlikely to be identical with different seeds over 50 steps
        assert h1 != h2


# ---------------------------------------------------------------------------
# summarize_history
# ---------------------------------------------------------------------------

class TestSummarizeHistory:
    def test_counts_correct(self):
        history = ["Sunny", "Rainy", "Sunny", "Cloudy", "Sunny"]
        counts = summarize_history(history, STATES)
        assert counts["Sunny"] == 3
        assert counts["Rainy"] == 1
        assert counts["Cloudy"] == 1

    def test_total_matches_history_length(self):
        history = simulate_markov_chain(Config(
            states=STATES,
            transition_probs=TRANSITION_PROBS,
            num_steps=50,
        ))
        counts = summarize_history(history, STATES)
        assert sum(counts.values()) == len(history)

    def test_all_states_represented_in_output(self):
        history = ["Sunny"] * 5
        counts = summarize_history(history, STATES)
        for s in STATES:
            assert s in counts

    def test_zero_count_for_unvisited_state(self):
        history = ["Sunny"] * 10
        counts = summarize_history(history, STATES)
        assert counts["Rainy"] == 0
        assert counts["Cloudy"] == 0
