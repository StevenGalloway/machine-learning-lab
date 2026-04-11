from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json
import logging
import random

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Config:
    states: List[str]
    transition_probs: Dict[str, List[float]]
    start_state: str = "Sunny"
    num_steps: int = 30
    seed: int = 42
    output_dir: Path = Path("results")
    chart_file: str = "weather_state_frequency.png"
    metrics_file: str = "weather_metrics.json"


def validate_transition_probabilities(states: List[str], transition_probs: Dict[str, List[float]]) -> None:
    missing = [s for s in states if s not in transition_probs]
    if missing:
        raise ValueError(f"Missing transition probabilities for states: {missing}")

    for state, probs in transition_probs.items():
        if len(probs) != len(states):
            raise ValueError(
                f"State '{state}' has {len(probs)} probabilities but expected {len(states)}."
            )
        if any(p < 0 for p in probs):
            raise ValueError(f"State '{state}' contains a negative probability.")
        if not np.isclose(sum(probs), 1.0):
            raise ValueError(
                f"Probabilities for state '{state}' sum to {sum(probs):.6f}, not 1.0."
            )


def build_transition_matrix(states: List[str], transition_probs: Dict[str, List[float]]) -> np.ndarray:
    return np.array([transition_probs[state] for state in states], dtype=float)


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    stationary = np.maximum(stationary, 0)
    stationary = stationary / stationary.sum()
    return stationary


def next_state(current_state: str, states: List[str], transition_probs: Dict[str, List[float]]) -> str:
    probs = transition_probs[current_state]
    return random.choices(states, weights=probs, k=1)[0]


def simulate_markov_chain(config: Config) -> List[str]:
    random.seed(config.seed)
    current = config.start_state
    history = [current]

    for _ in range(config.num_steps):
        current = next_state(current, config.states, config.transition_probs)
        history.append(current)

    return history


def summarize_history(history: List[str], states: List[str]) -> Dict[str, int]:
    return {state: history.count(state) for state in states}


def main() -> None:
    config = Config(
        states=["Sunny", "Rainy", "Cloudy"],
        transition_probs={
            "Sunny": [0.8, 0.1, 0.1],
            "Rainy": [0.2, 0.6, 0.2],
            "Cloudy": [0.3, 0.3, 0.4],
        },
    )

    validate_transition_probabilities(config.states, config.transition_probs)
    transition_matrix = build_transition_matrix(config.states, config.transition_probs)
    stationary_distribution = compute_stationary_distribution(transition_matrix)
    history = simulate_markov_chain(config)
    counts = summarize_history(history, config.states)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.bar(config.states, [counts[s] for s in config.states])
    plt.xlabel("Weather State")
    plt.ylabel("Frequency")
    plt.title("Weather State Frequency from Markov Simulation")
    plt.tight_layout()
    plt.savefig(config.output_dir / config.chart_file, dpi=150)
    plt.show()

    metrics = {
        "start_state": config.start_state,
        "num_steps": config.num_steps,
        "seed": config.seed,
        "history": history,
        "state_counts": counts,
        "stationary_distribution": {
            state: float(prob) for state, prob in zip(config.states, stationary_distribution)
        },
        "transition_matrix": transition_matrix.tolist(),
    }

    with open(config.output_dir / config.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to: %s", config.output_dir / config.metrics_file)
    logger.info("Saved chart to: %s", config.output_dir / config.chart_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    main()
