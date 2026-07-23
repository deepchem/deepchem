"""Tournament selection for symbolic regression.

Selects individuals from a population by running small tournaments
and picking the winner (lowest fitness = best).
"""

import random
from typing import List, Optional

from deepchem.models.symbolic_regression.core.node import Node


def tournament_select(population: List[Node],
                      fitnesses: List[float],
                      tournament_size: int = 7,
                      pick_prob: float = 1.0,
                      rng: Optional[random.Random] = None) -> Node:
    """Select the best individual from a random tournament.

    Picks ``tournament_size`` random individuals from the population
    and returns the one with the lowest fitness (best).

    Parameters
    ----------
    population : list[Node]
        List of expression trees.
    fitnesses : list[float]
        Corresponding fitness scores (lower is better).
    tournament_size : int
        Number of individuals in each tournament. Default 7.
    pick_prob : float
        Probability of selecting the best member in the sampled tournament.
        If < 1.0, lower-ranked members are sampled geometrically to preserve
        diversity (similar to PySR's tournament_selection_p).

    Returns
    -------
    Node
        The tournament winner (not copied — caller should copy if needed).
    """
    if tournament_size < 1:
        raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")
    if len(population) == 0:
        raise ValueError("population must be non-empty")
    if len(population) != len(fitnesses):
        raise ValueError(
            f"population and fitnesses must have the same length, got "
            f"{len(population)} and {len(fitnesses)}")
    if not (0.0 < pick_prob <= 1.0):
        raise ValueError(f"pick_prob must be in (0, 1], got {pick_prob}")

    rand = rng if rng is not None else random

    size = min(tournament_size, len(population))
    indices = rand.sample(range(len(population)), size)
    ranked = sorted(indices, key=lambda i: fitnesses[i])

    if pick_prob >= 1.0:
        return population[ranked[0]]

    for idx in ranked:
        if rand.random() < pick_prob:
            return population[idx]
    return population[ranked[-1]]
