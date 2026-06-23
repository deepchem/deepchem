"""Hall of Fame for symbolic regression.

Tracks the best expression found at each complexity level,
forming a Pareto front of accuracy vs simplicity.
Also serves as the migration source for the island model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from deepchem.models.symbolic_regression.core.node import Node


@dataclass
class HoFEntry:
    """A single Hall of Fame entry.

    Attributes
    ----------
    node : Node
        The expression tree (deep-copied when stored).
    mse : float
        Raw mean squared error (no parsimony penalty).
    fitness : float
        Fitness score (MSE + parsimony penalty).
    complexity : int
        Weighted complexity used as Hall-of-Fame key.
    """
    node: Node
    mse: float
    fitness: float
    complexity: int = -1


class HallOfFame:
    """Stores the best expression at each complexity level.

    Keyed by weighted complexity. A new expression replaces
    the current entry at that complexity only if it has lower MSE.
    """

    def __init__(self):
        self.entries: Dict[int, HoFEntry] = {}

    def update(self,
               node: Node,
               mse: float,
               fitness: float,
               complexity: Optional[int] = None) -> bool:
        """Update the Hall of Fame with a new candidate.

        Parameters
        ----------
        node : Node
            Expression tree to consider.
        mse : float
            Raw MSE of the expression.
        fitness : float
            Fitness score (MSE + parsimony).
        complexity : int, optional
            Weighted complexity to use as key. If None, falls back
            to node.size() (raw node count).

        Returns
        -------
        bool
            True if the entry was added/updated, False otherwise.
        """
        if complexity is None:
            complexity = node.size()
        if complexity not in self.entries or mse < self.entries[complexity].mse:
            self.entries[complexity] = HoFEntry(
                node=node.copy(),
                mse=mse,
                fitness=fitness,
                complexity=complexity,
            )
            return True
        return False

    def get_best(self) -> Optional[HoFEntry]:
        """Return the entry with the lowest MSE across all complexities."""
        if not self.entries:
            return None
        return min(self.entries.values(), key=lambda e: e.mse)

    def get_pareto_front(self) -> List[Tuple[int, HoFEntry]]:
        """Return all entries sorted by complexity (ascending)."""
        return sorted(self.entries.items())

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        if not self.entries:
            return "HallOfFame(empty)"
        lines = ["HallOfFame:"]
        for complexity in sorted(self.entries):
            e = self.entries[complexity]
            lines.append(f"  cplx={complexity:2d} | MSE={e.mse:.6f} | {e.node}")
        return "\n".join(lines)
