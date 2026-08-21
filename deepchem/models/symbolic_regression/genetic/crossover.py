"""Subtree crossover for symbolic regression.

Picks a random subtree from parent A, replaces it with a
size-compatible subtree from parent B.  Uses size-fair selection
to control bloat (Langdon & Poli, 2002).
"""

import random
from typing import List, Optional, Tuple

from deepchem.models.symbolic_regression.core.node import (
    Node,
    UnaryOpNode,
    BinaryOpNode,
)


def _node_positions(
    node: Node,
    parent: Optional[Node] = None,
    attr: Optional[str] = None,
) -> List[Tuple[Node, Optional[Node], Optional[str]]]:
    """Return (node, parent, attr_name) for every node in the tree.

    Used on the child tree so we know *where* to splice in the donor.
    """
    result = [(node, parent, attr)]
    if isinstance(node, UnaryOpNode):
        result += _node_positions(node.child, node, "child")
    elif isinstance(node, BinaryOpNode):
        result += _node_positions(node.left, node, "left")
        result += _node_positions(node.right, node, "right")
    return result


def _all_nodes(node: Node) -> List[Node]:
    """Return every node in the tree (used to gather donor candidates)."""
    result = [node]
    if isinstance(node, UnaryOpNode):
        result += _all_nodes(node.child)
    elif isinstance(node, BinaryOpNode):
        result += _all_nodes(node.left)
        result += _all_nodes(node.right)
    return result


def _pick_similar_size(
    target_size: int,
    candidates: List[Node],
    rng: random.Random,
) -> Optional[Node]:
    """Pick a donor subtree of similar size to the target.

    Prefers donors within 2x the target size. Falls back to
    uniform random if no size-compatible donors exist.
    """
    if not candidates:
        return None

    min_size = max(1, target_size // 2)
    max_size = max(1, target_size * 2)
    compatible = [n for n in candidates if min_size <= n.size() <= max_size]

    pool = compatible if compatible else candidates
    return rng.choice(pool)


def crossover(
    parent_a: Node,
    parent_b: Node,
    rng: Optional[random.Random] = None,
) -> Node:
    """Subtree crossover between two parent trees.

    Deep-copies parent_a, picks a random subtree in it, and replaces
    it with a size-compatible subtree copied from parent_b.

    Parameters
    ----------
    parent_a : Node
        First parent (provides the base structure).
    parent_b : Node
        Second parent (donates a subtree).

    Returns
    -------
    Node
        A new child tree (parents are not modified).
    """
    rand = rng if rng is not None else random

    child = parent_a.copy()

    child_positions = _node_positions(child)
    if not child_positions:
        return child

    target, parent, attr = rand.choice(child_positions)

    donor_nodes = _all_nodes(parent_b)
    picked = _pick_similar_size(target.size(), donor_nodes, rand)
    if picked is None:
        return child

    # Copy the selected subtree so we don't share nodes with parent_b
    replacement = picked.copy()

    if parent is None:
        return replacement
    setattr(parent, attr, replacement)
    return child
