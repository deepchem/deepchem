"""Mutation operators for symbolic regression.

Implements mutation types with configurable probability weights:
  mutate_operator    — swap operator for another of same arity
  add_node           — wrap a subtree with a new operator (grows tree)
  delete_node        — remove an operator, keep one child (shrinks tree)
  mutate_feature     — change which variable a leaf references
  swap_operands      — swap left/right of a non-commutative binary op
  append_argument    — append a leaf inside an operator's subtree
  optimize_constants — run constant optimization (BFGS)
  randomize_tree     — replace expression with a fresh random tree

Also provides generate_random_tree() for initial population creation.
"""

import random
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from deepchem.models.symbolic_regression.core.node import (
    Node,
    ConstantNode,
    VariableNode,
    UnaryOpNode,
    BinaryOpNode,
)
from deepchem.models.symbolic_regression.core.operators import Operator
from deepchem.models.symbolic_regression.core.operators import COMMUTATIVE_ASSOCIATIVE_BINARY_OPS

# ---------------------------------------------------------------------------
# Tree traversal helpers
# ---------------------------------------------------------------------------


def _get_nodes_with_parent(node: Node,
                           parent: Optional[Node] = None,
                           attr: Optional[str] = None
                          ) -> List[Tuple[Node, Optional[Node], Optional[str]]]:
    """Collect all (node, parent, attr_name) triples from a tree."""
    result = [(node, parent, attr)]
    if isinstance(node, UnaryOpNode):
        result += _get_nodes_with_parent(node.child, node, 'child')
    elif isinstance(node, BinaryOpNode):
        result += _get_nodes_with_parent(node.left, node, 'left')
        result += _get_nodes_with_parent(node.right, node, 'right')
    return result


def _replace_node(root: Node, parent: Optional[Node], attr: Optional[str],
                  replacement: Node) -> Node:
    """Replace a node in the tree. Returns new root."""
    if parent is None:
        return replacement
    setattr(parent, attr, replacement)
    return root


def _random_leaf(n_features: int, rng: Optional[random.Random] = None) -> Node:
    """Generate a random leaf node (constant or variable)."""
    rand = rng if rng is not None else random

    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    if rand.random() < 0.5:
        return ConstantNode(rand.gauss(0, 1))
    else:
        return VariableNode(rand.randint(0, n_features - 1))


# ---------------------------------------------------------------------------
# Random tree generation (used by mutations and evolution)
# ---------------------------------------------------------------------------


def generate_random_tree(unary_ops: Dict[str, Operator],
                         binary_ops: Dict[str, Operator],
                         n_features: int,
                         max_depth: int = 3,
                         method: str = "grow",
                         rng: Optional[random.Random] = None) -> Node:
    """Generate a random expression tree.

    Supports:
    - "grow": probabilistically create leaf/operator at each level.
    - "full": force operators until max_depth, then leaves.

    Parameters
    ----------
    unary_ops : dict[str, Operator]
        Available unary operators.
    binary_ops : dict[str, Operator]
        Available binary operators.
    n_features : int
        Number of input features (for VariableNode indices).
    max_depth : int
        Maximum tree depth. Default 3.
    method : {"grow", "full"}
        Tree initialization strategy.

    Returns
    -------
    Node
        A random expression tree.
    """
    rand = rng if rng is not None else random
    if method not in ("grow", "full"):
        raise ValueError(f"method must be 'grow' or 'full', got {method!r}")

    # At max depth always create a leaf.
    if max_depth <= 0:
        return _random_leaf(n_features, rng=rand)

    # Grow method can terminate early at random.
    if method == "grow" and rand.random() < 0.3:
        return _random_leaf(n_features, rng=rand)

    # Choose unary (30%) or binary (70%)
    if rand.random() < 0.3 and unary_ops:
        op_name = rand.choice(list(unary_ops.keys()))
        child = generate_random_tree(unary_ops,
                                     binary_ops,
                                     n_features,
                                     max_depth - 1,
                                     method=method,
                                     rng=rand)
        return UnaryOpNode(op_name, child)
    elif binary_ops:
        op_name = rand.choice(list(binary_ops.keys()))
        left = generate_random_tree(unary_ops,
                                    binary_ops,
                                    n_features,
                                    max_depth - 1,
                                    method=method,
                                    rng=rand)
        right = generate_random_tree(unary_ops,
                                     binary_ops,
                                     n_features,
                                     max_depth - 1,
                                     method=method,
                                     rng=rand)
        return BinaryOpNode(op_name, left, right)
    else:
        return _random_leaf(n_features, rng=rand)


# ---------------------------------------------------------------------------
# Individual mutation operators
# ---------------------------------------------------------------------------


def _mutate_operator(root: Node,
                     unary_ops: Dict[str, Operator],
                     binary_ops: Dict[str, Operator],
                     rng: Optional[random.Random] = None,
                     **kwargs) -> Node:
    """Replace an operator with another of the same arity."""
    rand = rng if rng is not None else random

    positions = _get_nodes_with_parent(root)
    op_nodes = [(n, p, a)
                for n, p, a in positions
                if isinstance(n, (UnaryOpNode, BinaryOpNode))]
    if not op_nodes:
        return root

    node, parent, attr = rand.choice(op_nodes)

    if isinstance(node, UnaryOpNode) and len(unary_ops) > 1:
        candidates = [name for name in unary_ops if name != node.op_name]
        if candidates:
            node.op_name = rand.choice(candidates)
    elif isinstance(node, BinaryOpNode) and len(binary_ops) > 1:
        candidates = [name for name in binary_ops if name != node.op_name]
        if candidates:
            node.op_name = rand.choice(candidates)

    return root


def _add_node(root: Node,
              unary_ops: Dict[str, Operator],
              binary_ops: Dict[str, Operator],
              n_features: int,
              rng: Optional[random.Random] = None,
              **kwargs) -> Node:
    """Wrap a random subtree with a new operator (grows the tree).

    Picks unary (40%) or binary (60%). For binary, a new random leaf
    is created as the second argument.
    """
    rand = rng if rng is not None else random

    positions = _get_nodes_with_parent(root)
    node, parent, attr = rand.choice(positions)

    if rand.random() < 0.4 and unary_ops:
        op_name = rand.choice(list(unary_ops.keys()))
        new_node = UnaryOpNode(op_name, node)
    elif binary_ops:
        op_name = rand.choice(list(binary_ops.keys()))
        leaf = _random_leaf(n_features, rng=rand)
        if rand.random() < 0.5:
            new_node = BinaryOpNode(op_name, node, leaf)
        else:
            new_node = BinaryOpNode(op_name, leaf, node)
    else:
        return root

    return _replace_node(root, parent, attr, new_node)


def _delete_node(root: Node,
                 rng: Optional[random.Random] = None,
                 **kwargs) -> Node:
    """Remove a random operator, replacing it with one of its children."""
    rand = rng if rng is not None else random

    positions = _get_nodes_with_parent(root)
    op_nodes = [(n, p, a)
                for n, p, a in positions
                if isinstance(n, (UnaryOpNode, BinaryOpNode))]
    if not op_nodes:
        return root

    node, parent, attr = rand.choice(op_nodes)

    if isinstance(node, UnaryOpNode):
        replacement = node.child
    else:  # BinaryOpNode
        replacement = rand.choice([node.left, node.right])

    return _replace_node(root, parent, attr, replacement)


def _mutate_feature(root: Node,
                    n_features: int,
                    rng: Optional[random.Random] = None,
                    **kwargs) -> Node:
    """Change which variable a VariableNode references."""
    rand = rng if rng is not None else random

    if n_features < 2:
        return root

    positions = _get_nodes_with_parent(root)
    var_nodes = [
        (n, p, a) for n, p, a in positions if isinstance(n, VariableNode)
    ]
    if not var_nodes:
        return root

    node, parent, attr = rand.choice(var_nodes)
    candidates = [i for i in range(n_features) if i != node.index]
    if candidates:
        node.index = rand.choice(candidates)

    return root


def _swap_operands(root: Node,
                   rng: Optional[random.Random] = None,
                   **kwargs) -> Node:
    """Swap left and right children of a random non-commutative binary op."""
    rand = rng if rng is not None else random

    positions = _get_nodes_with_parent(root)
    binary_nodes = [(n, p, a)
                    for n, p, a in positions
                    if (isinstance(n, BinaryOpNode) and
                        n.op_name not in COMMUTATIVE_ASSOCIATIVE_BINARY_OPS)]
    if not binary_nodes:
        return root

    node, _, _ = rand.choice(binary_nodes)
    node.left, node.right = node.right, node.left
    return root


def _append_argument(root: Node,
                     binary_ops: Dict[str, Operator],
                     n_features: int,
                     rng: Optional[random.Random] = None,
                     **kwargs) -> Node:
    """Append a local argument inside an operator argument subtree.

    Examples:
      sin(x) -> sin(x + c)
      add(x, y) -> add(x + c, y)
    """
    rand = rng if rng is not None else random
    if not binary_ops:
        return root

    positions = _get_nodes_with_parent(root)
    op_nodes = [(n, p, a)
                for n, p, a in positions
                if isinstance(n, (UnaryOpNode, BinaryOpNode))]
    if not op_nodes:
        return root

    node, _, _ = rand.choice(op_nodes)
    op_name = rand.choice(list(binary_ops.keys()))
    extra_leaf = _random_leaf(n_features, rng=rand)

    def _append_to(subtree: Node) -> BinaryOpNode:
        if rand.random() < 0.5:
            return BinaryOpNode(op_name, subtree, extra_leaf)
        return BinaryOpNode(op_name, extra_leaf, subtree)

    if isinstance(node, UnaryOpNode):
        node.child = _append_to(node.child)
        return root

    # Binary operator: append into one random argument.
    if rand.random() < 0.5:
        node.left = _append_to(node.left)
    else:
        node.right = _append_to(node.right)
    return root


def _randomize_tree(root: Node,
                    unary_ops: Dict[str, Operator],
                    binary_ops: Dict[str, Operator],
                    n_features: int,
                    max_depth: int = 3,
                    rng: Optional[random.Random] = None,
                    **kwargs) -> Node:
    """Replace the current tree with a fresh random tree."""
    rand = rng if rng is not None else random
    return generate_random_tree(unary_ops,
                                binary_ops,
                                n_features,
                                max_depth=max_depth,
                                rng=rand)


def _optimize_constants_mutation(
    root: Node,
    optimize_fn: Optional[Callable[..., Node]] = None,
    optimize_kwargs: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
    **kwargs,
) -> Node:
    """Run constant optimization as a mutation action.

    Falls back to no-op when no optimizer context is provided.
    """
    if optimize_fn is None:
        return root
    try:
        out = optimize_fn(root, **(optimize_kwargs or {}))
    except (RuntimeError, ValueError):
        return root
    return out if out is not None else root


# ---------------------------------------------------------------------------
# Mutation registry and main mutate function
# ---------------------------------------------------------------------------

# Mutation name -> (function, default weight)
# Weights are relative — they get normalized to probabilities.
_MUTATIONS = {
    "mutate_operator": (_mutate_operator, 1.5),
    "add_node": (_add_node, 2.0),
    "delete_node": (_delete_node, 1.0),
    "mutate_feature": (_mutate_feature, 0.5),
    "swap_operands": (_swap_operands, 0.2),
    "append_argument": (_append_argument, 0.5),
    "optimize_constants": (_optimize_constants_mutation, 0.5),
    "randomize_tree": (_randomize_tree, 0.1),
}


def _validate_mutation_weights(weights: Dict[str, float]) -> None:
    """Validate effective mutation weights before sampling."""
    for name, value in weights.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(
                f"mutation weight for '{name}' must be a finite number, got {value}"
            )
        if value < 0:
            raise ValueError(
                f"mutation weight for '{name}' must be non-negative, got {value}"
            )

    total = sum(float(v) for v in weights.values())
    if total <= 0:
        raise ValueError("sum of mutation weights must be > 0")


def validate_custom_mutation_weights(
        weights: Dict[str, float]) -> Dict[str, float]:
    """Validate and normalize a custom mutation-weight mapping.

    Returns a filtered dict that contains only known mutation names.
    Raises ValueError on invalid values or if no known names remain.
    """
    filtered = {
        name: value for name, value in weights.items() if name in _MUTATIONS
    }
    if not filtered:
        raise ValueError(
            "mutation_weights must include at least one known mutation name")
    _validate_mutation_weights(filtered)
    return filtered


def get_default_mutation_weights() -> Dict[str, float]:
    """Return a copy of default mutation weights."""
    return {name: weight for name, (_, weight) in _MUTATIONS.items()}


def mutate_with_info(node: Node,
                     unary_ops: Dict[str, Operator],
                     binary_ops: Dict[str, Operator],
                     n_features: int,
                     max_depth: int = 3,
                     weights: Optional[Dict[str, float]] = None,
                     optimize_fn: Optional[Callable[..., Node]] = None,
                     optimize_kwargs: Optional[Dict[str, Any]] = None,
                     rng: Optional[random.Random] = None) -> Tuple[Node, str]:
    """Apply a random mutation and return (mutated_tree, mutation_name)."""
    rand = rng if rng is not None else random

    if weights is None:
        mutation_weights = get_default_mutation_weights()
    else:
        mutation_weights = validate_custom_mutation_weights(weights)

    _validate_mutation_weights(mutation_weights)

    names = list(mutation_weights.keys())
    w = [mutation_weights[n] for n in names]
    chosen_name = rand.choices(names, weights=w, k=1)[0]

    # Skip optimize_constants when no optimizer is available
    if chosen_name == "optimize_constants" and optimize_fn is None:
        names = [n for n in names if n != "optimize_constants"]
        w = [mutation_weights[n] for n in names]
        if not names:
            return node, "optimize_constants"
        chosen_name = rand.choices(names, weights=w, k=1)[0]

    mutation_fn = _MUTATIONS[chosen_name][0]
    mutated = mutation_fn(root=node,
                          unary_ops=unary_ops,
                          binary_ops=binary_ops,
                          n_features=n_features,
                          max_depth=max_depth,
                          optimize_fn=optimize_fn,
                          optimize_kwargs=optimize_kwargs,
                          rng=rand)
    return mutated, chosen_name
