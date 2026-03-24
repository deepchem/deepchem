from Candidate import calculate_complexity
from DimensionalAnalysis import has_dimensional_mismatch


def calculate_depth(tree):
    """
    Depth definition (standard):
      - leaf has depth 1
      - depth = 1 + max(child depths)
    """
    if not tree.children:
        return 1
    return 1 + max(calculate_depth(ch) for ch in tree.children)


def is_constant_subtree(tree):
    """
    Returns True if subtree contains only const nodes and ops (no variables).
    This is optional and used only if forbid_div_by_const is enabled.
    """
    if tree.op == "var":
        return False
    if tree.op == "const":
        return True
    return all(is_constant_subtree(ch) for ch in tree.children)


# =========================
# Constraint checks
# =========================


def check_constraints(
    tree,
    maxsize,
    maxdepth,
    options,
    dataset=None,
):
    """
    Check if `tree` is valid after mutation/crossover.

    4 constraints:
      1) Complexity (size)
      2) Depth
      3) Operator child complexity limits
      4) Illegal nesting (op patterns)

    Returns True if valid, else False.
    """
    # 1) Complexity
    size = calculate_complexity(tree, options)
    if size > maxsize:
        return False

    # 2) Depth
    depth = calculate_depth(tree)
    if depth > maxdepth:
        return False

    # 3) Operator child complexity limits
    if options.max_child_complexity is not None or options.max_child_depth is not None:
        if not _check_child_limits(tree, options):
            return False

    # 4) Illegal nesting
    if options.illegal_nesting is not None:
        if not _check_illegal_nesting(tree, options.illegal_nesting):
            return False

    if options.forbid_div_by_const:
        if not _check_no_div_by_const(tree):
            return False

    # Domain-based constraints are left as a knob; enforcing them requires
    # operator-aware static analysis or runtime probing.
    if options.forbid_domain_violations:
        # You can implement later if you add ops like log/sqrt.
        pass

    if (
        getattr(options, "enforce_dimensional_constraints", False)
        and dataset is not None
    ):
        if has_dimensional_mismatch(tree, dataset, options):
            return False

    return True


def _check_child_limits(tree, options):
    """
    Enforce per-operator child complexity/depth constraints.
    """
    op = tree.op

    # Determine limits for this operator
    c_lims = None
    d_lims = None
    if options.max_child_complexity is not None:
        c_lims = options.max_child_complexity.get(op, None)
    if options.max_child_depth is not None:
        d_lims = options.max_child_depth.get(op, None)

    # Check each child against corresponding limit (if provided)
    for idx, ch in enumerate(tree.children):
        if c_lims is not None and idx < len(c_lims):
            if calculate_complexity(ch, options) > c_lims[idx]:
                return False
        if d_lims is not None and idx < len(d_lims):
            if calculate_depth(ch) > d_lims[idx]:
                return False

        # recurse
        if not _check_child_limits(ch, options):
            return False

    return True


def _check_illegal_nesting(tree, illegal_nesting):
    """
    Enforce direct parent->child illegal operator nesting.
    """
    parent_op = tree.op
    banned_children = illegal_nesting.get(parent_op, set())

    for ch in tree.children:
        child_op = ch.op
        if child_op in banned_children:
            return False
        if not _check_illegal_nesting(ch, illegal_nesting):
            return False

    return True


def _check_no_div_by_const(tree):
    """
    If there is a div node, forbid the denominator from being constant-only subtree.
    """
    if tree.op == "div":
        if len(tree.children) != 2:
            return False  # malformed
        denom = tree.children[1]
        if is_constant_subtree(denom):
            return False

    for ch in tree.children:
        if not _check_no_div_by_const(ch):
            return False
    return True
