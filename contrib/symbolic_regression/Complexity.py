def _get_var_complexity(options, feature):
    if options is None:
        return 1.0
    value = getattr(options, "complexity_of_variables", 1.0)
    if isinstance(value, (list, tuple)):
        if feature is None:
            return float(value[0]) if len(value) > 0 else 1.0
        idx = int(feature)
        if 0 <= idx < len(value):
            return float(value[idx])
        return float(value[-1]) if len(value) > 0 else 1.0
    return float(value)


def _compute_complexity_raw(
    tree,
    options,
    _seen,
):
    if _seen is None:
        _seen = set()
    node_id = id(tree)
    if node_id in _seen:
        raise ValueError("Cycle detected in Tree while computing complexity.")
    _seen.add(node_id)

    if tree.is_leaf():
        if tree.op == "const":
            raw = (
                float(getattr(options, "complexity_of_constants", 1.0))
                if options
                else 1.0
            )
        elif tree.op == "var":
            raw = _get_var_complexity(options, tree.feature)
        else:
            raw = 1.0
        _seen.remove(node_id)
        return raw

    op_weights = getattr(options, "complexity_of_operators", None) if options else None
    op_complexity = (
        float(op_weights.get(tree.op, 1.0)) if op_weights is not None else 1.0
    )
    child_sum = sum(
        _compute_complexity_raw(child, options, _seen) for child in tree.children
    )
    _seen.remove(node_id)
    return op_complexity + child_sum


def calculate_complexity(
    tree,
    options=None,
    _seen=None,
):
    if options is not None and getattr(options, "complexity_mapping", None) is not None:
        fn = options.complexity_mapping
        try:
            out = fn(tree, options)
        except TypeError:
            out = fn(tree)
        return max(1, int(round(float(out))))

    raw = _compute_complexity_raw(tree, options, _seen)
    return max(1, int(round(raw)))




