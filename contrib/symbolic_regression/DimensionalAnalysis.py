from typing import Iterable

_BASE_UNITS = ("m", "kg", "s", "A", "K", "mol", "cd")
_ZERO_UNIT = tuple(0.0 for _ in _BASE_UNITS)
_UNIT_INDEX = {u: i for i, u in enumerate(_BASE_UNITS)}


def _is_number(x):
    return isinstance(x, (int, float))


def _unit_from_term(term, sign):
    term = term.strip()
    if not term:
        return _ZERO_UNIT
    if "^" in term:
        base, pow_s = term.split("^", 1)
        base = base.strip()
        try:
            p = float(pow_s.strip())
        except Exception:
            p = 1.0
    else:
        base = term
        p = 1.0
    p *= sign
    if base not in _UNIT_INDEX:
        return _ZERO_UNIT
    out = list(_ZERO_UNIT)
    out[_UNIT_INDEX[base]] = p
    return tuple(out)


def _add_units(a, b):
    return tuple(float(x) + float(y) for x, y in zip(a, b))


def _sub_units(a, b):
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _scale_units(a, k):
    return tuple(float(x) * float(k) for x in a)


def _same_units(a, b, tol=1e-12):
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def is_dimensionless(unit, tol=1e-12):
    return all(abs(float(x)) <= tol for x in unit)


def canonicalize_unit(unit):
    if unit is None:
        return _ZERO_UNIT
    if isinstance(unit, str):
        s = unit.strip()
        if s in ("", "1"):
            return _ZERO_UNIT
        numer_denom = s.split("/")
        num = numer_denom[0]
        den = numer_denom[1:] if len(numer_denom) > 1 else []
        out = _ZERO_UNIT
        for term in [t for t in num.split("*") if t.strip()]:
            out = _add_units(out, _unit_from_term(term, +1.0))
        for d in den:
            for term in [t for t in d.split("*") if t.strip()]:
                out = _add_units(out, _unit_from_term(term, -1.0))
        return out
    if isinstance(unit, dict):
        out = [0.0 for _ in _BASE_UNITS]
        for k, v in unit.items():
            if k in _UNIT_INDEX:
                out[_UNIT_INDEX[k]] = float(v)
        return tuple(out)
    if isinstance(unit, Iterable):
        arr = list(unit)
        if len(arr) == len(_BASE_UNITS) and all(_is_number(x) for x in arr):
            return tuple(float(x) for x in arr)
    return _ZERO_UNIT


def _get_feature_unit(dataset, feature):
    x_units = getattr(dataset, "x_units", None)
    if x_units is None or feature is None:
        return _ZERO_UNIT
    idx = int(feature)
    if idx < 0 or idx >= len(x_units):
        return _ZERO_UNIT
    return canonicalize_unit(x_units[idx])


def _get_target_unit(dataset):
    y_units = getattr(dataset, "y_units", None)
    if y_units is None:
        return None
    return canonicalize_unit(y_units)


def infer_tree_unit(tree, dataset):
    op = getattr(tree, "op", None)
    children = getattr(tree, "children", [])

    if op == "var":
        return _get_feature_unit(dataset, getattr(tree, "feature", None))
    if op in ("const", "param"):
        return _ZERO_UNIT

    child_units = []
    for ch in children:
        u = infer_tree_unit(ch, dataset)
        if u is None:
            return None
        child_units.append(u)

    if op in ("add", "sub"):
        if len(child_units) != 2 or not _same_units(child_units[0], child_units[1]):
            return None
        return child_units[0]
    if op == "mul":
        if len(child_units) != 2:
            return None
        return _add_units(child_units[0], child_units[1])
    if op == "div":
        if len(child_units) != 2:
            return None
        return _sub_units(child_units[0], child_units[1])
    if op == "pow":
        if len(children) != 2:
            return None
        rhs = children[1]
        if getattr(rhs, "op", None) != "const":
            return None
        try:
            p = float(getattr(rhs, "value", 1.0))
        except Exception:
            p = 1.0
        return _scale_units(child_units[0], p)
    if op in ("neg",):
        return child_units[0] if child_units else _ZERO_UNIT
    if op in ("sin", "cos", "tan", "exp", "log"):
        if len(child_units) != 1:
            return None
        if not is_dimensionless(child_units[0]):
            return None
        return _ZERO_UNIT
    if op == "sqrt":
        if len(child_units) != 1:
            return None
        return _scale_units(child_units[0], 0.5)

    # Unknown operators are treated as dimensionally unconstrained.
    return _ZERO_UNIT


def has_dimensional_mismatch(tree, dataset, options):
    target = _get_target_unit(dataset)
    if target is None:
        return False
    inferred = infer_tree_unit(tree, dataset)
    if inferred is None:
        return True
    return not _same_units(inferred, target)


def dimensional_mismatch_score(tree, dataset, options):
    return 1.0 if has_dimensional_mismatch(tree, dataset, options) else 0.0
