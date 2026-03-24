import math
from Candidate import Candidate
from Register import OP_REGISTRY
import time
import torch
from TreeDS import Tree
from Complexity import calculate_complexity
from LossAndCost import calculate_loss_and_cost

def simplify_tree(tree):
    """
    A small, safe simplifier:
      - Recursively simplify children first
      - Constant folding for unary/binary ops when all children are const leaves
      - Algebraic identities:
          x + 0 = x
          0 + x = x
          x - 0 = x
          x * 1 = x
          1 * x = x
          x * 0 = 0
          0 * x = 0
          x / 1 = x
          0 / x = 0  (x != 0; we still keep it as 0)
          neg(neg(x)) = x
    """
    # Leaf => already simplest
    if tree.is_leaf():
        return tree

    # Simplify children first
    new_children = [simplify_tree(ch) for ch in tree.children]
    tree = Tree(tree.op, children=new_children, feature=tree.feature, value=tree.value)

    op = tree.op

    # --- Constant folding helpers ---
    def _is_const_leaf(t):
        return t.is_constant_leaf()

    def _const_value(t):
        return float(t.value)

    def _make_const(v):
        return Tree("const", value=float(v))

    def _same_tree(a, b):
        # Structural equality via string form (cheap and deterministic).
        return a.to_string() == b.to_string()

    # Unary constant folding
    if (
        op in ("neg", "sin", "cos")
        and len(tree.children) == 1
        and _is_const_leaf(tree.children[0])
    ):
        a = _const_value(tree.children[0])
        if op == "neg":
            return _make_const(-a)
        if op == "sin":
            return _make_const(math.sin(a))
        if op == "cos":
            return _make_const(math.cos(a))

    # Binary constant folding
    if op in ("add", "sub", "mul", "div") and len(tree.children) == 2:
        a, b = tree.children

        if _is_const_leaf(a) and _is_const_leaf(b):
            av = _const_value(a)
            bv = _const_value(b)
            if op == "add":
                return _make_const(av + bv)
            if op == "sub":
                return _make_const(av - bv)
            if op == "mul":
                return _make_const(av * bv)
            if op == "div":
                # Keep protected semantics consistent with evaluation:
                # division by zero should be invalid (NaN), not silently shifted.
                if bv == 0.0:
                    return _make_const(float("nan"))
                return _make_const(av / bv)

        # Algebraic identities with 0/1
        if op == "add":
            if _is_const_leaf(a) and _const_value(a) == 0.0:
                return b
            if _is_const_leaf(b) and _const_value(b) == 0.0:
                return a
            # x + (-x) = 0, (-x) + x = 0
            if a.op == "neg" and _same_tree(a.children[0], b):
                return _make_const(0.0)
            if b.op == "neg" and _same_tree(b.children[0], a):
                return _make_const(0.0)

        if op == "sub":
            if _is_const_leaf(b) and _const_value(b) == 0.0:
                return a
            # x - x = 0
            if _same_tree(a, b):
                return _make_const(0.0)

        if op == "mul":
            if (_is_const_leaf(a) and _const_value(a) == 0.0) or (
                _is_const_leaf(b) and _const_value(b) == 0.0
            ):
                return _make_const(0.0)
            if _is_const_leaf(a) and _const_value(a) == 1.0:
                return b
            if _is_const_leaf(b) and _const_value(b) == 1.0:
                return a

        if op == "div":
            # 0/x is only safe when x != 0. Keep as-is to avoid changing domain behavior.
            if _is_const_leaf(b) and _const_value(b) == 1.0:
                return a

    # neg(neg(x)) = x
    if op == "neg" and len(tree.children) == 1:
        child = tree.children[0]
        if child.op == "neg" and len(child.children) == 1:
            return child.children[0]

    return tree


def combine_ops(tree):
    if tree.is_leaf():
        return tree

    # First recurse
    children = [combine_ops(ch) for ch in tree.children]
    tree = Tree(tree.op, children=children, feature=tree.feature, value=tree.value)

    def _collect(op, t, out):
        if t.op == op and len(t.children) == 2:
            _collect(op, t.children[0], out)
            _collect(op, t.children[1], out)
        else:
            out.append(t)

    def _rebuild(op, terms):
        # deterministic fold: (((t0 op t1) op t2) op ...)
        cur = terms[0]
        for nxt in terms[1:]:
            cur = Tree(op, children=[cur, nxt])
        return cur

    if tree.op in ("add", "mul") and len(tree.children) == 2:
        terms = []
        _collect(tree.op, tree, terms)

        consts = [t for t in terms if t.is_constant_leaf()]
        nonconsts = [t for t in terms if not t.is_constant_leaf()]
        terms = nonconsts + consts

        if len(terms) == 1:
            return terms[0]
        return _rebuild(tree.op, terms)

    return tree


def optimize_constants(dataset, cand, options):
    # If no constants => nothing to do
    if cand.tree.count_constants() == 0:
        return cand

    X = dataset.X
    y = dataset.y
    w = getattr(dataset, "weights", None)

    # ---- Collect constant nodes (paths) ----
    paths = []

    def _collect_paths(t, path):
        if t.is_constant_leaf():
            paths.append(path)
            return
        for i, ch in enumerate(t.children):
            _collect_paths(ch, path + (i,))

    _collect_paths(cand.tree, ())

    # Initial values
    init_vals = []
    for p in paths:
        node = _get_node_by_path(cand.tree, p)
        init_vals.append(float(node.value))

    x0 = torch.tensor(init_vals, device=X.device, dtype=X.dtype)
    method = str(getattr(options, "opt_method", "lbfgs")).lower()
    lr = float(getattr(options, "opt_lr", 0.1))
    steps = int(getattr(options, "opt_steps", 50))
    nrestarts = max(0, int(getattr(options, "optimizer_nrestarts", 0)))

    def _objective_from_params(param_values):
        with torch.no_grad():
            yhat = _forward_with_constants(cand.tree, X, paths, param_values)
            loss_t = _optimization_loss(yhat, y, w, options)
            return float(loss_t.detach().cpu().item())

    def _fit_from_start(start):
        params = torch.nn.Parameter(start.clone())
        large_penalty = torch.full((), 1e12, device=X.device, dtype=X.dtype)

        def objective():
            yhat = _forward_with_constants(cand.tree, X, paths, params)
            loss_t = _optimization_loss(yhat, y, w, options)
            if not torch.isfinite(loss_t):
                return (params * 0.0).sum() + large_penalty
            return loss_t

        if method == "adam":
            opt = torch.optim.Adam([params], lr=lr)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss_t = objective()
                loss_t.backward()
                opt.step()
        else:
            opt = torch.optim.LBFGS([params], lr=lr, max_iter=steps)

            def closure():
                opt.zero_grad(set_to_none=True)
                loss_t = objective()
                loss_t.backward()
                return loss_t

            opt.step(closure)

        with torch.no_grad():
            params.data = torch.nan_to_num(
                params.data, nan=0.0, posinf=1e3, neginf=-1e3
            )
            params.data = torch.clamp(params.data, -1e3, 1e3)
            final_loss = objective().detach().cpu().item()

        return float(final_loss), params.detach().clone()

    baseline_loss = _objective_from_params(x0)
    best_loss, best_params = _fit_from_start(x0)

    for _ in range(nrestarts):
        eps = torch.randn_like(x0)
        xt = x0 * (1.0 + 0.5 * eps)
        trial_loss, trial_params = _fit_from_start(xt)
        if trial_loss < best_loss:
            best_loss = trial_loss
            best_params = trial_params

    # Only accept if constant optimization improved objective.
    if not math.isfinite(best_loss) or best_loss >= baseline_loss:
        return cand

    # Write back constants into new tree
    new_tree = cand.tree.clone()
    with torch.no_grad():
        for i, p in enumerate(paths):
            _set_const_value_by_path(
                new_tree, p, float(best_params[i].detach().cpu().item())
            )

    # Recompute loss/cost/complexity using your existing functions
    new_complexity = calculate_complexity(new_tree, options)
    new_loss, new_cost = calculate_loss_and_cost(
        new_complexity, dataset, new_tree, options.parsimony_penalty, options=options
    )
    if (
        math.isnan(new_loss)
        or math.isinf(new_loss)
        or math.isnan(new_cost)
        or math.isinf(new_cost)
    ):
        return cand
    if new_cost >= cand.cost:
        return cand

    return Candidate.from_values(
        tree=new_tree,
        cost=new_cost,
        loss=new_loss,
        complexity=new_complexity,
    )


# -------------------------
# Helpers for optimize_constants
# -------------------------


def _optimization_loss(
    yhat,
    y,
    w,
    options,
):
    """
    Differentiable loss used during constant optimization.
    Mirrors the configured builtin losses when possible.
    """
    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
        return torch.full((), float("inf"), device=yhat.device, dtype=yhat.dtype)

    name = str(
        getattr(options, "loss_name", "mse") if options is not None else "mse"
    ).lower()
    err = yhat - y

    if name == "mse":
        loss_vec = err.square()
    elif name == "mae":
        loss_vec = torch.abs(err)
    elif name == "huber":
        delta = float(getattr(options, "huber_delta", 1.0))
        abs_err = torch.abs(err)
        quad = torch.minimum(
            abs_err, torch.tensor(delta, device=abs_err.device, dtype=abs_err.dtype)
        )
        lin = abs_err - quad
        loss_vec = 0.5 * quad.square() + delta * lin
    elif name == "logcosh":
        loss_vec = torch.log(torch.cosh(torch.clamp(err, -20.0, 20.0)))
    elif name == "quantile":
        q = float(getattr(options, "quantile", 0.5))
        q = min(1.0, max(0.0, q))
        u = y - yhat
        loss_vec = torch.maximum(q * u, (q - 1.0) * u)
    else:
        # Fallback to MSE.
        loss_vec = err.square()

    if w is None:
        return loss_vec.mean()
    return (loss_vec * w).sum() / (w.sum() + 1e-12)


def _get_node_by_path(tree, path):
    cur = tree
    for idx in path:
        cur = cur.children[idx]
    return cur


def _set_const_value_by_path(tree, path, value):
    node = _get_node_by_path(tree, path)
    if node.op != "const":
        raise ValueError("Path does not point to a const leaf.")
    node.value = float(value)


def _forward_with_constants(
    tree,
    X,
    const_paths,
    params,
):
    """
    Evaluate `tree` on X, but replace constant leaves at const_paths with params[i].
    This avoids editing the Tree structure during optimization.
    """
    # Build a lookup from path -> param index for O(1) checks
    path_to_i = {p: i for i, p in enumerate(const_paths)}

    def rec(t, path):
        op = t.op

        if op == "var":
            return X[:, t.feature]

        if op == "const":
            if path in path_to_i:
                v = params[path_to_i[path]]
            else:
                v = torch.tensor(float(t.value), device=X.device, dtype=X.dtype)
            return v.expand(X.shape[0])

        spec = OP_REGISTRY.get(op)
        if spec is not None:
            if spec.arity != len(t.children):
                raise ValueError(
                    f"Arity mismatch for op '{op}': expected {spec.arity}, got {len(t.children)}"
                )
            args = [rec(t.children[i], path + (i,)) for i in range(spec.arity)]
            return spec.fn(*args)

        raise ValueError(f"Unknown op: {op}")

    return rec(tree, ())


def update_hof_from_candidates(hof, cands, options):
    for cand in cands:
        size = cand.complexity
        if 0 < size <= options.maxsize:
            if (size not in hof) or (cand.cost < hof[size].cost):
                hof[size] = cand.copy() if hasattr(cand, "copy") else cand.deep_copy()


def update_hof_from_best_seen(hof, best_seen, options):
    for size, cand in best_seen.items():
        if 0 < size <= options.maxsize:
            if (size not in hof) or (cand.cost < hof[size].cost):
                hof[size] = cand.copy() if hasattr(cand, "copy") else cand.deep_copy()


def get_cur_maxsize(options, total_cycles, cycles_remaining):
    """
    update current maxsize based on warmup schedule
    """
    warmup = getattr(options, "warmup_maxsize_by", 0.0)
    if warmup is None:
        warmup = 0.0

    cycles_elapsed = total_cycles - cycles_remaining
    fraction_elapsed = (
        float(cycles_elapsed) / float(total_cycles) if total_cycles > 0 else 1.0
    )
    in_warmup_period = (warmup > 0.0) and (fraction_elapsed <= warmup)

    if warmup > 0.0 and in_warmup_period:
        return 3 + int((options.maxsize - 3) * fraction_elapsed / warmup)
    else:
        return options.maxsize


def calculate_pareto_frontier(hof):
    """
    Dominating Pareto frontier:
    keep equations that are better than all simpler equations.
    """
    if not hof:
        return []
    dominating = []
    best_loss = float("inf")
    for size in sorted(hof.keys()):
        cand = hof[size]
        if cand.loss < best_loss:
            dominating.append(
                cand.copy() if hasattr(cand, "copy") else cand.deep_copy()
            )
            best_loss = cand.loss
    return dominating


def _relu(x):
    return x if x > 0.0 else 0.0


def compute_direct_score(cur_loss, last_loss, delta_c):
    delta = cur_loss - last_loss
    return _relu(-delta / max(1, delta_c))


def compute_zero_centered_score(cur_loss, last_loss, delta_c):
    ratio = _relu(cur_loss / (last_loss + 1e-30))
    log_ratio = math.log(ratio + 1e-30)
    return _relu(-log_ratio / max(1, delta_c))


def format_hall_of_fame(hof, options):
    dominating = calculate_pareto_frontier(hof)
    trees = [member.tree for member in dominating]
    losses = [float(member.loss) for member in dominating]
    complexities = [int(member.complexity) for member in dominating]
    scores = []

    loss_scale = str(getattr(options, "loss_scale", "log")).lower()
    if loss_scale not in ("linear", "log"):
        loss_scale = "log"

    if loss_scale == "log":
        for loss in losses:
            if loss < 0.0:
                raise ValueError(
                    "loss_scale='log' requires non-negative losses. "
                    "Set options.loss_scale='linear' for negative losses."
                )

    last_loss = float("inf")
    last_complexity = 0
    for i, (cur_loss, complexity) in enumerate(zip(losses, complexities)):
        if i == 0:
            scores.append(0.0)
        else:
            delta_c = complexity - last_complexity
            if loss_scale == "linear":
                s = compute_direct_score(cur_loss, last_loss, delta_c)
            else:
                s = compute_zero_centered_score(cur_loss, last_loss, delta_c)
            scores.append(float(s))
        last_loss = cur_loss
        last_complexity = complexity

    return {
        "candidates": dominating,
        "trees": trees,
        "scores": scores,
        "losses": losses,
        "complexities": complexities,
    }


def select_best_candidate(hof, options):
    if not hof:
        return None

    model_selection = str(getattr(options, "model_selection", "pareto_score")).lower()
    if model_selection == "min_cost":
        return min(hof.values(), key=lambda c: c.cost).deep_copy()

    formatted = format_hall_of_fame(hof, options)
    cands = formatted["candidates"]
    scores = formatted["scores"]
    losses = formatted["losses"]
    if not cands:
        return None

    topn = int(getattr(options, "topn", 0))
    if topn > 0 and len(cands) > topn:
        keep = sorted(
            range(len(cands)),
            key=lambda i: (scores[i], -cands[i].complexity, -cands[i].loss),
            reverse=True,
        )[:topn]
        cands = [cands[i] for i in keep]
        scores = [scores[i] for i in keep]
        losses = [losses[i] for i in keep]

    if model_selection == "best":
        min_loss = min(losses)
        mult = float(getattr(options, "best_loss_multiplier", 1.5))
        if min_loss > 0.0:
            cutoff = min_loss * max(mult, 1.0)
            eligible = [i for i, loss in enumerate(losses) if loss <= cutoff]
        else:
            eligible = [i for i, loss in enumerate(losses) if loss <= min_loss]
        if not eligible:
            eligible = list(range(len(cands)))
        best_idx = max(
            eligible,
            key=lambda i: (scores[i], -cands[i].complexity, -cands[i].loss),
        )
        return cands[best_idx].deep_copy()

    best_idx = max(
        range(len(cands)),
        key=lambda i: (scores[i], -cands[i].complexity, -cands[i].loss),
    )
    return cands[best_idx].deep_copy()


def poisson_sample(lam, rng=None):
    """
    Simple Poisson sampler (Knuth) for small to moderate lambda.
    """
    if lam <= 0.0:
        return 0
    rng = rng or __import__("random")
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def migrate_candidates(
    candidates,
    population_members,
    frac,
    rng=None,
):
    """
    Replace a fraction of population members with random candidates.
    """
    if not candidates or not population_members or frac <= 0.0:
        return

    rng = rng or __import__("random")
    pop_size = len(population_members)
    mean_replace = pop_size * frac
    num_replace = poisson_sample(mean_replace, rng=rng)
    num_replace = min(num_replace, len(candidates), pop_size)
    if num_replace <= 0:
        return

    locations = [rng.randrange(pop_size) for _ in range(num_replace)]
    migrants = [rng.choice(candidates) for _ in range(num_replace)]
    for idx, cand in zip(locations, migrants):
        copied = cand.deep_copy()
        copied.birth = int(time.time_ns())
        population_members[idx] = copied
