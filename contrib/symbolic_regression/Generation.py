import math
import random
from Utils import simplify_tree, optimize_constants, combine_ops

from Options import V_MUTATIONS
from CheckConstrains import check_constraints
from Candidate import Candidate
from Complexity import calculate_complexity
from LossAndCost import calculate_loss_and_cost


def count_scalar_constants(tree):
    return tree.count_constants()


def condition_mutate_constant(weights, member):
    n_constants = count_scalar_constants(member.tree)
    scale = min(8, n_constants) / 8.0
    weights.mutate_constant *= scale


def condition_mutation_weights(weights, member, options, curmaxsize, nfeatures):
    tree = member.tree

    if not options.should_optimize_constants:
        weights.optimize = 0.0

    if tree.is_leaf():
        weights.mutate_operator = 0.0
        weights.swap_operands = 0.0
        weights.delete_node = 0.0
        weights.rotate_tree = 0.0
        weights.simplify = 0.0

        if not tree.is_constant_leaf():  # leaf is variable x_i
            weights.optimize = 0.0
            weights.mutate_constant = 0.0
        else:  # leaf is constant
            weights.mutate_feature = 0.0
        return

    # swap_operands only if any binary ops exist
    if not tree.has_any_binary_op():
        weights.swap_operands = 0.0
        weights.rotate_tree = 0.0

    n_constants = count_scalar_constants(tree)
    condition_mutate_constant(weights, member)
    if n_constants == 0:
        weights.mutate_constant = 0.0
        weights.optimize = 0.0

    if nfeatures <= 1:
        weights.mutate_feature = 0.0

    complexity = calculate_complexity(tree, options)
    if complexity >= curmaxsize:
        weights.add_node = 0.0
        weights.insert_node = 0.0

    if not options.should_simplify:
        weights.simplify = 0.0


def sample_mutation(w):
    weights = w.to_vector()
    return random.choices(V_MUTATIONS, weights=weights, k=1)[0]


def mutate_constant(tree, temperature, options):
    return tree.mutate_constant(temperature, options)


def mutate_operator(tree, options):
    return tree.mutate_operator(options)


def mutate_feature(tree, nfeatures):
    return tree.mutate_a_feature(nfeatures)


def swap_operands(tree):
    return tree.swap_random_operands()


def append_random_op(tree, options, nfeatures):
    return tree.append_random_op(options, nfeatures)


def prepend_random_op(tree, options, nfeatures):
    return tree.prepend_random_op(options, nfeatures)


def insert_random_op(tree, options, nfeatures):
    return tree.insert_random_op(options, nfeatures)


def delete_random_op(tree):
    return tree.delete_random_op()


def randomly_rotate_tree(tree):
    return tree.rotate_randomly()


def randomize_tree(tree, curmaxsize, options, nfeatures):
    return tree.randomize(curmaxsize, options, nfeatures)


class MutationResult:
    def __init__(self, tree=None, member=None, return_immediately=False):
        self.tree = tree  # expected Tree
        self.member = member  # expected Candidate
        self.return_immediately = return_immediately
        assert (self.tree is None) ^ (
            self.member is None
        ), "MutationResult must return either a tree or a member, not both."


def mutate(tree, member, mutation_choice, options, **kws):
    if mutation_choice == "mutate_constant":
        t = mutate_constant(tree, kws["temperature"], options)
        return MutationResult(tree=t)

    if mutation_choice == "mutate_operator":
        t = mutate_operator(tree, options)
        return MutationResult(tree=t)

    if mutation_choice == "mutate_feature":
        t = mutate_feature(tree, kws["nfeatures"])
        return MutationResult(tree=t)

    if mutation_choice == "swap_operands":
        t = swap_operands(tree)
        return MutationResult(tree=t)

    if mutation_choice == "add_node":
        if random.random() < 0.5:
            t = append_random_op(tree, options, kws["nfeatures"])
        else:
            t = prepend_random_op(tree, options, kws["nfeatures"])
        return MutationResult(tree=t)

    if mutation_choice == "insert_node":
        t = insert_random_op(tree, options, kws["nfeatures"])
        return MutationResult(tree=t)

    if mutation_choice == "delete_node":
        t = delete_random_op(tree)
        return MutationResult(tree=t)

    if mutation_choice == "rotate_tree":
        t = randomly_rotate_tree(tree)
        return MutationResult(tree=t)

    # ---- early return mutations ----
    if mutation_choice == "simplify":
        assert options.should_simplify
        t = simplify_tree(tree)
        t = combine_ops(t)
        new_complexity = calculate_complexity(t, options)
        new_loss, new_cost = calculate_loss_and_cost(
            new_complexity,
            kws["dataset"],
            t,
            options.parsimony_penalty,
            options=options,
        )
        new_member = Candidate.from_values(
            tree=t,
            cost=new_cost,
            loss=new_loss,
            complexity=new_complexity,
        )
        return MutationResult(member=new_member, return_immediately=True)

    if mutation_choice == "randomize":
        t = randomize_tree(tree, kws["curmaxsize"], options, kws["nfeatures"])
        return MutationResult(tree=t)

    if mutation_choice == "optimize":
        new_member = optimize_constants(kws["dataset"], member, options)
        return MutationResult(member=new_member, return_immediately=True)

    if mutation_choice == "do_nothing":
        new_member = member.deep_copy()
        return MutationResult(member=new_member, return_immediately=True)

    raise ValueError(f"Unknown mutation choice: {mutation_choice}")


def next_generation(
    dataset,
    member,
    temperature,
    curmaxsize,
    running_stats,
    options,
):

    before_cost, before_loss = member.cost, member.loss

    nfeatures = options.nfeatures

    # choose ONE mutation
    weights = options.mutation_weights.copy()
    condition_mutation_weights(weights, member, options, curmaxsize, nfeatures)
    mutation_choice = sample_mutation(weights)

    successful = False
    attempts = 0
    max_attempts = 10

    tree = None
    while (not successful) and attempts < max_attempts:
        tree_copy = member.tree.clone()

        result = mutate(
            tree_copy,
            member,
            mutation_choice,
            options,
            temperature=temperature,
            dataset=dataset,
            curmaxsize=curmaxsize,
            nfeatures=nfeatures,
        )
        if result.return_immediately:
            return result.member, True

        tree = result.tree
        successful = check_constraints(
            tree=tree,
            options=options,
            maxsize=curmaxsize,
            maxdepth=options.maxdepth,
            dataset=dataset,
        )
        attempts += 1

    if not successful:
        return member.deep_copy(), False

    complexity = calculate_complexity(tree, options)
    after_loss, after_cost = calculate_loss_and_cost(
        dataset=dataset,
        tree=tree,
        parsimony_penalty=options.parsimony_penalty,
        complexity=complexity,
        options=options,
    )

    if math.isnan(after_cost):
        return member.deep_copy(), False

    # Always accept improvements.
    if after_cost <= before_cost:
        baby = Candidate.from_values(
            tree=tree,
            cost=after_cost,
            loss=after_loss,
            complexity=complexity,
        )
        return baby, True

    prob = 1.0
    if options.annealing:
        # Guard against zero/negative temperature or alpha.
        if temperature <= 0.0 or options.alpha <= 0.0:
            prob = 1.0
        else:
            delta = after_cost - before_cost
            arg = -delta / (temperature * options.alpha)
            # Clamp to avoid overflow while preserving acceptance behavior.
            if arg > 700.0:
                prob *= math.exp(700.0)
            elif arg < -700.0:
                prob *= 0.0
            else:
                prob *= math.exp(arg)

    if options.use_frequency and running_stats is not None:
        old_size = member.complexity
        new_size = complexity
        if 1 <= old_size <= options.maxsize:
            old_freq = running_stats.normalized_frequencies[old_size - 1]
        else:
            old_freq = 1e-6
        if 1 <= new_size <= options.maxsize:
            new_freq = running_stats.normalized_frequencies[new_size - 1]
        else:
            new_freq = 1e-6
        if new_freq <= 0:
            new_freq = 1e-6
        prob *= old_freq / new_freq

    if prob < random.random():
        return member.deep_copy(), False

    baby = Candidate.from_values(
        tree=tree,
        cost=after_cost,
        loss=after_loss,
        complexity=complexity,
    )
    return baby, True


def _random_node_and_parent(tree, rng=None):
    """
    Returns (node, parent, idx)
    - idx == 0 means node is the root (so parent is returned as node itself)
    - else idx is 1-based index in parent.children
    """
    rng = rng or random
    node = tree.random_node(rng)

    if node is tree:
        return node, node, 0  # root special case

    parent, idx = tree.find_parent(node)
    return node, parent, idx


def crossover_trees(tree1, tree2, rng=None):
    """
    Returns (new_tree1, new_tree2). Does NOT modify inputs.
    """
    rng = rng or random

    if tree1 is tree2:
        raise ValueError("Attempted to crossover the same tree object!")

    # Copy whole trees so original remain unchanged
    t1 = tree1.clone()
    t2 = tree2.clone()

    # Pick random nodes and parents
    n1, p1, i1 = _random_node_and_parent(t1, rng)
    n2, p2, i2 = _random_node_and_parent(t2, rng)

    # Copy n1 because we will overwrite its spot in t1
    n1_copy = n1.clone()

    # Splice n2 into t1
    if i1 == 0:
        t1 = n2.clone()  # replace root
    else:
        p1.set_child(i1, n2.clone())

    # Splice n1 into t2
    if i2 == 0:
        t2 = n1_copy  # replace root
    else:
        p2.set_child(i2, n1_copy)

    return t1, t2


from typing import Tuple, Optional, Dict, Any


def crossover_generation(
    member1,
    member2,
    dataset,
    curmaxsize,
    options,
):

    tree1 = member1.tree
    tree2 = member2.tree
    crossover_accepted = False

    child_tree1, child_tree2 = crossover_trees(tree1, tree2)

    num_tries = 1
    max_tries = 10
    num_evals = 0.0

    afterSize1 = -1
    afterSize2 = -1

    while True:
        afterSize1 = calculate_complexity(child_tree1, options)
        afterSize2 = calculate_complexity(child_tree2, options)

        # Both trees satisfy constraints
        if check_constraints(
            tree=child_tree1,
            options=options,
            maxsize=curmaxsize,
            maxdepth=options.maxdepth,
            dataset=dataset,
        ) and check_constraints(
            tree=child_tree2,
            options=options,
            maxsize=curmaxsize,
            maxdepth=options.maxdepth,
            dataset=dataset,
        ):
            break

        if num_tries > max_tries:
            crossover_accepted = False
            return member1, member2, crossover_accepted

        child_tree1, child_tree2 = crossover_trees(tree1, tree2)
        num_tries += 1

    # Evaluate children
    after_loss1, after_cost1 = calculate_loss_and_cost(
        dataset=dataset,
        tree=child_tree1,
        parsimony_penalty=options.parsimony_penalty,
        complexity=afterSize1,
        options=options,
    )
    after_loss2, after_cost2 = calculate_loss_and_cost(
        dataset=dataset,
        tree=child_tree2,
        parsimony_penalty=options.parsimony_penalty,
        complexity=afterSize2,
        options=options,
    )

    # Build babies (you may have your own constructor)
    baby1 = Candidate.from_values(
        tree=child_tree1,
        cost=float(after_cost1),
        loss=float(after_loss1),
        complexity=afterSize1,
    )

    baby2 = Candidate.from_values(
        tree=child_tree2,
        cost=float(after_cost2),
        loss=float(after_loss2),
        complexity=afterSize2,
    )

    crossover_accepted = True
    return baby1, baby2, crossover_accepted
