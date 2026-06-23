import random
from Utils import simplify_tree, optimize_constants, convert_to_canonical_shape
from Candidate import Candidate, calculate_complexity, calculate_loss_and_cost

class Mutationresult:
    """
        i use this to diff between the mutations that need to be return_immediately and those that just return a tree
    """
    def __init__(self, tree=None, member=None, return_immediately=False):
        self.tree = tree
        self.member = member
        self.return_immediately = return_immediately

def mutate_constant(tree,temperature, options):
    return tree.mutate_constant(temperature,options)

def mutate_operator(tree, options):
    return tree.mutate_operator(options)

def mutate_feature(tree, nfeatures):
    return tree.mutate_feature(nfeatures)

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



def mutate(tree,member, mutation_choice, options, **kws):
    if mutation_choice == "mutate_constant":
        t = mutate_constant(tree, kws["temperature"], options)
        return Mutationresult(tree=t)

    if mutation_choice == "mutate_operator":
        t = mutate_operator(tree, options)
        return Mutationresult(tree=t)

    if mutation_choice == "mutate_feature":
        t = mutate_feature(tree, kws["nfeatures"])
        return Mutationresult(tree=t)

    if mutation_choice == "swap_operands":
        t = swap_operands(tree)
        return Mutationresult(tree=t)

    if mutation_choice == "add_node":
        if random.random() < 0.5:
            t = append_random_op(tree, options, kws["nfeatures"])
        else:
            t = prepend_random_op(tree, options, kws["nfeatures"])
        return Mutationresult(tree=t)

    if mutation_choice == "insert_node":
        t = insert_random_op(tree, options, kws["nfeatures"])
        return Mutationresult(tree=t)

    if mutation_choice == "delete_node":
        t = delete_random_op(tree)
        return Mutationresult(tree=t)

    if mutation_choice == "rotate_tree":
        t = randomly_rotate_tree(tree)
        return Mutationresult(tree=t)


    if mutation_choice == "simplify":
        assert options.should_simplify
        t = simplify_tree(tree)
        t = convert_to_canonical_shape(t)
        new_complexity = calculate_complexity(t)
        new_loss, new_cost = calculate_loss_and_cost(
            new_complexity, kws["dataset"], t, options.parsimony_penalty
        )
        new_member = Candidate.from_values(
            tree=t,
            cost=new_cost,
            loss=new_loss,
            complexity=new_complexity,
        )
        return Mutationresult(member=new_member, return_immediately=True)

    if mutation_choice == "randomize":
        t = randomize_tree(tree, kws["curmaxsize"], options, kws["nfeatures"])
        return Mutationresult(tree=t)

    if mutation_choice == "optimize":
        new_member = optimize_constants(kws["dataset"], member, options)
        return Mutationresult(member=new_member, return_immediately=True)

    if mutation_choice == "do_nothing":
        new_member = member.deep_copy()
        return Mutationresult(member=new_member, return_immediately=True)

    raise ValueError(f"Unknown mutation choice: {mutation_choice}")