import random
import math
from Candidate import Candidate
from TreeDS import Tree


def generate_random_tree(
    max_depth,
    options,
    *,
    p_const=0.2,
):
    """
    Generate a random expression tree up to max_depth.

    Leaves:
      - variable node: "var" with feature in [0, nfeatures-1]
      - constant node: "const" with random value

    Internal nodes:
      sampled from options.ops with correct arity.
    """
    if max_depth <= 0:
        if random.random() < p_const:
            return Tree("const", value=random.uniform(-3.0, 3.0))
        return Tree("var", feature=random.randrange(options.nfeatures))

    if random.random() < 0.3:
        if random.random() < p_const:
            return Tree("const", value=random.uniform(-3.0, 3.0))
        return Tree("var", feature=random.randrange(options.nfeatures))

    op = random.choice(list(options.ops))
    arity = Tree._arity(op)

    if arity == 1:
        child = generate_random_tree(max_depth - 1, options, p_const=p_const)
        return Tree(op, children=[child])

    if arity == 2:
        left = generate_random_tree(max_depth - 1, options, p_const=p_const)
        right = generate_random_tree(max_depth - 1, options, p_const=p_const)
        return Tree(op, children=[left, right])

    # If ops contains unknowns or unsupported arity, fail loudly
    raise ValueError(f"Unsupported op arity for options.ops: {op} (arity={arity})")


# ============================================================
# Population
# ============================================================


class Population:
    """
    Population holds a list of Candidate objects.
    """

    def __init__(self, candidates):
        self.candidates = candidates
        self.members = self.candidates
        self.n = len(candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, idx):
        return self.candidates[idx]

    def deep_copy(self):
        return Population([c.deep_copy() for c in self.candidates])

    @classmethod
    def random_population(
        cls,
        dataset,
        population_size,
        tree_depth,
        options,
    ):
        """
        Create random population:
          - generate random trees
          - compute complexity/loss/cost
          - wrap into Candidate
        """
        candidates = []
        for _ in range(population_size):
            tree = generate_random_tree(tree_depth, options)
            cand = Candidate.from_dataset(
                dataset,
                tree,
                parsimony_penalty=options.parsimony_penalty,
                options=options,
            )
            candidates.append(cand)
        return cls(candidates)

    def sample_pop(self, tournament_selection_n):
        """
        Return a new Population with k randomly selected candidates (no replacement).
        """
        k = tournament_selection_n
        if k > self.n:
            raise ValueError("tournament_selection_n cannot exceed population size.")
        sampled = random.sample(self.candidates, k=k)
        return Population(sampled)

    def best_of_sample(
        self,
        options,
        running_stats=None,
    ):
        sample = self.sample_pop(options.tournament_selection_n).candidates

        if options.use_frequency_in_tournament and running_stats is not None:
            adjusted = []
            scale = float(options.adaptive_parsimony_scaling)
            for cand in sample:
                size = cand.complexity
                if 1 <= size <= options.maxsize:
                    freq = running_stats.normalized_frequencies[size - 1]
                else:
                    freq = 0.0
                adjusted.append(cand.cost * math.exp(scale * freq))
            order = sorted(range(len(sample)), key=lambda i: adjusted[i])
        else:
            order = sorted(range(len(sample)), key=lambda i: sample[i].cost)

        p = float(options.tournament_selection_p)
        n = len(sample)
        if p >= 1.0 or n == 1:
            return sample[order[0]].deep_copy()

        weights = []
        for r in range(n):
            weights.append(p * ((1.0 - p) ** r))
        s = sum(weights)
        weights = [w / s for w in weights]

        # choose rank index 0..n-1
        rank = random.choices(range(n), weights=weights, k=1)[0]
        return sample[order[rank]].deep_copy()

    def argmin_birth(self):
        """
        Return index of the oldest candidate (smallest birth timestamp).
        """
        if self.n == 0:
            raise ValueError("Population is empty.")

        oldest_idx = 0
        oldest_birth = self.candidates[0].birth

        for i in range(1, self.n):
            b = self.candidates[i].birth
            if b < oldest_birth:
                oldest_birth = b
                oldest_idx = i

        return oldest_idx

    def argmin_birth_excluding(self, exclude_idx):
        """
        Return index of the oldest candidate excluding `exclude_idx`.
        """
        if self.n <= 1:
            raise ValueError("Population must contain at least 2 candidates.")

        oldest_idx = None
        oldest_birth = None

        for i, c in enumerate(self.candidates):
            if i == exclude_idx:
                continue

            if oldest_idx is None or c.birth < oldest_birth:
                oldest_idx = i
                oldest_birth = c.birth

        if oldest_idx is None:
            raise ValueError("No valid candidate found (check exclude_idx).")

        return oldest_idx
