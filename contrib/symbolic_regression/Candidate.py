import time
from Complexity import calculate_complexity
from LossAndCost import calculate_loss_and_cost

class Candidate:
    def __init__(self, tree, cost, loss, birth, complexity):
        self.tree = tree
        self.cost = cost
        self.loss = loss
        self.birth = birth
        self.complexity = complexity

    @classmethod
    def from_dataset(
        cls,
        dataset,
        tree,
        parsimony_penalty,
        *,
        options=None,
        birth=None,
    ):
        """
        Constructor: takes (dataset, tree) and computes:
          complexity, loss, cost, and sets birth.
        """
        complexity = calculate_complexity(tree, options)
        loss, cost = calculate_loss_and_cost(
            complexity, dataset, tree, parsimony_penalty, options=options
        )

        if birth is None:
            birth = time.time_ns()

        return cls(
            tree=tree,
            cost=cost,
            loss=loss,
            birth=int(birth),
            complexity=int(complexity),
        )

    @classmethod
    def from_values(
        cls,
        tree,
        cost,
        loss,
        complexity,
        *,
        birth=None,
    ):
        """
        Direct constructor:
          takes tree, cost, loss, complexity.
        Birth is auto-generated if not provided.
        """
        if birth is None:
            birth = time.time_ns()

        return cls(
            tree=tree,
            cost=float(cost),
            loss=float(loss),
            birth=int(birth),
            complexity=int(complexity),
        )

    def deep_copy(self):
        """
        Deep copy:
          - deep copy tree
          - copy scalar fields
        """
        return Candidate(
            tree=self.tree.clone(),
            cost=float(self.cost),
            loss=float(self.loss),
            birth=int(self.birth),
            complexity=int(self.complexity),
        )

    def copy(self):
        return self.deep_copy()
