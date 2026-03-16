"""
Genetic Algorithm for Symbolic Regression.

Author: Nandini A R
Date: March 2, 2026
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
import random
from typing import List, Tuple, Optional

# FIXED IMPORTS - use deepchem module path
from deepchem.models.symbolic_regression.expression_tree import (
    ExpressionNode, ExpressionTree, NodeType, Operator,
    make_variable, make_constant, make_operator
)
from deepchem.models.symbolic_regression.fitness import FitnessFunction


class GeneticProgramming:
    """
    Genetic algorithm for symbolic regression.
    Evolves mathematical expressions to fit data.
    """

    def __init__(
        self,
        population_size: int = 100,
        max_depth: int = 5,
        tournament_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        complexity_weight: float = 0.001,
        operators: Optional[List[Operator]] = None,
        n_features: int = 6,
        task: str = "regression",
    ):
        self.population_size = population_size
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.complexity_weight = complexity_weight
        self.n_features = n_features
        self.task = task

        if operators is None:
            self.operators = [
                Operator.ADD,
                Operator.SUB,
                Operator.MUL,
                Operator.DIV,
            ]
        else:
            self.operators = operators

        self.binary_ops = [op for op in self.operators if op.arity == 2]
        self.unary_ops = [op for op in self.operators if op.arity == 1]

    def generate_random_tree(self,
                             max_depth: int,
                             method: str = "grow") -> ExpressionNode:
        if max_depth == 1 or (method == "grow" and random.random() < 0.3):
            if random.random() < 0.7:
                return make_variable(
                    feature_index=random.randint(0, self.n_features - 1))
            else:
                return make_constant(random.uniform(-5, 5))

        if random.random() < 0.7 and self.binary_ops:
            op = random.choice(self.binary_ops)
            left = self.generate_random_tree(max_depth - 1, method)
            right = self.generate_random_tree(max_depth - 1, method)
            return make_operator(op, left, right)
        elif self.unary_ops:
            op = random.choice(self.unary_ops)
            child = self.generate_random_tree(max_depth - 1, method)
            return make_operator(op, child)
        else:
            return make_variable(
                feature_index=random.randint(0, self.n_features - 1))

    def initialize_population(self) -> List[ExpressionTree]:
        population = []
        half = self.population_size // 2

        for _ in range(half):
            depth = random.randint(2, self.max_depth)
            root = self.generate_random_tree(depth, method="grow")
            population.append(ExpressionTree(root))

        for _ in range(self.population_size - half):
            depth = random.randint(2, self.max_depth)
            root = self.generate_random_tree(depth, method="full")
            population.append(ExpressionTree(root))

        return population

    def tournament_selection(self, population: List[ExpressionTree],
                             fitnesses: List[float]) -> ExpressionTree:
        tournament_indices = random.sample(range(len(population)),
                                           self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(
            min(tournament_fitnesses))]
        return population[winner_idx]

    def mutate(self, node: ExpressionNode, depth: int = 0) -> ExpressionNode:
        if random.random() < 0.1:
            return self.generate_random_tree(self.max_depth - depth,
                                             method="grow")

        if node.node_type == NodeType.CONSTANT:
            new_value = node.value + random.gauss(0, 0.5)
            return make_constant(new_value)

        elif node.node_type == NodeType.VARIABLE:
            return make_variable(
                feature_index=random.randint(0, self.n_features - 1))

        elif node.node_type == NodeType.OPERATOR:
            left_mutated = self.mutate(node.left, depth +
                                       1) if node.left else None
            right_mutated = self.mutate(node.right, depth +
                                        1) if node.right else None

            new_op = node.operator
            if random.random() < 0.2:
                if node.operator.arity == 2 and self.binary_ops:
                    new_op = random.choice(self.binary_ops)
                elif node.operator.arity == 1 and self.unary_ops:
                    new_op = random.choice(self.unary_ops)

            return make_operator(new_op, left_mutated, right_mutated)

        else:
            return node.copy()

    def crossover(
            self, parent1: ExpressionNode,
            parent2: ExpressionNode) -> Tuple[ExpressionNode, ExpressionNode]:
        child1 = parent1.copy()
        child2 = parent2.copy()
        return child1, child2

    def evolve(self,
               x_train: torch.Tensor,
               y_train: torch.Tensor,
               generations: int = 50,
               verbose: bool = True) -> Tuple[ExpressionTree, dict]:
        population = self.initialize_population()
        fitness_fn = FitnessFunction(x_train, y_train, self.complexity_weight, task=self.task)

        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_expression': [],
        }

        if verbose:
            print(f"Generation 0/{generations}")
            print(f"Population size: {self.population_size}")
            print("-" * 60)

        for gen in range(generations):
            fitnesses = []
            for tree in population:
                fitness, _ = fitness_fn.evaluate(tree)
                fitnesses.append(fitness)

            best_idx = fitnesses.index(min(fitnesses))
            best_fitness = fitnesses[best_idx]
            best_tree = population[best_idx]

            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(sum(fitnesses) / len(fitnesses))
            history['best_expression'].append(str(best_tree))

            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {best_fitness:.6f}")
                print(f"         Best expr = {best_tree}")
                print()

            new_population = []
            new_population.append(ExpressionTree(best_tree.root.copy()))

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)

                if random.random() < self.crossover_rate:
                    child1_root, child2_root = self.crossover(
                        parent1.root, parent2.root)
                else:
                    child1_root = parent1.root.copy()
                    child2_root = parent2.root.copy()

                if random.random() < self.mutation_rate:
                    child1_root = self.mutate(child1_root)
                if random.random() < self.mutation_rate:
                    child2_root = self.mutate(child2_root)

                new_population.append(ExpressionTree(child1_root))
                if len(new_population) < self.population_size:
                    new_population.append(ExpressionTree(child2_root))

            population = new_population

        fitnesses = [fitness_fn.evaluate(tree)[0] for tree in population]
        best_idx = fitnesses.index(min(fitnesses))
        best_tree = population[best_idx]

        if verbose:
            print("-" * 60)
            print("FINAL RESULT:")
            print(f"Best fitness: {fitnesses[best_idx]:.6f}")
            print(f"Best expression: {best_tree}")
            print(f"Complexity: {best_tree.complexity()}")

        return best_tree, history