"""
Fitness function for symbolic regression.

Fitness = MSE + complexity_penalty

Author: Anand
Date: March 2, 2025
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
from typing import Tuple
from deepchem.models.symbolic_regression.expression_tree import ExpressionTree


class FitnessFunction:
    """
    Evaluates fitness of symbolic expressions.
    
    Lower fitness is better.
    Fitness = MSE(predictions, targets) + λ * complexity
    
    where:
    - MSE: Mean squared error on training data
    - complexity: Number of nodes in expression tree
    - λ: Complexity weight (default 0.001)
    """
    
    def __init__(
        self, 
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        complexity_weight: float = 0.001
    ):
        """
        Initialize fitness function.
        
        Args:
            x_train: Training inputs, shape (n_samples,)
            y_train: Training targets, shape (n_samples,)
            complexity_weight: Weight for complexity penalty
        """
        self.x_train = x_train
        self.y_train = y_train
        self.complexity_weight = complexity_weight
        
        # For normalization
        self.y_std = y_train.std()
        if self.y_std < 1e-8:
            self.y_std = 1.0
    
    def evaluate(self, tree: ExpressionTree) -> Tuple[float, dict]:
        """
        Evaluate fitness of expression tree.
        
        Args:
            tree: Expression tree to evaluate
        
        Returns:
            fitness: Scalar fitness value (lower is better)
            metrics: Dictionary with detailed metrics
        """
        try:
            # Get predictions
            y_pred = tree.evaluate(self.x_train)
            
            # Calculate MSE
            mse = torch.mean((y_pred - self.y_train) ** 2).item()
            
            # Normalize by target standard deviation
            normalized_mse = mse / (self.y_std ** 2)
            
            # Get complexity
            complexity = tree.complexity()
            
            # Calculate total fitness
            fitness = normalized_mse + self.complexity_weight * complexity
            
            # Additional metrics
            metrics = {
                'fitness': fitness,
                'mse': mse,
                'normalized_mse': normalized_mse,
                'complexity': complexity,
                'complexity_penalty': self.complexity_weight * complexity,
            }
            
            return fitness, metrics
            
        except Exception as e:
            # If evaluation fails (e.g., numerical instability),
            # return very high fitness
            return float('inf'), {
                'fitness': float('inf'),
                'mse': float('inf'),
                'normalized_mse': float('inf'),
                'complexity': tree.complexity(),
                'error': str(e)
            }
    
    def evaluate_batch(self, trees: list) -> list:
        """
        Evaluate fitness for multiple trees.
        
        Args:
            trees: List of ExpressionTree objects
        
        Returns:
            results: List of (fitness, metrics) tuples
        """
        results = []
        for tree in trees:
            fitness, metrics = self.evaluate(tree)
            results.append((fitness, metrics))
        return results


# Test the fitness function
if __name__ == "__main__":
    from expression import (
        ExpressionTree, make_variable, make_constant, 
        make_operator, Operator
    )
    import numpy as np
    
    print("=" * 60)
    print("FITNESS FUNCTION TESTS")
    print("=" * 60)
    print()
    
    # Generate test data: y = 2*x + 3
    print("=== Test Data: y = 2*x + 3 ===")
    x_train = torch.linspace(-5, 5, 50)
    y_train = 2 * x_train + 3
    print(f"Training samples: {len(x_train)}")
    print(f"X range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print()
    
    # Create fitness function
    fitness_fn = FitnessFunction(x_train, y_train, complexity_weight=0.001)
    
    # Test 1: Perfect match (y = 2*x + 3)
    print("=== Test 1: Perfect Match (y = 2*x + 3) ===")
    x_node = make_variable()
    const_2 = make_constant(2.0)
    const_3 = make_constant(3.0)
    mul_node = make_operator(Operator.MUL, const_2, x_node)
    add_node = make_operator(Operator.ADD, mul_node, const_3)
    perfect_tree = ExpressionTree(add_node)
    
    fitness, metrics = fitness_fn.evaluate(perfect_tree)
    print(f"Expression: {perfect_tree}")
    print(f"Fitness: {fitness:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Complexity: {metrics['complexity']}")
    print(f"Expected: Near-zero fitness (perfect match)")
    print()
    
    # Test 2: Wrong slope (y = 3*x + 3)
    print("=== Test 2: Wrong Slope (y = 3*x + 3) ===")
    const_3_wrong = make_constant(3.0)
    mul_node_wrong = make_operator(Operator.MUL, const_3_wrong, make_variable())
    add_node_wrong = make_operator(Operator.ADD, mul_node_wrong, make_constant(3.0))
    wrong_tree = ExpressionTree(add_node_wrong)
    
    fitness, metrics = fitness_fn.evaluate(wrong_tree)
    print(f"Expression: {wrong_tree}")
    print(f"Fitness: {fitness:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Complexity: {metrics['complexity']}")
    print(f"Expected: Higher fitness (wrong slope)")
    print()
    
    # Test 3: Too complex (y = sin(2*x) + cos(x) + 3)
    print("=== Test 3: Complex Expression ===")
    x1 = make_variable()
    x2 = make_variable()
    const_2 = make_constant(2.0)
    mul = make_operator(Operator.MUL, const_2, x1)
    sin_node = make_operator(Operator.SIN, mul)
    cos_node = make_operator(Operator.COS, x2)
    add1 = make_operator(Operator.ADD, sin_node, cos_node)
    add2 = make_operator(Operator.ADD, add1, make_constant(3.0))
    complex_tree = ExpressionTree(add2)
    
    fitness, metrics = fitness_fn.evaluate(complex_tree)
    print(f"Expression: {complex_tree}")
    print(f"Fitness: {fitness:.6f}")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Complexity: {metrics['complexity']}")
    print(f"Complexity penalty: {metrics['complexity_penalty']:.6f}")
    print(f"Expected: High fitness (complex + poor fit)")
    print()
    
    # Test 4: Compare different expressions
    print("=== Test 4: Ranking Multiple Expressions ===")
    trees = [
        (perfect_tree, "Perfect: 2*x + 3"),
        (wrong_tree, "Wrong slope: 3*x + 3"),
        (complex_tree, "Complex: sin(2*x) + cos(x) + 3"),
    ]
    
    results = []
    for tree, name in trees:
        fitness, _ = fitness_fn.evaluate(tree)
        results.append((fitness, name))
    
    # Sort by fitness (lower is better)
    results.sort(key=lambda x: x[0])
    
    print("Ranked by fitness (best to worst):")
    for i, (fitness, name) in enumerate(results, 1):
        print(f"{i}. {name}: {fitness:.6f}")
    print()
    
    print("=" * 60)
    print("FITNESS FUNCTION TESTS COMPLETED")
    print("=" * 60)