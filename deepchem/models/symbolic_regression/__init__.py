"""
Symbolic Regression for DeepChem
"""

from deepchem.models.symbolic_regression.symbolic_regressor import GeneticProgramming
from deepchem.models.symbolic_regression.expression_tree import ExpressionTree
from deepchem.models.symbolic_regression.descriptors import compute_descriptors

__all__ = ['GeneticProgramming', 'ExpressionTree', 'compute_descriptors']