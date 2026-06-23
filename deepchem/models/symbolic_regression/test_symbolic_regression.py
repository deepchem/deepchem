"""Basic tests for symbolic regression"""
import torch
from deepchem.models.symbolic_regression import GeneticProgramming

def test_basic_evolution():
    """Test that GP can evolve"""
    X = torch.randn(100, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2]
    
    gp = GeneticProgramming(population_size=20, n_features=3)
    best, _ = gp.evolve(X, y, generations=5, verbose=False)
    
    assert best is not None
    assert best.complexity() > 0
    print("✅ test_basic_evolution passed!")

def test_evaluation():
    """Test tree evaluation"""
    from deepchem.models.symbolic_regression.expression_tree import (
        ExpressionTree, make_variable, make_constant, make_operator, Operator
    )
    
    # Build: y = 2*x + 3
    x = make_variable(feature_index=0)
    const_2 = make_constant(2.0)
    const_3 = make_constant(3.0)
    mul = make_operator(Operator.MUL, const_2, x)
    add = make_operator(Operator.ADD, mul, const_3)
    tree = ExpressionTree(add)
    
    X = torch.tensor([[1.0], [2.0], [3.0]])
    y_pred = tree.evaluate(X)
    y_expected = torch.tensor([5.0, 7.0, 9.0])
    
    assert torch.allclose(y_pred, y_expected, atol=1e-5)
    print("✅ test_evaluation passed!")

if __name__ == "__main__":
    test_basic_evolution()
    test_evaluation()
    print("\n🎉 All tests passed!")