import pytest
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_editable_module():
    from deepchem.utils.differentiation_utils import EditableModule

    class A(EditableModule):

        def __init__(self, a):
            self.b = a * a

        def mult(self, x):
            return self.b * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix + "b"]
            else:
                raise KeyError()

    a = torch.tensor(2.0).requires_grad_()
    x = torch.tensor(0.4).requires_grad_()
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(1.6000)
    assert alpha.getparamnames("mult") == ['b']
