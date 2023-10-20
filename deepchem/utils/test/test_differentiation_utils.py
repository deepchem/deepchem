import pytest
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_traverse_obj():
    from deepchem.utils.differentiation_utils.editable_module import _traverse_obj, torch_float_type

    class A:

        def __init__(self):
            self.a = 2
            self.b = torch.tensor(3.0)
            self.c = torch.tensor(4.0)
            self.d = torch.tensor(5.0)

    a = A()

    def action(elmt, name, objdict, key):
        print(name, elmt)

    def crit(elmt):
        return isinstance(elmt, torch.Tensor) and elmt.dtype in torch_float_type

    a = _traverse_obj(a, "", action, crit)  # Check Doesn't Crashes


@pytest.mark.torch
def test_get_tensor():
    from deepchem.utils.differentiation_utils.editable_module import _get_tensors

    class A:

        def __init__(self):
            self.a = 2
            self.b = torch.tensor(3.0)
            self.c = torch.tensor(4.0)
            self.d = torch.tensor(5.0)

    a = A()
    outputs = _get_tensors(a)
    assert outputs[0] == [torch.tensor(3.), torch.tensor(4.), torch.tensor(5.)]
    assert outputs[1] == ['b', 'c', 'd']


@pytest.mark.torch
def test_set_tensor():
    from deepchem.utils.differentiation_utils.editable_module import _set_tensors

    class A:

        def __init__(self):
            self.a = 2
            self.b = torch.tensor(3.0)
            self.c = torch.tensor(4.0)
            self.d = torch.tensor(5.0)

    a = A()
    _set_tensors(a, [torch.tensor(6.), torch.tensor(7.), torch.tensor(8.)])
    assert a.b == torch.tensor(6.)
    assert a.c == torch.tensor(7.)
