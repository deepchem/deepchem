import pytest
try:
    import torch
    from deepchem.utils.differentiation_utils import EditableModule
    has_torch = True
except ModuleNotFoundError:
    has_torch = False


@pytest.mark.torch
def test_getparams():

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

    a = torch.tensor(2.0)
    x = torch.tensor(0.4)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(1.6)
    assert alpha.getparams("mult") == [torch.tensor(4.)]


@pytest.mark.torch
def test_setparams():

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

    a = torch.tensor(2.0)
    x = torch.tensor(4.0)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(16)
    alpha.setparams("mult", torch.tensor(5.0))
    assert alpha.mult(x) == torch.tensor(20.0)


@pytest.mark.torch
def test_cached_getparamnames():

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

    a = torch.tensor(2.0)
    alpha = A(a)
    assert alpha.cached_getparamnames("mult") == ["b"]


@pytest.mark.torch
def test_getuniqueparams():

    class A(EditableModule):

        def __init__(self, a):
            self.b = a * a

        def mult(self, x):
            return self.b**2 * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix + "b"]
            else:
                raise KeyError()

    a = torch.tensor(2.0)
    x = torch.tensor(0.4)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(6.4)
    assert alpha.getuniqueparams("mult") == [torch.tensor(4.)]  # Not 16.0


@pytest.mark.torch
def test_setuniqueparams():

    class A(EditableModule):

        def __init__(self, a):
            self.b = a * a

        def mult(self, x):
            return self.b**2 * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix + "b"]
            else:
                raise KeyError()

    a = torch.tensor(2.0)
    x = torch.tensor(0.4)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(6.4)
    assert alpha.getuniqueparams("mult") == [torch.tensor(4.)]
    alpha.setuniqueparams("mult", torch.tensor(5.0))
    assert alpha.mult(x) == torch.tensor(10.0)


@pytest.mark.torch
def test_get_unique_params_idxs():

    class A(EditableModule):

        def __init__(self, a):
            self.b = a * a
            self.c = a * a * a

        def mult(self, x):
            return self.b * self.c * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix + "b", prefix + "c"]
            else:
                raise KeyError()

    a = torch.tensor(2.0)
    x = torch.tensor(4.0)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(128.0)
    assert alpha.getparams("mult") == [torch.tensor(4.), torch.tensor(8.)]
    assert alpha._get_unique_params_idxs("mult") == [0, 1]


@pytest.mark.torch
def test_assertparams():
    """Test that assertparams works correctly.
    also checks the private methods as they are used in it.
    - __assert_method_preserve
    - __assert_get_correct_params
    - __list_operating_params

    """

    class A(EditableModule):

        def __init__(self, a):
            self.b = a * a
            self.c = a * a * a

        def mult(self, x):
            return self.b * self.c * x

        def getparamnames(self, methodname, prefix=""):
            if methodname == "mult":
                return [prefix + "b", prefix + "c"]
            else:
                raise KeyError()

    a = torch.tensor(2.0)
    x = torch.tensor(4.0)
    alpha = A(a)
    assert alpha.mult(x) == torch.tensor(128.0)
    assert alpha.getparams("mult") == [torch.tensor(4.), torch.tensor(8.)]
    alpha.assertparams(alpha.mult, x)


@pytest.mark.torch
def test_getparamnames():

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


@pytest.mark.torch
def test_normalize_bcast_dims():
    from deepchem.utils.differentiation_utils import normalize_bcast_dims
    assert normalize_bcast_dims([1, 2, 3], [2, 3]) == [[1, 2, 3], [1, 2, 3]]


@pytest.mark.torch
def test_get_bcasted_dims():
    from deepchem.utils.differentiation_utils import get_bcasted_dims
    assert get_bcasted_dims([1, 2, 5], [2, 3, 4]) == [2, 3, 5]


@pytest.mark.torch
def test_match_dim():
    from deepchem.utils.differentiation_utils import match_dim
    x = torch.randn(10, 5)
    xq = torch.randn(10, 3)
    x_new, xq_new = match_dim(x, xq)
    assert x_new.shape == torch.Size([10, 5])
    assert xq_new.shape == torch.Size([10, 3])


def test_set_default_options():
    from deepchem.utils.differentiation_utils import set_default_option
    assert set_default_option({'a': 1, 'b': 2}, {'a': 3}) == {'a': 3, 'b': 2}


def test_get_and_pop_keys():
    from deepchem.utils.differentiation_utils import get_and_pop_keys
    assert get_and_pop_keys({'a': 1, 'b': 2}, ['a']) == {'a': 1}


def test_get_method():
    from deepchem.utils.differentiation_utils import get_method
    assert get_method('foo', {'bar': lambda: 1}, 'bar')() == 1


def test_dummy_context_manager():
    """Just checks that dummy_context_manager doesn't crash"""
    from deepchem.utils.differentiation_utils import dummy_context_manager
    with dummy_context_manager() as x:
        if x is None:
            pass
        else:
            raise AssertionError()


def test_assert_runtime():
    from deepchem.utils.differentiation_utils import assert_runtime
    try:
        assert_runtime(False, "This should fail")
    except RuntimeError:
        pass
