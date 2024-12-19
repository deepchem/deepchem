import pytest
import numpy as np
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


@pytest.mark.torch
def test_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator
    torch.manual_seed(100)

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

        def _rmv(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _mm(self, x):
            return torch.matmul(self.param, x)

        def _rmm(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

        def _fullmatrix(self):
            return self.param

    linop = MyLinOp((1, 3, 1, 2))
    x = torch.rand(1, 3, 2, 2)
    assert torch.allclose(linop.mv(x), torch.matmul(linop.param, x))
    x = torch.rand(1, 3, 1, 1)
    assert torch.allclose(linop.rmv(x),
                          torch.matmul(linop.param.transpose(-2, -1).conj(), x))
    x = torch.rand(1, 3, 2, 2)
    assert torch.allclose(linop.mm(x), torch.matmul(linop.param, x))
    x = torch.rand(1, 3, 1, 2)
    assert torch.allclose(linop.rmm(x),
                          torch.matmul(linop.param.transpose(-2, -1).conj(), x))
    assert torch.allclose(linop.fullmatrix(), linop.param)


@pytest.mark.torch
def test_add_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator

    class Operator(LinearOperator):

        def __init__(self, mat: torch.Tensor, is_hermitian: bool) -> None:
            super(Operator, self).__init__(
                shape=mat.shape,
                is_hermitian=is_hermitian,
                dtype=mat.dtype,
                device=mat.device,
                _suppress_hermit_warning=True,
            )
            self.mat = mat

        def _mv(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(self.mat, x.unsqueeze(-1)).squeeze(-1)

        def _mm(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(self.mat, x)

        def _rmv(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(
                self.mat.transpose(-3, -1).conj(), x.unsqueeze(-1)).squeeze(-1)

        def _rmm(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(self.mat.transpose(-2, -1).conj(), x)

        def _fullmatrix(self) -> torch.Tensor:
            return self.mat

        def _getparamnames(self, prefix: str = ""):
            return [prefix + "mat"]

    op = Operator(torch.tensor([[1, 2.], [3, 4]]), is_hermitian=False)
    x = torch.tensor([[2, 2], [1, 2.]])
    op2 = op + op
    assert torch.allclose(op2.mm(x), 2 * op.mm(x))


@pytest.mark.torch
def test_mul_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

    linop = MyLinOp((1, 3, 1, 2))
    linop2 = linop * 2
    x = torch.rand(1, 3, 2, 2)
    torch.allclose(linop.mv(x) * 2, linop2.mv(x))


@pytest.mark.torch
def test_adjoint_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

        def _rmv(self, x):
            return torch.matmul(self.param.transpose(-2, -1).conj(), x)

    linop = MyLinOp((1, 3, 1, 2))
    x = torch.rand(1, 3, 1, 1)
    result_rmv = linop.rmv(x)

    adjoint_linop = linop.H
    result_mv = adjoint_linop.mv(x)

    assert torch.allclose(result_rmv, result_mv)


@pytest.mark.torch
def test_matmul_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator

    class MyLinOp(LinearOperator):

        def __init__(self, shape):
            super(MyLinOp, self).__init__(shape)
            self.param = torch.rand(shape)

        def _getparamnames(self, prefix=""):
            return [prefix + "param"]

        def _mv(self, x):
            return torch.matmul(self.param, x)

    linop1 = MyLinOp((1, 3, 1, 2))
    linop2 = MyLinOp((1, 3, 2, 1))
    linop_result = linop1.matmul(linop2)
    x = torch.rand(1, 3, 1, 1)
    result = linop_result.mv(x)
    assert result.shape == torch.Size([1, 3, 1, 1])


@pytest.mark.torch
def test_matrix_linear_operator():
    from deepchem.utils.differentiation_utils import LinearOperator

    mat = torch.rand(2, 2)
    linop = LinearOperator.m(mat)
    x = torch.randn(2, 2)

    result_mm = linop.mm(x)
    expected_mm = torch.matmul(mat, x)

    result_mv = linop.mv(x)
    expected_mv = torch.matmul(mat, x.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(result_mm, expected_mm)
    assert torch.allclose(result_mv, expected_mv)


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


@pytest.mark.torch
def test_take_eigpairs():
    from deepchem.utils.differentiation_utils.symeig import _take_eigpairs
    eival = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    eivec = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                          [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    neig = 2
    mode = "lowest"
    eival, eivec = _take_eigpairs(eival, eivec, neig, mode)
    assert torch.allclose(eival, torch.tensor([[1., 2.], [4., 5.]]))
    assert torch.allclose(
        eivec,
        torch.tensor([[[1., 2.], [4., 5.], [7., 8.]],
                      [[1., 2.], [4., 5.], [7., 8.]]]))


@pytest.mark.torch
def test_exacteig():
    from deepchem.utils.differentiation_utils.symeig import exacteig
    from deepchem.utils.differentiation_utils import LinearOperator
    torch.manual_seed(100)
    mat = LinearOperator.m(torch.randn(2, 2))
    eival, eivec = exacteig(mat, 2, "lowest", None)
    assert eival.shape == torch.Size([2])
    assert eivec.shape == torch.Size([2, 2])


@pytest.mark.torch
def test_degen_symeig():
    from deepchem.utils.differentiation_utils.symeig import degen_symeig
    from deepchem.utils.differentiation_utils import LinearOperator
    A = LinearOperator.m(torch.rand(2, 2))
    evals, evecs = degen_symeig.apply(A.fullmatrix())
    assert evals.shape == torch.Size([2])
    assert evecs.shape == torch.Size([2, 2])


@pytest.mark.torch
def test_davidson():
    from deepchem.utils.differentiation_utils.symeig import davidson
    from deepchem.utils.differentiation_utils import LinearOperator
    A = LinearOperator.m(torch.rand(2, 2))
    neig = 2
    mode = "lowest"
    eigen_val, eigen_vec = davidson(A, neig, mode)
    assert eigen_val.shape == torch.Size([2])
    assert eigen_vec.shape == torch.Size([2, 2])


@pytest.mark.torch
def test_pure_function():
    from deepchem.utils.differentiation_utils import PureFunction

    class WrapperFunction(PureFunction):

        def _get_all_obj_params_init(self):
            return []

        def _set_all_obj_params(self, objparams):
            pass

    def fcn(x, y):
        return x + y

    pfunc = WrapperFunction(fcn)
    assert pfunc(1, 2) == 3


@pytest.mark.torch
def test_function_pure_function():
    from deepchem.utils.differentiation_utils.pure_function import FunctionPureFunction

    def fcn(x, y):
        return x + y

    pfunc = FunctionPureFunction(fcn)
    assert pfunc(1, 2) == 3


@pytest.mark.torch
def test_editable_module_pure_function():
    from deepchem.utils.differentiation_utils import EditableModule
    from deepchem.utils.differentiation_utils.pure_function import EditableModulePureFunction

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

    B = A(4)
    m = EditableModulePureFunction(B, B.mult)
    m.set_objparams([3])
    assert m(2) == 6


@pytest.mark.torch
def test_torch_nn_pure_function():
    from deepchem.utils.differentiation_utils import get_pure_function

    class A(torch.nn.Module):

        def __init__(self, a):
            super().__init__()
            self.b = torch.nn.Parameter(torch.tensor(a * a))

        def forward(self, x):
            return self.b * x

    B = A(4.)
    m = get_pure_function(B.forward)
    m.set_objparams([3.])
    assert m(2) == 6.0


@pytest.mark.torch
def test_check_identical_objs():
    from deepchem.utils.differentiation_utils.pure_function import _check_identical_objs
    a = [1, 2, 3]
    assert _check_identical_objs([a], [a])


@pytest.mark.torch
def test_get_pure_function():
    from deepchem.utils.differentiation_utils import get_pure_function

    def fcn(x, y):
        return x + y

    pfunc = get_pure_function(fcn)
    assert pfunc(1, 2) == 3


@pytest.mark.torch
def test_make_siblings():
    from deepchem.utils.differentiation_utils import make_sibling

    def fcn1(x, y):
        return x + y

    @make_sibling(fcn1)
    def fcn3(x, y):
        return x * y

    assert fcn3(1, 2) == 2


@pytest.mark.torch
def test_wrap_gmres():
    from deepchem.utils.differentiation_utils.solve import wrap_gmres
    from deepchem.utils.differentiation_utils import LinearOperator
    A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    B = torch.tensor([[[5., 6], [7, 8]]])
    assert torch.allclose(A.fullmatrix() @ wrap_gmres(A, B, None, None), B)


@pytest.mark.torch
def test_exact_solve():
    from deepchem.utils.differentiation_utils.solve import exactsolve
    from deepchem.utils.differentiation_utils import LinearOperator
    A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    B = torch.tensor([[5., 6], [7, 8]])
    assert torch.allclose(A.fullmatrix() @ exactsolve(A, B, None, None), B)


@pytest.mark.torch
def test_solve_ABE():
    from deepchem.utils.differentiation_utils.solve import solve_ABE
    A = torch.tensor([[1., 2], [3, 4]])
    B = torch.tensor([[5., 6], [7, 8]])
    E = torch.tensor([1., 2])
    expected_result = torch.tensor([[-0.1667, 0.5000], [2.5000, 3.2500]])
    assert torch.allclose(solve_ABE(A, B, E), expected_result, 0.001)


@pytest.mark.torch
def test_get_batch_dims():
    from deepchem.utils.differentiation_utils.solve import get_batchdims
    from deepchem.utils.differentiation_utils import MatrixLinearOperator
    A = MatrixLinearOperator(torch.randn(4, 3, 3), True)
    B = torch.randn(3, 3, 2)
    assert get_batchdims(A, B, None,
                         None) == [max(A.shape[:-2], B.shape[:-2])[0]]


@pytest.mark.torch
def test_setup_precond():
    from deepchem.utils.differentiation_utils.solve import setup_precond
    from deepchem.utils.differentiation_utils import MatrixLinearOperator
    A = MatrixLinearOperator(torch.randn(4, 3, 3), True)
    B = torch.randn(4, 3, 2)
    cond = setup_precond(A)
    assert cond(B).shape == torch.Size([4, 3, 2])


@pytest.mark.torch
def test_dot():
    from deepchem.utils.differentiation_utils.solve import dot
    r = torch.tensor([[1, 2], [3, 4]])
    z = torch.tensor([[5, 6], [7, 8]])
    assert torch.allclose(dot(r, z), torch.tensor([[26, 44]]))
    assert torch.allclose(dot(r, z), sum(r * z))


@pytest.mark.torch
def test_gmres():
    from deepchem.utils.differentiation_utils.solve import gmres
    from deepchem.utils.differentiation_utils import MatrixLinearOperator
    A = MatrixLinearOperator(torch.tensor([[1., 2], [3, 4]]), True)
    B = torch.tensor([[5., 6], [7, 8]])
    expected_result = torch.tensor([[0.8959, 1.0697], [1.2543, 1.4263]])
    assert torch.allclose(gmres(A, B), expected_result, 0.001)


@pytest.mark.torch
def test_setup_linear_problem():
    from deepchem.utils.differentiation_utils import MatrixLinearOperator
    from deepchem.utils.differentiation_utils.solve import setup_linear_problem
    A = MatrixLinearOperator(torch.randn(4, 3, 3), True)
    B = torch.randn(4, 3, 2)
    A_fcn, AT_fcn, B_new, col_swapped = setup_linear_problem(
        A, B, None, None, [4], None, False)
    assert A_fcn(B).shape == torch.Size([4, 3, 2])


@pytest.mark.torch
def test_safe_denom():
    from deepchem.utils.differentiation_utils.solve import safedenom
    r = torch.tensor([[0., 2], [3, 4]])
    assert torch.allclose(
        safedenom(r, 1e-9),
        torch.tensor([[1.0000e-09, 2.0000e+00], [3.0000e+00, 4.0000e+00]]))


@pytest.mark.torch
def test_get_largest_eival():
    from deepchem.utils.differentiation_utils.solve import get_largest_eival

    def Afcn(x):
        return 10 * x

    x = torch.tensor([[1., 2], [3, 4]])
    assert torch.allclose(get_largest_eival(Afcn, x), torch.tensor([[10.,
                                                                     10.]]))


@pytest.mark.torch
def test_broyden1_solve():
    from deepchem.utils.differentiation_utils.solve import broyden1_solve
    A = torch.tensor([[1., 2], [3, 4]])
    B = torch.tensor([[5., 6], [7, 8]])
    assert torch.allclose(broyden1_solve(A, B),
                          torch.tensor([[-3.0000, -4.0000], [4.0000, 5.0000]]))


@pytest.mark.torch
def test_rootfinder_solve():
    from deepchem.utils.differentiation_utils.solve import _rootfinder_solve
    A = torch.tensor([[1., 2], [3, 4]])
    B = torch.tensor([[5., 6], [7, 8]])
    assert torch.allclose(_rootfinder_solve("broyden1", A, B),
                          torch.tensor([[-3.0000, -4.0000], [4.0000, 5.0000]]))


@pytest.mark.torch
def test_symeig():
    from deepchem.utils.differentiation_utils import LinearOperator, symeig
    A = LinearOperator.m(torch.tensor([[3, -1j], [1j, 4]]))
    evals, evecs = symeig(A)
    assert evecs.shape == torch.Size([2, 2])
    assert evals.shape == torch.Size([2])


@pytest.mark.torch
def test_check_degen():
    from deepchem.utils.differentiation_utils.symeig import _check_degen
    evals = torch.tensor([1, 1, 2, 3, 3, 3, 4, 5, 5])
    degen_atol = 0.1
    degen_rtol = 0.1
    idx_degen, isdegenerate = _check_degen(evals, degen_atol, degen_rtol)
    assert idx_degen.shape == torch.Size([9, 9])


@pytest.mark.torch
def test_ortho():
    from deepchem.utils.differentiation_utils import ortho
    A = torch.tensor([[1, 2], [3, 4]])
    B = torch.tensor([[1, 0], [0, 1]])
    assert torch.allclose(ortho(A, B), torch.tensor([[0, 2], [3, 0]]))


@pytest.mark.torch
def test_jac():
    from deepchem.utils.differentiation_utils import jac

    def fcn(x, y):
        return x * y

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    n_lin = jac(fcn, [x, y])
    input = torch.tensor([[1, 3, 3], [4, 5, 6]])
    assert torch.allclose(n_lin[1].mv(input), x * input)


@pytest.mark.torch
def test_jac_class():
    from deepchem.utils.differentiation_utils.grad import get_pure_function, _Jac

    def fcn(x, y):
        return x * y

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    pfcn = get_pure_function(fcn)
    assert torch.allclose(
        _Jac(pfcn, [x, y], 0).mv(torch.tensor([1.0, 1.0, 1.0])), y)


@pytest.mark.torch
def test_connect_graph():
    from deepchem.utils.differentiation_utils.grad import connect_graph
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    out = x * y
    assert torch.allclose(connect_graph(out, [x, y]),
                          torch.tensor([4., 10., 18.]))


@pytest.mark.torch
def test_setup_idxs():
    from deepchem.utils.differentiation_utils.grad import _setup_idxs
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    assert _setup_idxs(None, [x, y]) == [0, 1]


@pytest.mark.torch
def test_solve():
    from deepchem.utils.differentiation_utils import LinearOperator, solve
    A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    B = torch.tensor([[5., 6], [7, 8]])
    assert torch.allclose(solve(A, B),
                          torch.tensor([[-3.0000, -4.0000], [4.0000, 5.0000]]))


@pytest.mark.torch
def test_cg():
    from deepchem.utils.differentiation_utils import LinearOperator, cg
    A = LinearOperator.m(torch.tensor([[1., 2], [3, 4]]))
    B = torch.tensor([[5., 6], [7, 8]])
    assert torch.allclose(cg(A, B),
                          torch.tensor([[-3.0000, -4.0000], [4.0000, 5.0000]]))


@pytest.mark.torch
def test_svd():
    from deepchem.utils.differentiation_utils import LinearOperator, svd
    A = LinearOperator.m(torch.tensor([[3, 1], [1, 4.]]))
    U, S, _ = svd(A)
    assert torch.allclose(torch.tensor([[-0.8507, 0.5257], [0.5257, 0.8507]]),
                          U, 0.001)
    assert torch.allclose(torch.tensor([2.3820, 4.6180]), S, 0.001)


@pytest.mark.torch
def test_BroydenFirst():
    from deepchem.utils.differentiation_utils.optimize.jacobian import BroydenFirst
    jacobian = BroydenFirst()
    x0 = torch.tensor([1.0, 1.0], requires_grad=True)

    def func(x):
        return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])

    y0 = func(x0)
    v = torch.tensor([1.0, 1.0])
    jacobian.setup(x0, y0, func)
    assert torch.allclose(jacobian.solve(v), torch.tensor([-0.7071, -0.7071]))


@pytest.mark.torch
def test_BroydenSecond():
    from deepchem.utils.differentiation_utils.optimize.jacobian import BroydenSecond
    jacobian = BroydenSecond()
    x0 = torch.tensor([1.0, 1.0], requires_grad=True)

    def func(x):
        return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])

    y0 = func(x0)
    v = torch.tensor([1.0, 1.0])
    jacobian.setup(x0, y0, func)
    assert torch.allclose(jacobian.solve(v), torch.tensor([-0.7071, -0.7071]))


@pytest.mark.torch
def test_LinearMixing():
    from deepchem.utils.differentiation_utils.optimize.jacobian import LinearMixing
    jacobian = LinearMixing()
    x0 = torch.tensor([1.0, 1.0], requires_grad=True)

    def func(x):
        return torch.tensor([x[0]**2 + x[1]**2 - 1.0, x[0] - x[1]])

    y0 = func(x0)
    v = torch.tensor([1.0, 1.0])
    jacobian.setup(x0, y0, func)
    assert torch.allclose(jacobian.solve(v), torch.tensor([1., 1.]))


@pytest.mark.torch
def test_low_rank_matrix():
    from deepchem.utils.differentiation_utils.optimize.jacobian import LowRankMatrix
    import torch
    alpha = 1.0
    uv0 = (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    reduce_method = "restart"
    matrix = LowRankMatrix(alpha, uv0, reduce_method)
    v = torch.tensor([1.0, 1.0])
    assert torch.allclose(matrix.mv(v), torch.tensor([3., 3.]))


@pytest.mark.torch
def test_full_rank_matrix():
    from deepchem.utils.differentiation_utils.optimize.jacobian import FullRankMatrix
    alpha = 1.0
    cns = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    dns = [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])]
    matrix = FullRankMatrix(alpha, cns, dns)
    v = torch.tensor([1.0, 1.0])
    assert torch.allclose(matrix.mv(v), torch.tensor([5., 5.]))
    assert torch.allclose(matrix.rmv(v), torch.tensor([5., 5.]))


@pytest.mark.torch
def test_gd():
    from deepchem.utils.differentiation_utils.optimize.minimizer import gd

    def fcn(x):
        return (x - 2)**2, 2 * (x - 2)

    x0 = torch.tensor(0.0, requires_grad=True)
    x = gd(fcn, x0, [])
    assert torch.allclose(x, torch.tensor(2.0000))


@pytest.mark.torch
def test_adam():
    from deepchem.utils.differentiation_utils.optimize.minimizer import adam

    def fcn(x):
        return (x - 2)**2, 2 * (x - 2)

    x0 = torch.tensor(0.0, requires_grad=True)
    x = adam(fcn, x0, [], maxiter=10000)
    assert torch.allclose(x, torch.tensor(2.0000))


@pytest.mark.torch
def test_termination_condition():
    from deepchem.utils.differentiation_utils.optimize.minimizer import TerminationCondition
    stop_cond = TerminationCondition(1e-8, 1e-8, 1e-8, 1e-8, True)
    assert not stop_cond.to_stop(0, torch.tensor(0.0), torch.tensor(0.0),
                                 torch.tensor(0.0), torch.tensor(0.0))


@pytest.mark.torch
def test_anderson_acc():
    from deepchem.utils.differentiation_utils.optimize.equilibrium import anderson_acc

    def fcn(x):
        return x

    x0 = torch.tensor([0.0], requires_grad=True)
    x = anderson_acc(fcn, x0, [])
    assert torch.allclose(x, torch.tensor([0.]))


@pytest.mark.torch
def test_broyden1():
    from deepchem.utils.differentiation_utils.optimize.rootsolver import broyden1

    def fcn(x):
        return x**2 - 4

    x0 = torch.tensor(0.0, requires_grad=True)
    x = broyden1(fcn, x0)
    assert torch.allclose(x, torch.tensor(-2.0000))


@pytest.mark.torch
def test_broyden2():
    from deepchem.utils.differentiation_utils.optimize.rootsolver import broyden2

    def fcn(x):
        return x**2 - 4

    x0 = torch.tensor(0.0, requires_grad=True)
    x = broyden2(fcn, x0)
    assert torch.allclose(x, torch.tensor(-2.0000))


@pytest.mark.torch
def test_linear_mixing():
    from deepchem.utils.differentiation_utils.optimize.rootsolver import linearmixing

    def fcn(x):
        return x**2 - 4

    x0 = torch.tensor(0.0, requires_grad=True)
    x = linearmixing(fcn, x0)
    assert torch.allclose(x, torch.tensor(2.0000))


@pytest.mark.torch
def test_rootfinder():
    from deepchem.utils.differentiation_utils import rootfinder

    def func1(y, A):
        return torch.tanh(A @ y + 0.1) + y / 2.0

    A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    y0 = torch.zeros((2, 1))  # zeros as the initial guess
    yroot = rootfinder(func1, y0, params=(A,))
    assert torch.allclose(yroot, torch.tensor([[-0.0459], [-0.0663]]), 0.001)


@pytest.mark.torch
def test_equilibrium():
    from deepchem.utils.differentiation_utils import equilibrium

    def func1(y, A):
        return torch.tanh(A @ y + 0.1) + y / 2.0

    A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    y0 = torch.zeros((2, 1))  # zeros as the initial guess
    yequil = equilibrium(func1, y0, params=(A,))
    assert torch.allclose(yequil, torch.tensor([[0.2313], [-0.5957]]), 0.001)


@pytest.mark.torch
def test_minimize():
    from deepchem.utils.differentiation_utils import minimize

    def func1(y, A):  # example function
        return torch.sum((A @ y)**2 + y / 2.0)

    A = torch.tensor([[1.1, 0.4], [0.3, 0.8]]).requires_grad_()
    y0 = torch.zeros((2, 1))  # zeros as the initia
    ymin = minimize(func1, y0, params=(A,))
    assert torch.allclose(ymin, torch.tensor([[-0.0519], [-0.2684]]), 0.001)


@pytest.mark.torch
def test_get_rootfinder_default_method():
    from deepchem.utils.differentiation_utils.optimize.rootfinder import _get_rootfinder_default_method
    assert _get_rootfinder_default_method(None) == 'broyden1'


@pytest.mark.torch
def test_get_equilibrium_default_method():
    from deepchem.utils.differentiation_utils.optimize.rootfinder import _get_equilibrium_default_method
    assert _get_equilibrium_default_method(None) == 'broyden1'


@pytest.mark.torch
def test_get_minimizer_default_method():
    from deepchem.utils.differentiation_utils.optimize.rootfinder import _get_minimizer_default_method
    assert _get_minimizer_default_method(None) == 'broyden1'


@pytest.mark.torch
def test_tableau():
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import _Tableau
    euler = _Tableau(c=[0.0], b=[1.0], a=[[0.0]])
    assert euler.c == [0.0]


@pytest.mark.torch
def test_explicit_rk():
    from deepchem.utils.differentiation_utils import explicit_rk
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import rk4_tableau
    from scipy.integrate import solve_ivp

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    y0 = torch.tensor([[10], [1]])
    t_start = 0
    t_end = 10
    steps = 100
    t = torch.linspace(t_start, t_end, steps)
    params = torch.tensor([1.1, 0.4, 0.1, 0.4])
    sol = explicit_rk(rk4_tableau,
                      lotka_volterra,
                      y0,
                      t,
                      params,
                      batch_size=1,
                      device="cpu")

    def lotka_volterra(t, z, *params):
        y1, y2 = z
        a, b, c, d = params
        return [(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)]

    sol_scipy = solve_ivp(lotka_volterra, (t_start, t_end), [10, 1],
                          t_eval=np.linspace(t_start, t_end, steps),
                          args=([1.1, 0.4, 0.1, 0.4]))
    assert torch.allclose(sol[-1][0],
                          torch.tensor(sol_scipy.y[0][-1], dtype=torch.float32),
                          0.01, 0.001)


def test_explicit_rk_multi_ode():
    from deepchem.utils.differentiation_utils import explicit_rk
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import rk4_tableau

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    batch_size = 3
    y0 = torch.randn(2, batch_size)
    t_start = 0
    t_end = 10
    steps = 100
    t = torch.linspace(t_start, t_end, steps)
    params = torch.randn(4, batch_size)

    sol = explicit_rk(rk4_tableau,
                      lotka_volterra,
                      y0,
                      t,
                      params,
                      batch_size=batch_size,
                      device="cpu")

    assert sol.shape == (steps, 2, batch_size)


@pytest.mark.torch
def test_rk38_ivp():
    from deepchem.utils.differentiation_utils import rk38_ivp
    from scipy.integrate import solve_ivp

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    y0 = torch.tensor([[10.], [1.]])
    t_start = 0
    t_end = 10
    steps = 100
    t = torch.linspace(t_start, t_end, steps)
    params = torch.tensor([1.1, 0.4, 0.1, 0.4])
    sol = rk38_ivp(lotka_volterra, y0, t, params)

    def lotka_volterra(t, z, *params):
        y1, y2 = z
        a, b, c, d = params
        return [(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)]

    sol_scipy = solve_ivp(lotka_volterra, (t_start, t_end), [10, 1],
                          t_eval=np.linspace(t_start, t_end, steps),
                          args=([1.1, 0.4, 0.1, 0.4]))
    assert torch.allclose(sol[-1][0],
                          torch.tensor(sol_scipy.y[0][-1], dtype=torch.float),
                          0.01, 0.001)


@pytest.mark.torch
def test_rk4_ivp():
    from deepchem.utils.differentiation_utils import rk4_ivp
    from scipy.integrate import solve_ivp

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    y0 = torch.tensor([[10.], [1.]])
    t_start = 0
    t_end = 10
    steps = 100
    t = torch.linspace(t_start, t_end, steps)
    params = torch.tensor([1.1, 0.4, 0.1, 0.4])
    sol = rk4_ivp(lotka_volterra, y0, t, params)

    def lotka_volterra(t, z, *params):
        y1, y2 = z
        a, b, c, d = params
        return [(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)]

    sol_scipy = solve_ivp(lotka_volterra, (t_start, t_end), [10, 1],
                          t_eval=np.linspace(t_start, t_end, steps),
                          args=([1.1, 0.4, 0.1, 0.4]))
    assert torch.allclose(sol[-1][0],
                          torch.tensor(sol_scipy.y[0][-1], dtype=torch.float),
                          0.01, 0.001)


@pytest.mark.torch
def test_euler():
    from deepchem.utils.differentiation_utils import fwd_euler_ivp
    from scipy.integrate import solve_ivp

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    y0 = torch.tensor([[10], [1]])
    t_start = 0
    t_end = 10
    # Euler method performs poorly with large steps hence the increased resolution
    steps = 1000
    t = torch.linspace(t_start, t_end, steps)
    params = torch.tensor([1.1, 0.4, 0.1, 0.4])
    sol = fwd_euler_ivp(lotka_volterra, y0, t, params)

    def lotka_volterra(t, z, *params):
        y1, y2 = z
        a, b, c, d = params
        return [(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)]

    sol_scipy = solve_ivp(lotka_volterra, (t_start, t_end), [10, 1],
                          t_eval=np.linspace(t_start, t_end, steps),
                          args=([1.1, 0.4, 0.1, 0.4]))
    assert torch.allclose(sol[-1][0],
                          torch.tensor(sol_scipy.y[0][-1], dtype=torch.float),
                          0.1, 0.001)


@pytest.mark.torch
def test_midpoint():
    from deepchem.utils.differentiation_utils import mid_point_ivp
    from scipy.integrate import solve_ivp

    def lotka_volterra(t, y, params):
        y1, y2 = y
        a, b, c, d = params
        return torch.stack([(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)])

    y0 = torch.tensor([[10.], [1.]])
    t_start = 0
    t_end = 10
    steps = 100
    t = torch.linspace(t_start, t_end, steps)
    params = torch.tensor([1.1, 0.4, 0.1, 0.4])
    sol = mid_point_ivp(lotka_volterra, y0, t, params)

    def lotka_volterra(t, z, *params):
        y1, y2 = z
        a, b, c, d = params
        return [(a * y1 - b * y1 * y2), (c * y2 * y1 - d * y2)]

    sol_scipy = solve_ivp(lotka_volterra, (t_start, t_end), [10, 1],
                          t_eval=np.linspace(t_start, t_end, steps),
                          args=([1.1, 0.4, 0.1, 0.4]))
    assert torch.allclose(sol[-1][0],
                          torch.tensor(sol_scipy.y[0][-1], dtype=torch.float),
                          0.01, 0.001)


@pytest.mark.torch
def test_terminate_param():
    from deepchem.utils.differentiation_utils import gd
    import torch

    def fun(x):
        return torch.tan(x), (1 / torch.cos(x))**2

    x0 = torch.tensor(0.0, requires_grad=True)
    x0.grad = torch.tensor(1.0)
    x1 = gd(fun, x0, [], terminate=True)
    x2 = gd(fun, x0, [], terminate=False)
    assert not torch.allclose(x1, x2)
