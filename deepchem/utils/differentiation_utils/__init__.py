# flake8: noqa
try:
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import explicit_rk
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import rk38_ivp
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import fwd_euler_ivp
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import rk4_ivp
    from deepchem.utils.differentiation_utils.integrate.explicit_rk import mid_point_ivp

    from deepchem.utils.differentiation_utils.editable_module import EditableModule

    from deepchem.utils.differentiation_utils.bcast import normalize_bcast_dims
    from deepchem.utils.differentiation_utils.bcast import get_bcasted_dims
    from deepchem.utils.differentiation_utils.bcast import match_dim

    from deepchem.utils.differentiation_utils.misc import set_default_option
    from deepchem.utils.differentiation_utils.misc import get_and_pop_keys
    from deepchem.utils.differentiation_utils.misc import get_method
    from deepchem.utils.differentiation_utils.misc import dummy_context_manager
    from deepchem.utils.differentiation_utils.misc import assert_runtime

    from deepchem.utils.differentiation_utils.linop import LinearOperator
    from deepchem.utils.differentiation_utils.linop import AddLinearOperator
    from deepchem.utils.differentiation_utils.linop import MulLinearOperator
    from deepchem.utils.differentiation_utils.linop import AdjointLinearOperator
    from deepchem.utils.differentiation_utils.linop import MatmulLinearOperator
    from deepchem.utils.differentiation_utils.linop import MatrixLinearOperator

    from deepchem.utils.differentiation_utils.pure_function import PureFunction
    from deepchem.utils.differentiation_utils.pure_function import get_pure_function
    from deepchem.utils.differentiation_utils.pure_function import make_sibling

    from deepchem.utils.differentiation_utils.grad import jac

    from deepchem.utils.differentiation_utils.solve import wrap_gmres
    from deepchem.utils.differentiation_utils.solve import exactsolve
    from deepchem.utils.differentiation_utils.solve import solve_ABE
    from deepchem.utils.differentiation_utils.solve import broyden1_solve
    from deepchem.utils.differentiation_utils.solve import get_batchdims
    from deepchem.utils.differentiation_utils.solve import setup_precond
    from deepchem.utils.differentiation_utils.solve import dot
    from deepchem.utils.differentiation_utils.solve import get_largest_eival
    from deepchem.utils.differentiation_utils.solve import safedenom
    from deepchem.utils.differentiation_utils.solve import setup_linear_problem
    from deepchem.utils.differentiation_utils.solve import gmres
    from deepchem.utils.differentiation_utils.solve import solve
    from deepchem.utils.differentiation_utils.solve import cg
    from deepchem.utils.differentiation_utils.solve import bicgstab

    from deepchem.utils.differentiation_utils.symeig import lsymeig
    from deepchem.utils.differentiation_utils.symeig import usymeig
    from deepchem.utils.differentiation_utils.symeig import symeig
    from deepchem.utils.differentiation_utils.symeig import ortho
    from deepchem.utils.differentiation_utils.symeig import exacteig
    from deepchem.utils.differentiation_utils.symeig import svd

    from deepchem.utils.differentiation_utils.optimize.rootsolver import broyden1
    from deepchem.utils.differentiation_utils.optimize.rootsolver import broyden2
    from deepchem.utils.differentiation_utils.optimize.rootsolver import linearmixing

    from deepchem.utils.differentiation_utils.optimize.equilibrium import anderson_acc

    from deepchem.utils.differentiation_utils.optimize.minimizer import gd
    from deepchem.utils.differentiation_utils.optimize.minimizer import adam

    from deepchem.utils.differentiation_utils.optimize.rootfinder import rootfinder
    from deepchem.utils.differentiation_utils.optimize.rootfinder import equilibrium
    from deepchem.utils.differentiation_utils.optimize.rootfinder import minimize
except:
    pass
