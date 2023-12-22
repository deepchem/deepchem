try:
    from deepchem.utils.differentiation_utils.editable_module import EditableModule

    from deepchem.utils.differentiation_utils.bcast import normalize_bcast_dims
    from deepchem.utils.differentiation_utils.bcast import get_bcasted_dims
    from deepchem.utils.differentiation_utils.bcast import match_dim

    from deepchem.utils.differentiation_utils.linop import LinearOperator
    from deepchem.utils.differentiation_utils.linop import AddLinearOperator
    from deepchem.utils.differentiation_utils.linop import MulLinearOperator
    from deepchem.utils.differentiation_utils.linop import AdjointLinearOperato
    from deepchem.utils.differentiation_utils.linop import MatmulLinearOperator
    from deepchem.utils.differentiation_utils.linop import MatrixLinearOperator

    from deepchem.utils.differentiation_utils.misc import set_default_option
    from deepchem.utils.differentiation_utils.misc import get_and_pop_keys
    from deepchem.utils.differentiation_utils.misc import get_method
    from deepchem.utils.differentiation_utils.misc import dummy_context_manager
    from deepchem.utils.differentiation_utils.misc import assert_runtime
    from deepchem.utils.differentiation_utils.misc import assert_type    
    from deepchem.utils.differentiation_utils.misc import tallqr
    from deepchem.utils.differentiation_utils.misc import to_fortran_order
    from deepchem.utils.differentiation_utils.misc import UnimplementedError
    from deepchem.utils.differentiation_utils.misc import GetSetParamsError
    from deepchem.utils.differentiation_utils.misc import ConvergenceWarning
    from deepchem.utils.differentiation_utils.misc import MathWarning
    from deepchem.utils.differentiation_utils.misc import get_np_dtype
    from deepchem.utils.differentiation_utils.misc import Uniquifier
    from deepchem.utils.differentiation_utils.misc import TensorNonTensorSeparator

    from deepchem.utils.differentiation_utils.pure_function import get_pure_function
    from deepchem.utils.differentiation_utils.pure_function import make_sibling
    from deepchem.utils.differentiation_utils.pure_function import PureFunction
except:
    pass
