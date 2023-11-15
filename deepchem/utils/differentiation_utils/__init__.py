try:
    from deepchem.utils.differentiation_utils.editable_module import EditableModule

    from deepchem.utils.differentiation_utils.bcast import normalize_bcast_dims
    from deepchem.utils.differentiation_utils.bcast import get_bcasted_dims
    from deepchem.utils.differentiation_utils.bcast import match_dim

    from deepchem.utils.differentiation_utils.misc import set_default_option
    from deepchem.utils.differentiation_utils.misc import get_and_pop_keys
    from deepchem.utils.differentiation_utils.misc import get_method
    from deepchem.utils.differentiation_utils.misc import dummy_context_manager
    from deepchem.utils.differentiation_utils.misc import assert_runtime
except:
    pass
