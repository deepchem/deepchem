import warnings
from deepchem.models.torch_models.layers import DTNNEmbedding

def test_dtnn_embedding_initializer_backward_compat():
    # Correct spelling should work
    layer = DTNNEmbedding(n_embedding=10, initializer='xavier')
    assert layer is not None
    assert layer.initializer == 'xavier'

    # Old misspelled spelling should still work with DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        layer_old = DTNNEmbedding(n_embedding=10, initalizer='xavier')
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
