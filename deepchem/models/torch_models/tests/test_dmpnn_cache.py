def test_numerical_parity_minimal():
    """
    Test that cached values are same as recomputed values
    """
    from deepchem.feat import DMPNNFeaturizer
    from deepchem.models.torch_models.dmpnn import _MapperDMPNN
    import numpy as np
    smiles = "CCO" # Ethanol
    featurizer = DMPNNFeaturizer(cache_mapper=True)
    graph = featurizer.featurize([smiles])[0]
    cached_values = graph._cached_dmpnn.to_tuple()
    mapper = _MapperDMPNN(graph)
    recomputed_values = mapper.values
    for cached, recomputed in zip(cached_values, recomputed_values):
        if cached is None:
            assert recomputed is None
            continue
        cached_nparray = cached.detach().cpu().numpy() if hasattr(cached,'detach') else cached
        recomputed_nparray = recomputed.detach().cpu().numpy() if hasattr(recomputed, 'detach') else recomputed
        np.testing.assert_array_almost_equal(cached_nparray, recomputed_nparray,decimal=6)

def test_serilization():
    """
    Test that the cahced data passes pickling (DiskDataset Compatibility)
    """
    import pickle
    from deepchem.feat import DMPNNFeaturizer
    featurizer = DMPNNFeaturizer(cache_mapper=True)
    graph = featurizer.featurize(['CCO'])[0]
    assert graph._cached_dmpnn is not None
    pickle_data = pickle.dumps(graph)
    restored_graph = pickle.loads(pickle_data)
    assert hasattr(restored_graph, '_cached_dmpnn')
    assert restored_graph._cached_dmpnn is not None
    assert restored_graph._cached_dmpnn.mapping.shape == graph._cached_dmpnn.mapping.shape