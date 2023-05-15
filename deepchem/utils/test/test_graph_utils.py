def test_fourier_encode_dist():
    import numpy as np

    from deepchem.utils.graph_utils import fourier_encode_dist
    x = np.array([1.0, 2.0, 3.0])
    num_encodings = 4
    include_self = True

    encoded_x = fourier_encode_dist(x,
                                    num_encodings=num_encodings,
                                    include_self=include_self)
    assert encoded_x.shape == (x.shape[0],
                               num_encodings * 2 + int(include_self))

    scales = 2**np.arange(num_encodings)
    x_scaled = x[..., np.newaxis] / scales
    x_sin = np.sin(x_scaled)
    x_cos = np.cos(x_scaled)
    x_expected = np.concatenate([x_sin, x_cos], axis=-1)
    if include_self:
        x_expected = np.concatenate((x_expected, x[..., np.newaxis]), axis=-1)

    assert np.allclose(encoded_x, x_expected, atol=1e-5)
