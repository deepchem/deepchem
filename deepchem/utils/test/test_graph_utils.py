def test_fourier_encode_dist():
    import numpy as np
    import torch

    from deepchem.utils.graph_utils import fourier_encode_dist
    x = torch.tensor([1.0, 2.0, 3.0])
    num_encodings = 4
    include_self = True

    encoded_x = fourier_encode_dist(x,
                                    num_encodings=num_encodings,
                                    include_self=include_self)
    assert encoded_x.shape == (x.shape[0],
                               num_encodings * 2 + int(include_self))

    scales = 2**np.arange(num_encodings)
    x_scaled = x.unsqueeze(-1) / scales
    x_sin = torch.sin(x_scaled)
    x_cos = torch.cos(x_scaled)
    x_expected = torch.cat([x_sin, x_cos], dim=-1)
    if include_self:
        x_expected = torch.cat((x_expected, x.unsqueeze(-1)), dim=-1)

    assert torch.allclose(encoded_x.float(), x_expected.float(), atol=1e-5)
