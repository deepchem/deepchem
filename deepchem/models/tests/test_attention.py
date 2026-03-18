import pytest
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    pass


@pytest.mark.torch
def testScaledDotProductAttention():
    from deepchem.models import ScaledDotProductAttention as SDPA
    attn = SDPA()
    x = torch.ones(2, 5)
    # Linear layers for making query, key, value
    Q, K, V = nn.Parameter(torch.ones(5)), nn.Parameter(
        torch.ones(5)), nn.Parameter(torch.ones(5))
    query, key, value = Q * x, K * x, V * x
    mask = torch.Tensor([1, 0])
    x_out, attn_score = attn(query, key, value, mask=mask)
    torch.testing.assert_close(x_out, torch.ones(2, 5))
    torch.testing.assert_close(attn_score, torch.Tensor([[1, 0], [1, 0]]))


@pytest.mark.torch
def testSelfAttention():
    from deepchem.models import SelfAttention as SA
    n, in_feature, out_feature = 10, 4, 8
    attn = SA(in_feature, out_feature, hidden_size=16)
    x = torch.randn((n, in_feature))
    x, attn = attn(x)
    assert x.size() == (out_feature, in_feature)
