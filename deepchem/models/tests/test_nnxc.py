import pytest
try:
    import torch
    from deepchem.models.dft.nnxc import NNLDA, HybridXC
    import torch.nn as nn
    from dqc.utils.datastruct import ValGrad
    has_dqc = True
except ModuleNotFoundError:
    has_dqc = False


@pytest.mark.dqc
def dummymodel():
    n = 2

    class DummyModel(torch.nn.Module):

        def __init__(self, n):
            super(DummyModel, self).__init__()
            self.linear = nn.Linear(n, 1)

        def forward(self, x):
            return self.linear(x)

    return DummyModel(n)


@pytest.mark.dqc
def test_nnlda():
    torch.manual_seed(42)
    # https://github.com/diffqc/dqc/blob/742eb2576418464609f942def4fb7c3bbdc0cd82/dqc/test/test_xc.py#L15
    n = 2
    model = dummymodel()
    k = NNLDA(model)
    densinfo = ValGrad(
        value=torch.rand((n,), dtype=torch.float32).requires_grad_())
    output = k.get_edensityxc(densinfo).detach()
    expected_output = torch.tensor([0.3386, 0.0177])
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)


@pytest.mark.dqc
def test_hybridxc():
    torch.manual_seed(42)
    n = 2
    nnmodel = dummymodel()
    k = HybridXC("lda_x", nnmodel, aweight0=0.0)
    densinfo = ValGrad(
        value=torch.rand((n,), dtype=torch.float32).requires_grad_())
    output = k.get_edensityxc(densinfo).detach()
    expected_output = torch.tensor([-0.6988, -0.2108], dtype=torch.float64)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=0)
