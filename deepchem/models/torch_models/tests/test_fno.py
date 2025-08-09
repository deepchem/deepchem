import pytest
import deepchem as dc
import tempfile
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    pass


@pytest.mark.torch
def test_fno_block_construction():
    """
    Test if FNOBlock can be constructed in 1D, 2D, and 3D.
    """
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=1)
    assert block is not None, "1D FNO Block construction failed"

    block = FNOBlock(width=32, modes=8, dims=2)
    assert block is not None, "2D FNO Block construction failed"

    block = FNOBlock(width=32, modes=8, dims=3)
    assert block is not None, "3D FNO Block construction failed"


@pytest.mark.torch
def test_fno_block_forward():
    """
    Test if FNOBlock can be forward-passed in 2D, without any errors.
    """
    from deepchem.models.torch_models.fno import FNOBlock
    block = FNOBlock(width=32, modes=8, dims=2)
    x = torch.randn(100, 32, 100, 100)
    y = block(x)
    assert y.shape == (100, 32, 100, 100)


@pytest.mark.torch
def test_fno_base_construction():
    """
    Test if FNO can be constructed in 1D, 2D, and 3D.
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    assert model is not None, "FNO base model construction failed"


@pytest.mark.torch
def test_fno_base_forward():
    """
    Test if FNO can be forward-passed in 2D, without any errors.
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    x = torch.randn(100, 1, 100, 100)
    y = model(x)
    assert y.shape == (100, 1, 100, 100)


@pytest.mark.torch
def test_fno_base_meshgrids():
    """
    Test if meshgrids are properly generated
    """
    from deepchem.models.torch_models.fno import FNO
    model = FNO(in_channels=1, out_channels=1, modes=8, width=32, dims=2)
    x = torch.randn(100, 1, 100, 100)
    meshgrid = model._generate_meshgrid(x)
    assert meshgrid.shape == (100, 2, 100, 100)


@pytest.mark.torch
def test_fno_construction():
    """Test that FNO Model can be constructed without crash."""
    from deepchem.models.torch_models import FNOModel

    model_1d = FNOModel(in_channels=1,
                        out_channels=1,
                        modes=8,
                        width=32,
                        dims=1,
                        depth=2)
    assert model_1d is not None

    model_2d = FNOModel(in_channels=2,
                        out_channels=3,
                        modes=8,
                        width=64,
                        dims=2,
                        depth=3)
    assert model_2d is not None

    model_3d = FNOModel(in_channels=3,
                        out_channels=1,
                        modes=4,
                        width=32,
                        dims=3,
                        depth=2)
    assert model_3d is not None


@pytest.mark.torch
def test_fno_overfit():
    """Test that FNO model can overfit simple data."""
    from deepchem.models.torch_models import FNOModel

    model = FNOModel(in_channels=1,
                     out_channels=1,
                     modes=8,
                     width=128,
                     dims=1,
                     depth=4)
    X = torch.randn(10, 1, 32)
    y = X  # Identity mapping
    dataset = dc.data.NumpyDataset(X=X, y=y)
    loss = model.fit(dataset, nb_epoch=200)
    assert loss < 1e-1, "Model can't overfit"


@pytest.mark.torch
def test_fno_overfit_2d():
    """Test that 2D FNO model can overfit simple data."""
    from deepchem.models.torch_models import FNOModel

    model = FNOModel(in_channels=1,
                     out_channels=1,
                     modes=8,
                     width=64,
                     dims=2,
                     depth=3)
    X = torch.rand(10, 1, 32, 32)
    y = X
    dataset = dc.data.NumpyDataset(X=X, y=y)
    loss = model.fit(dataset, nb_epoch=200)
    assert loss < 1e-1, "2D Model can't overfit"


@pytest.mark.torch
def test_fno_prediction_shape():
    """Test that FNO predictions have correct shape."""
    from deepchem.models.torch_models import FNOModel

    model = FNOModel(in_channels=2,
                     out_channels=3,
                     modes=8,
                     width=32,
                     dims=1,
                     depth=2)
    X = torch.randn(10, 2, 50)
    y = torch.randn(10, 3, 50)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    # Test prediction shape
    predictions = model.predict(dataset)
    assert predictions.shape == y.shape


@pytest.mark.torch
def test_fno_with_different_modes():
    """Test FNO with different numbers of modes."""
    from deepchem.models.torch_models import FNOModel

    X = torch.randn(10, 1, 32)
    y = torch.randn(10, 1, 32)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    # Test with different mode counts
    for modes in [4, 8, 16]:
        model = FNOModel(in_channels=1,
                         out_channels=1,
                         modes=modes,
                         width=32,
                         dims=1,
                         depth=2)
        predictions = model.predict(dataset)
        assert predictions.shape == y.shape


@pytest.mark.torch
def test_fno_with_different_widths():
    """Test FNO with different widths."""
    from deepchem.models.torch_models import FNOModel

    X = torch.randn(10, 1, 32)
    y = torch.randn(10, 1, 32)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    # Test with different widths
    for width in [16, 32, 64]:
        model = FNOModel(in_channels=1,
                         out_channels=1,
                         modes=8,
                         width=width,
                         dims=1,
                         depth=2)
        predictions = model.predict(dataset)
        assert predictions.shape == y.shape


@pytest.mark.torch
def test_fno_with_different_depths():
    """Test FNO with different depths."""
    from deepchem.models.torch_models import FNOModel

    X = torch.randn(10, 1, 32)
    y = torch.randn(10, 1, 32)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    # Test with different depths
    for depth in [1, 2, 4, 6]:
        model = FNOModel(in_channels=1,
                         out_channels=1,
                         modes=8,
                         width=32,
                         dims=1,
                         depth=depth)
        model.fit(dataset, nb_epoch=5)
        predictions = model.predict(dataset)
        assert predictions.shape == y.shape


@pytest.mark.torch
def test_fno_loss_function():
    """Test that FNO loss function works correctly."""
    from deepchem.models.torch_models import FNOModel

    model = FNOModel(in_channels=1,
                     out_channels=1,
                     modes=8,
                     width=32,
                     dims=1,
                     depth=2)

    # Test loss function directly
    # Outputs and labels should have shape (batch, channels, spatial_dims)
    outputs = [torch.randn(10, 1, 32)]
    labels = [torch.randn(10, 1, 32)]

    loss_value = model._loss_fn(outputs, labels)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.item() >= 0.0  # MSE loss should be non-negative


@pytest.mark.torch
def test_fno_reload():
    """Test that FNO model can be reloaded."""
    from deepchem.models.torch_models import FNOModel
    import numpy as np

    np.random.seed(123)
    torch.manual_seed(123)

    n_samples = 10
    n_features = 32
    X = torch.randn(n_samples, 1, n_features)
    y = torch.randn(n_samples, 1, n_features)
    dataset = dc.data.NumpyDataset(X=X, y=y)

    model_dir = tempfile.mkdtemp()
    orig_model = FNOModel(in_channels=1,
                          out_channels=1,
                          modes=8,
                          width=32,
                          dims=1,
                          depth=2,
                          model_dir=model_dir)
    orig_model.fit(dataset, nb_epoch=5)

    reloaded_model = FNOModel(in_channels=1,
                              out_channels=1,
                              modes=8,
                              width=32,
                              dims=1,
                              depth=2,
                              model_dir=model_dir)
    reloaded_model.restore()

    X_new = torch.randn(n_samples, 1, n_features)
    orig_preds = orig_model.predict_on_batch(X_new)
    reloaded_preds = reloaded_model.predict_on_batch(X_new)

    assert np.all(orig_preds == reloaded_preds), "Predictions are not the same"
