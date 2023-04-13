from deepchem.models.dft.dftxc import XCModel
from deepchem.data.data_loader import DFTYamlLoader
import pytest


@pytest.mark.dqc
def test_dftxcloss():
    inputs = 'deepchem/models/tests/assets/test_dftxcdata.yaml'
    k = DFTYamlLoader()
    dataset = (k.create_dataset(inputs))
    model = XCModel("lda_x", batch_size=1)
    loss = model.fit(dataset, nb_epoch=1, checkpoint_interval=1)
    assert loss == 0.003224595397314571
