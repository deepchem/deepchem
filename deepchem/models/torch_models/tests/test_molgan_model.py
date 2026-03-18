import os
import unittest
import pytest
import pandas as pd
from deepchem.data import NumpyDataset
from deepchem.feat.molecule_featurizers import MolGanFeaturizer
from deepchem.models.optimizers import ExponentialDecay
try:
    import torch
    import torch.nn.functional as F
    from deepchem.models.torch_models import BasicMolGANModel as MolGAN
    has_torch = True
except:
    has_torch = False


class test_molgan_model(unittest.TestCase):
    """
  Unit testing for MolGAN basic layers
  """

    @pytest.mark.torch
    def setUp(self):
        self.training_attempts = 6
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.vertices = 9
        self.nodes = 5
        self.edges = 5
        self.embedding_dim = 10
        self.dropout_rate = 0.0
        self.batch_size = 100
        self.first_convolution_unit = 128
        self.second_convolution_unit = 64
        self.aggregation_unit = 128
        self.model = MolGAN(edges=self.edges,
                            vertices=self.vertices,
                            nodes=self.nodes,
                            embedding_dim=self.embedding_dim,
                            dropout_rate=self.dropout_rate,
                            device=torch.device('cpu'))

    @pytest.mark.torch
    def test_build(self):
        """
    Test if initialization data is set-up correctly
    """
        model = self.model
        assert model.batch_size == self.batch_size
        assert model.edges == self.edges
        assert model.nodes == self.nodes
        assert model.vertices == self.vertices
        assert model.dropout_rate == self.dropout_rate
        assert len(model.generators) == 1
        assert len(model.discriminators) == 1

    @pytest.mark.torch
    def test_training(self):
        """
    Check training of the basicMolGANmodel on small number of compounds.
    Due to training instability try a few times and see if it worked at least once.
    Typically it fails between 1-3 times of 10.
    This is something that needs to be addressed in future releases.
    """
        input_file = os.path.join("deepchem/models/tests/assets/",
                                  "molgan_example.csv")
        data = pd.read_csv(input_file)
        molecules = list(data['Molecule'])
        feat = MolGanFeaturizer()
        featurized = feat.featurize(molecules)
        dataset = NumpyDataset([x.adjacency_matrix for x in featurized],
                               [x.node_features for x in featurized])

        # True will be assigned up successful training attempt
        success = False

        for _ in range(self.training_attempts):
            # create new model
            gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000),
                         device=torch.device('cpu'))

            # to avoid flake8 E125/yapf incompatibility
            s = gan.batch_size

            # generate input
            def iterbatches(epochs):
                for __ in range(epochs):
                    for batch in dataset.iterbatches(batch_size=s,
                                                     pad_batches=True):
                        adjacency_tensor = F.one_hot(
                            torch.Tensor(batch[0]).to(torch.int64),
                            gan.edges).to(torch.float32)
                        node_tesor = F.one_hot(
                            torch.Tensor(batch[1]).to(torch.int64),
                            gan.nodes).to(torch.float32)
                        yield {
                            gan.data_inputs[0]: adjacency_tensor,
                            gan.data_inputs[1]: node_tesor
                        }

            # train model
            gan.fit_gan(iterbatches(1000),
                        generator_steps=0.2,
                        checkpoint_interval=0)

            # generate sample
            g = gan.predict_gan_generator(1000)

            # check how many valid molecules were created and add to list
            generated_molecules = feat.defeaturize(g)
            valid_molecules_count = len(
                list(filter(lambda x: x is not None, generated_molecules)))
            print(valid_molecules_count)
            if valid_molecules_count:
                success = True
                break

        # finally test if there was at least one valid training session
        # as the model structure improves this should become more and more strict
        assert success


if __name__ == '__main__':
    unittest.main()
