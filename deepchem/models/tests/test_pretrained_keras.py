import unittest
import pytest
import deepchem as dc
import numpy as np
from deepchem.feat.mol_graphs import ConvMol

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense

    class MLP(dc.models.KerasModel):

        def __init__(self,
                     n_tasks=1,
                     feature_dim=100,
                     hidden_layer_size=64,
                     **kwargs):
            self.feature_dim = feature_dim
            self.hidden_layer_size = hidden_layer_size
            self.n_tasks = n_tasks

            model, loss, output_types = self._build_graph()
            super(MLP, self).__init__(model=model,
                                      loss=loss,
                                      output_types=output_types,
                                      **kwargs)

        def _build_graph(self):
            inputs = Input(dtype=tf.float32,
                           shape=(self.feature_dim,),
                           name="Input")
            out1 = Dense(units=self.hidden_layer_size,
                         activation='relu')(inputs)

            final = Dense(units=self.n_tasks, activation='sigmoid')(out1)
            outputs = [final]
            output_types = ['prediction']
            loss = dc.models.losses.BinaryCrossEntropy()

            model = tf.keras.Model(inputs=[inputs], outputs=outputs)
            return model, loss, output_types

    has_tensorflow = True
except:
    has_tensorflow = False


class TestPretrained(unittest.TestCase):

    @pytest.mark.tensorflow
    def setUp(self):
        self.feature_dim = 2
        self.hidden_layer_size = 10
        data_points = 10

        X = np.random.randn(data_points, self.feature_dim)
        y = (X[:, 0] > X[:, 1]).astype(np.float32)

        self.dataset = dc.data.NumpyDataset(X, y)

    @pytest.mark.tensorflow
    def test_load_from_pretrained(self):
        """Tests loading pretrained model."""
        source_model = MLP(hidden_layer_size=self.hidden_layer_size,
                           feature_dim=self.feature_dim,
                           batch_size=10)

        source_model.fit(self.dataset, nb_epoch=1000, checkpoint_interval=0)

        dest_model = MLP(feature_dim=self.feature_dim,
                         hidden_layer_size=self.hidden_layer_size,
                         n_tasks=10)

        assignment_map = dict()
        value_map = dict()
        dest_vars = dest_model.model.trainable_variables[:-2]

        for idx, dest_var in enumerate(dest_vars):
            source_var = source_model.model.trainable_variables[idx]
            assignment_map[source_var.experimental_ref()] = dest_var
            value_map[source_var.experimental_ref()] = source_var.numpy()

        dest_model.load_from_pretrained(source_model=source_model,
                                        assignment_map=assignment_map,
                                        value_map=value_map)

        for source_var, dest_var in assignment_map.items():
            source_val = source_var.deref().numpy()
            dest_val = dest_var.numpy()
            np.testing.assert_array_almost_equal(source_val, dest_val)

    @pytest.mark.tensorflow
    def test_load_pretrained_subclassed_model(self):
        from rdkit import Chem
        bi_tasks = ['a', 'b']
        y = np.ones((3, 2))
        smiles = ['C', 'CC', 'CCC']
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        featurizer = dc.feat.ConvMolFeaturizer()
        X = featurizer.featurize(mols)
        dataset = dc.data.NumpyDataset(X, y, ids=smiles)

        source_model = dc.models.GraphConvModel(n_tasks=len(bi_tasks),
                                                graph_conv_layers=[128, 128],
                                                dense_layer_size=512,
                                                dropout=0,
                                                mode='regression',
                                                learning_rate=0.001,
                                                batch_size=8,
                                                model_dir="model")
        source_model.fit(dataset)

        dest_model = dc.models.GraphConvModel(n_tasks=len(bi_tasks),
                                              graph_conv_layers=[128, 128],
                                              dense_layer_size=512,
                                              dropout=0,
                                              mode='regression',
                                              learning_rate=0.001,
                                              batch_size=8)

        X_b, y_b, w_b, ids_b = next(
            dataset.iterbatches(batch_size=8,
                                deterministic=True,
                                pad_batches=True))
        multiConvMol = ConvMol.agglomerate_mols(X_b)
        n_samples = np.array(X_b.shape[0])
        inputs = [
            multiConvMol.get_atom_features(), multiConvMol.deg_slice,
            np.array(multiConvMol.membership), n_samples
        ]
        for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
            inputs.append(multiConvMol.get_deg_adjacency_lists()[i])

        dest_model.load_from_pretrained(source_model=source_model,
                                        assignment_map=None,
                                        value_map=None,
                                        include_top=False,
                                        inputs=inputs)

        source_vars = source_model.model.trainable_variables[:-2]
        dest_vars = dest_model.model.trainable_variables[:-2]
        assert len(source_vars) == len(dest_vars)

        for source_var, dest_var in zip(*(source_vars, dest_vars)):
            source_val = source_var.numpy()
            dest_val = dest_var.numpy()
            np.testing.assert_array_almost_equal(source_val, dest_val)

    @pytest.mark.tensorflow
    def test_restore_equivalency(self):
        """Test for restore based pretrained model loading."""
        source_model = MLP(feature_dim=self.feature_dim,
                           hidden_layer_size=self.hidden_layer_size)

        source_model.fit(self.dataset, nb_epoch=1000)

        dest_model = MLP(feature_dim=self.feature_dim,
                         hidden_layer_size=self.hidden_layer_size)

        dest_model.load_from_pretrained(source_model=source_model,
                                        assignment_map=None,
                                        value_map=None,
                                        model_dir=None,
                                        include_top=True)

        predictions = np.squeeze(dest_model.predict_on_batch(self.dataset.X))
        np.testing.assert_array_almost_equal(self.dataset.y,
                                             np.round(predictions))
