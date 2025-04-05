import json
import os
# from deepchem.configuration import DeepChemConfig
from deepchem import models
from deepchem.models.torch_models.graphconvmodel import GraphConvModel
from deepchem.models.torch_models.cnn import CNN
from deepchem.models.torch_models.text_cnn import TextCNNModel
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.torch_models.dtnn import DTNNModel
from deepchem.models.optimizers import (AdaGrad, Adam, SparseAdam, AdamW,
                                        RMSProp, ExponentialDecay,
                                        LambdaLRWithWarmup, PolynomialDecay,
                                        LinearCosineDecay,
                                        PiecewiseConstantSchedule, KFAC, Lamb)
from deepchem.models.losses import L1Loss, L2Loss, HuberLoss, HingeLoss, SquaredHingeLoss

# Model mapping dictionary
mapping_models_torch = {
    "graph-conv": GraphConvModel,
    "torch-model": TorchModel,
    "CNN": CNN,
    "text-CNN": TextCNNModel,
    "DTNN": DTNNModel
}

# Optimizer mapping dictionary
optimizers_map = {
    "AdaGrad": AdaGrad,
    "Adam": Adam,
    "SparseAdam": SparseAdam,
    "AdamW": AdamW,
    "RMSProp": RMSProp,
    "ExponentialDecay": ExponentialDecay,
    "LambdaLRWithWarmup": LambdaLRWithWarmup,
    "PolynomialDecay": PolynomialDecay,
    "LinearCosineDecay": LinearCosineDecay,
    "PiecewiseConstantSchedule": PiecewiseConstantSchedule,
    "KFAC": KFAC,
    "Lamb": Lamb
}

# Loss function mapping dictionary
losses_map = {
    "L1Loss": L1Loss,
    "L2Loss": L2Loss,
    "HuberLoss": HuberLoss,
    "HingeLoss": HingeLoss,
    "SquaredHingeLoss": SquaredHingeLoss
}


class Map:
    """
    Example:
        >>> import deepchem as dc
        >>> tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
        >>> train_dataset, valid_dataset, test_dataset = datasets
        >>> n_tasks = len(tasks)
        >>> num_features = train_dataset.X[0].get_atom_features().shape[1]
        >>> model = dc.models.torch_models.GraphConvModel(n_tasks, mode='classification', number_input_features=[num_features, 64])
        >>> model.fit(train_dataset, nb_epoch=50)
        >>> model.save_pretrained("save_graph")  # Saving the model
        >>> model_reload = deepchem.deepchemmap.Map.load_from_pretrained("save_graph")  # The model gets loaded

    Mappings:
        - `mapping_models_torch`: Maps model name strings to DeepChem model classes.
        - `optimizers_map`: Maps optimizer name strings to DeepChem optimizer classes.
        - `losses_map`: Maps loss function names to DeepChem loss classes.

    Methods:
        - `load_param_dict(json_file)`: Loads a parameter dictionary from a JSON file.
        - `load_from_config(directory)`: Instantiates a model from a configuration directory.
        - `load_from_pretrained(directory, strict=True)`: Loads and restores a pretrained model.
    """

    @staticmethod
    def load_param_dict(json_file):
        """
        Loads a parameter dictionary from a JSON file.

        Args:
            json_file (str): Path to the JSON file containing model parameters.

        Returns:
            dict: A deserialized dictionary containing the model parameters.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            ValueError: If the model name is not found in the mapping dictionary.
            AttributeError: If the model class lacks a `deserialize_dict` method.
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Parameter file not found: {json_file}")

        with open(json_file, "r") as f:
            data = json.load(f)

        model_name = data["name"]["__value__"]
        if model_name not in mapping_models_torch:
            raise ValueError(f"Model '{model_name}' not found in model map")

        model_class = mapping_models_torch[model_name]

        if not hasattr(model_class, "deserialize_dict"):
            raise AttributeError(
                f"Model class '{model_name}' is missing 'deserialize_dict' method."
            )

        return model_class.deserialize_dict(data)

    @staticmethod
    def load_from_config(directory):
        """
        Loads a model from the configuration file in a directory.

        Args:
            directory (str): The directory containing the `parameters.json` file.

        Returns:
            torch.nn.Module: The instantiated model based on the configuration.

        Raises:
            FileNotFoundError: If `parameters.json` is missing.
            ValueError: If the model name is unknown.
        """
        param_path = os.path.join(directory, "parameters.json")
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"Parameter file not found: {param_path}")

        deserialized_params = Map.load_param_dict(param_path)
        model_name = deserialized_params['name']

        if model_name not in mapping_models_torch:
            raise ValueError(
                f"Unknown model name '{model_name}' in parameters.json")

        model_class = mapping_models_torch[model_name]
        return model_class(**deserialized_params)

    @staticmethod
    def load_from_pretrained(directory, strict=True):
        """
        Loads a pretrained model from a specified directory.

        Args:
            directory (str): Path to the directory containing model weights and configurations.
            strict (bool, optional): Whether to enforce strict loading of weights. Defaults to True.

        Returns:
            torch.nn.Module: A restored model instance.

        Raises:
            FileNotFoundError: If the directory or model weights file is missing.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory not found: {directory}")

        model = Map.load_from_config(directory)

        weights_path = os.path.join(directory, "model_weights.pt")
        if os.path.exists(weights_path):
            model.restore(weights_path)  # Keep DeepChem's restore method
        else:
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        return model
