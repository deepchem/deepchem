import json
import os
from deepchem.configuration import DeepChemConfig
from models.torch_models.graphconvmodel import GraphConvModel
from models.torch_models.cnn import CNN
from models.torch_models.text_cnn import TextCNNModel
from models.torch_models.torch_model import TorchModel
from models.torch_models.dtnn import DTNNModel

# Model mapping dictionary
mapping_models_torch = {
    "graph-conv": GraphConvModel,
    "torch-model": TorchModel,
    "CNN": CNN,
    "text-CNN": TextCNNModel,
    "DTNN": DTNNModel
}

class Map:
    mapping_models_torch = mapping_models_torch  # Store the model mapping

    @staticmethod
    def load_config(model_dir):
        """Loads the configuration from config.json."""
        config_path = os.path.join(model_dir, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            serialized_config = json.load(f)
        config = DeepChemConfig.deserialize(serialized_config)
        return config

    @staticmethod
    def save_config(model_dir, config):
        """Saves the configuration to config.json."""
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved at {config_path}")

    @staticmethod
    def load_model(model_dir):
        """Loads the model and restores its weights."""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        config = Map.load_config(model_dir)
        model_name = config.get("name")

        model_class = Map.mapping_models_torch.get(model_name)
        if model_class is None:
            raise ValueError(f"Unknown model name '{model_name}' in config")

        # Instantiate the model with config parameters
        deserialized_config = DeepChemConfig.deserialize(config)
        model = model_class(**deserialized_config)

        # Load weights
        weights_path = os.path.join(model_dir, 'model_weights.pt')
        if os.path.exists(weights_path):
            model.restore(weights_path)  # Assuming model has restore() method
        else:
            raise FileNotFoundError(f"Model weights not found: {weights_path}")

        return model

    @staticmethod
    def save_model_weights(model, model_dir):
        """Saves the model weights to model_weights.pt."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        weights_path = os.path.join(model_dir, 'model_weights.pt')
        model.save(weights_path)  # Assuming model has save() method
        print(f"Model weights saved at {weights_path}")

    @staticmethod
    def list_available_models():
        """Returns a list of supported model names."""
        return list(Map.mapping_models_torch.keys())

    @staticmethod
    def check_model_directory(model_dir):
        """Checks if the required files exist in the model directory."""
        required_files = ['config.json', 'model_weights.pt']
        missing_files = [file for file in required_files if not os.path.exists(os.path.join(model_dir, file))]

        if missing_files:
            raise FileNotFoundError(f"Missing required files in model directory: {', '.join(missing_files)}")
        print("Model directory structure is valid.")

