import json
import base64
import inspect
from datetime import datetime
from decimal import Decimal
from enum import Enum
import os
import torch
import tensorflow as tf
import numpy as np
from sklearn import metrics


class DeepChemConfig:
    """
    A class for handling model metadata, inspired by Hugging Face.
    """

    def __init__(self, config: dict = None, config_path=None, default_config=None):
        if config is None and config_path is None:
            raise ValueError("Either `config` dictionary or `config_path` must be provided.")

        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found at {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)

        self.config = config
        self.default_config = default_config if default_config else config.copy()

        for key, value in config.items():
            setattr(self, key, value)

    @staticmethod
    def serialize(obj):

        """Converts any Python object into a JSON-compatible format."""
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        if isinstance(obj, (list, tuple, set)):
            return {"__type__": type(obj).__name__, "__value__": [DeepChemConfig.serialize(i) for i in obj]}

        if isinstance(obj, type({})):
            return {"__type__": "dict", "__value__": {k: DeepChemConfig.serialize(v) for k, v in obj.items()}}

        if isinstance(obj, complex):
            return {"__type__": "complex", "real": obj.real, "imag": obj.imag}

        if isinstance(obj, datetime):
            return {"__type__": "datetime", "value": obj.isoformat()}

        if isinstance(obj, Decimal):
            return {"__type__": "Decimal", "value": str(obj)}

        if isinstance(obj, Enum):
            return {"__type__": "Enum", "name": obj.name, "value": obj.value}

        # Handle functions and their arguments
        if callable(obj):
            try:
                module_name = obj.__module__
                function_name = obj.__name__
                signature = inspect.signature(obj)
                parameters = {
                    name: (param.default if param.default is not inspect.Parameter.empty else None)
                    for name, param in signature.parameters.items()
                }

                return {
                    "__type__": "function",
                    "module": module_name,
                    "name": function_name,
                    "parameters": parameters
                }
            except Exception:
                return {"__type__": "function", "name": str(obj)}

        if isinstance(obj, (bytes, bytearray, memoryview)):
            return {"__type__": "bytes", "value": base64.b64encode(obj).decode("utf-8")}

        # Handle PyTorch optimizers
        if isinstance(obj, torch.optim.Optimizer):
            return {
                "__type__": "torch_optimizer",
                "name": obj.__class__.__name__,
                "module": obj.__module__,
                "defaults": obj.defaults
            }

        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return {
                "__type__": "object",
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "attributes": {k: DeepChemConfig.serialize(v) for k, v in obj.__dict__.items()}
            }

        raise TypeError(f"Type {type(obj)} is not serializable")

    @staticmethod
    def deserialize(obj):
        """Converts a JSON-compatible format back into a Python object."""
        
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        if isinstance(obj, type({})):
            obj_type = obj.get("__type__")

            if obj_type in ["list", "tuple", "set"]:
                return eval(obj_type)(DeepChemConfig.deserialize(i) for i in obj["__value__"])

            if obj_type == "dict":
                return {k: DeepChemConfig.deserialize(v) for k, v in obj["__value__"].items()}

            if obj_type == "complex":
                return complex(obj["real"], obj["imag"])

            if obj_type == "datetime":
                return datetime.fromisoformat(obj["value"])

            if obj_type == "Decimal":
                return Decimal(obj["value"])

            if obj_type == "Enum":
                return Enum(obj["name"], obj["value"])

            # Reconstruct functions
            if obj_type == "function":
                module_name = obj["module"]
                function_name = obj["name"]
                parameters = obj.get("parameters", {})

                try:
                    module = __import__(module_name, fromlist=[function_name])
                    func = getattr(module, function_name)

                    def wrapped_function(*args, **kwargs):
                        kwargs = {**parameters, **kwargs}  # Apply default parameters
                        return func(*args, **kwargs)

                    return wrapped_function
                except Exception as e:
                    print(f"Failed to deserialize function {function_name}: {e}")
                    return None

            # Reconstruct PyTorch optimizers
            if obj_type == "torch_optimizer":
                optimizer_class = getattr(torch.optim, obj["name"])
                return optimizer_class  # Returning the class, not an instance

        return obj

    def save_config(self, save_path: str):
        """Save the configuration as a JSON file."""
        with open(save_path, "w") as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {save_path}")

    @staticmethod
    def load_config(load_path: str):
        """Load configuration from a JSON file."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Configuration file not found at {load_path}")

        with open(load_path, "r") as f:
            config = json.load(f)
        
        return DeepChemConfig(config=config)

    def show_config(self):
        """Pretty print the configuration."""
        print(json.dumps(self.config, indent=4))

    @staticmethod
    def update_config(old_config, new_config: dict):
        """
        Updates the existing configuration with a new configuration dictionary.
        """
        old_config.update(new_config)

