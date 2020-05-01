"""
Keeps track of available pretrained configs for different deepchem models.
"""

__author__ = "Vignesh Ram Somnath"
__license_ = "MIT"

PRETRAINED_CONFIGS = {
    'ChemCeption': [{
        'filters': 16,
        'layers_per_block': 3,
        'imgspec': 'engd'
    }, {
        'filters': 32,
        'layers_per_block': 1,
        'imgspec': 'engd'
    }, {
        'filters': 32,
        'layers_per_block': 1,
        'imgspec': 'std'
    }, {
        'filters': 64,
        'layers_per_block': 3,
        'imgspec': 'std'
    }]
}

DEFAULT_CONFIGS = {
    'ChemCeption': {
        'filters': 16,
        'layers_per_block': 3,
        'imgspec': 'engd'
    }
}


def get_model_str(model_name, hparams):
  """
    Returns the name of the saved directory for the corresponding model and
    hyperparameters. If no pretrained model is available for the corresponding
    hyperparameters, the default configuration for the corresponding model is used.

    Parameters
    ----------
    model_name: str,
        Name of the model
    hparams: dict,
        Hyperparameter configuration of the model.
    """
  if model_name == 'ChemCeption':
    filters = hparams.get('filters')
    layers_per_block = hparams.get('layers_per_block')
    img_spec = hparams.get('img_spec')

    config = {
        "filters": filters,
        "layers_per_block": layers_per_block,
        "imgspec": img_spec
    }

  else:
    raise ValueError(f"No custom configs available for {model_name}")

  if config in PRETRAINED_CONFIGS[model_name]:
    return "_".join(f"{key}_{value}" for key, value in config.items())
  else:
    msg = "No pretrained model found for given hyperparameters. Using default hyperparameters"
    print(msg)
    def_config = DEFAULT_CONFIGS.get(model_name)
    return "_".join(f"{key}_{value}" for key, value in def_config.items())
