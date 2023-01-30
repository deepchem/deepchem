from deepchem.models import TorchModel

class ModularTorchModel(TorchModel):
    
    """Abstract modular torch model class. This class is meant to be subclassed for modular TorchModels, meaning they can be passed to a Pretrainer object.
    
    PretainableTorchModels define methods to build the embedding layers and prediction head layers of the model, and a public attribute to access the embedding. The model is then built by connecting the embedding and head layers. Generating the embedding and head layers separately allows for the pretrainer to generate a model with the same embedding architecture but a different prediction head. This is useful for transfer learning on different tasks which requires specific predictions heads and loss functions.
    """

    @property
    def embedding():
        return NotImplementedError("Subclass must define the embedding")

    def build_embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_head(self):
        return NotImplementedError("Subclass must define the head")

    def build_model(self):
        return NotImplementedError("Subclass must define the model")

class Pretrainer(TorchModel):
    """Abstract pretrainer class. This class is meant to be subclassed for pretraining ModularTorchModels.

    The pretrainer is a TorchModel which takes a ModularTorchModel and duplicates the embedding layers of the model. The pretrainer then adds a new prediction head and loss function to the duplicated embedding layers. The pretrainer is then trained, and the embedding layers can be used to initialize the embedding layers of a ModularTorchModel. This allows for transfer learning on different tasks.
    """

    def __init__(self, torchmodel: ModularTorchModel, **kwargs):
        super().__init__(torchmodel.model, torchmodel.loss, **kwargs)

    @property
    def embedding(self):
        return NotImplementedError("Subclass must define the embedding")

    def build_pretrain_loss(self):
        return NotImplementedError("Subclass must define the pretrain loss")

