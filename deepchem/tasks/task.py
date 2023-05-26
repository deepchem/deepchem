from torch.functional import F

from deepchem.models.losses import SoftmaxCrossEntropy


class Task():

    def __init__(self):
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def loss_func(self, inputs, labels, weights):
        raise NotImplementedError("Subclasses must implement loss_func")


class Regression(Task):

    def __init__(self, num_tasks, **kwargs):
        super().__init__()
        self.output_dim = num_tasks
        self.criterion = F.mse_loss

    def loss_func(self, inputs, labels, weights):
        out = self.model(inputs)
        loss = self.criterion(out, labels)
        return (loss * weights).mean()


class Classification(Task):

    def __init__(self, num_tasks, num_classes, **kwargs):
        super().__init__()
        self.output_dim = num_tasks * num_classes
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.criterion = SoftmaxCrossEntropy()._create_pytorch_loss()

    def loss_func(self, inputs, labels, weights):
        out = self.model(inputs)
        out = F.softmax(out, dim=2)
        loss = self.criterion(out, labels)
        return (loss * weights).mean()
