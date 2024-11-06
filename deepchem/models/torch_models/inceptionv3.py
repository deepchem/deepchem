import torch
import torch.nn as nn
import torch.nn.functional as F
from deepchem.models.losses import CategoricalCrossEntropy
from deepchem.models.torch_models import TorchModel
from deepchem.data import Dataset
from deepchem.models.optimizers import RMSProp
from typing import Optional, List, Callable, Any


class InceptionV3(nn.Module):

    def __init__(self,
                 input_shape,
                 num_classes=1000,
                 aux_logits=True,
                 dropout_rate=0.5):
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.input_shape = input_shape

        # Initial layers
        self.Conv2d_1a_3x3 = BasicConv2d(input_shape[0],
                                         32,
                                         kernel_size=3,
                                         stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)

        # Additional convolutional layers
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(3, stride=2)

        # Inception blocks
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # Auxiliary classifier (only used during training)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        # Final Inception blocks
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Forward pass through initial layers
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)

        # Forward pass through Inception blocks
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        # Auxiliary output if training
        if self.aux_logits and self.training:
            aux = self.AuxLogits(x)

        # Final Inception blocks
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_logits and self.training:
            return x, aux
        else:
            return x


# Helper layers
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels,
                                       pool_features,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(channels_7x7,
                                       channels_7x7,
                                       kernel_size=(1, 7),
                                       padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(channels_7x7,
                                       192,
                                       kernel_size=(7, 1),
                                       padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels,
                                          channels_7x7,
                                          kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(1, 7),
                                          padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channels_7x7,
                                          channels_7x7,
                                          kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channels_7x7,
                                          192,
                                          kernel_size=(1, 7),
                                          padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192,
                                         192,
                                         kernel_size=(1, 7),
                                         padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192,
                                         192,
                                         kernel_size=(7, 1),
                                         padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384,
                                        384,
                                        kernel_size=(1, 3),
                                        padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384,
                                        384,
                                        kernel_size=(3, 1),
                                        padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384,
                                           384,
                                           kernel_size=(1, 3),
                                           padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384,
                                           384,
                                           kernel_size=(3, 1),
                                           padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl)
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class InceptionV3Model(TorchModel):
    """
    InceptionV3 model adapted for variant calling with DeepVariant configuration settings.

    Parameters
    ----------
    input_shape: tuple
        Shape of the input image, here set to (6, 100, 221) for DeepVariant.
    """

    def __init__(self, input_shape=(6, 100, 221), **kwargs):
        # Fixed hyperparameters
        learning_rate = 0.001
        decay_steps = 2  # epochs per decay
        decay_rate = 0.947
        warmup_steps = 10000
        rho = 0.9
        momentum = 0.9
        epsilon = 1.0
        # weight_decay = 0.00004
        # dropout_rate = 0.2

        # Initialize the InceptionV3 model architecture
        model = InceptionV3(
            num_classes=3, input_shape=input_shape, aux_logits=False
        )  # Set aux_logits to False to disable auxiliary output

        loss = CategoricalCrossEntropy()

        # Define optimizer as DeepChem's RMSProp
        optimizer = RMSProp(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=rho,  # Using decay as rho
            epsilon=epsilon)

        # Custom attributes for learning rate decay and warmup
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Initialize base TorchModel
        super(InceptionV3Model, self).__init__(model=model,
                                               optimizer=optimizer,
                                               loss=loss,
                                               **kwargs)

    def adjust_learning_rate(self):
        """
        Adjusts learning rate manually based on warmup and decay steps.
        """
        if self.current_step < self.warmup_steps:
            lr = self.learning_rate * (self.current_step / self.warmup_steps)
        else:
            decay_factor = self.decay_rate**(self.current_step //
                                             self.decay_steps)
            lr = self.learning_rate * decay_factor

        self.optimizer.learning_rate = lr


def fit(self,
        dataset: Dataset,
        nb_epoch: int = 1,
        max_checkpoints_to_keep: int = 5,
        checkpoint_interval: int = 1000,
        deterministic: bool = False,
        restore: bool = False,
        variables: Optional[List] = None,
        loss: Optional[Callable[[List[Any], List[Any], List[Any]], Any]] = None,
        callbacks: Optional[List[Callable[..., Any]]] = None,
        all_losses: Optional[List[float]] = None) -> float:
    """
    Trains the model on the given dataset, adjusting learning rate with warmup and decay.

    Parameters
    ----------
    dataset: Dataset
        Dataset to be used for training.
    nb_epoch: int, optional (default 1)
        Number of epochs to train the model.
    max_checkpoints_to_keep: int, optional
        Number of checkpoints to keep.
    checkpoint_interval: int, optional
        Interval for saving checkpoints.
    deterministic: bool, optional
        If True, runs in deterministic mode.
    restore: bool, optional
        If True, restores the model from the last checkpoint.
    variables: list, optional
        List of parameters to train.
    loss: callable, optional
        Custom loss function.
    callbacks: list of callables, optional
        Custom callbacks.
    all_losses: list, optional (default None)
        List to store all losses during training.

    Returns
    -------
    float
        The final loss value after training.
    """
    if all_losses is None:
        all_losses = []

    for epoch in range(nb_epoch):
        self.current_step = epoch
        self.adjust_learning_rate()  # Adjust learning rate before each epoch

        # Perform one epoch of training
        epoch_loss = super(InceptionV3Model, self).fit(
            dataset,
            nb_epoch=1,
            max_checkpoints_to_keep=max_checkpoints_to_keep,
            checkpoint_interval=checkpoint_interval,
            deterministic=deterministic,
            restore=restore,
            variables=variables,
            loss=loss,
            callbacks=callbacks)

        # Store the numeric loss from this epoch
        all_losses.append(epoch_loss)

    return all_losses[-1] if all_losses else 0.0
