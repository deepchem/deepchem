import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Type
from deepchem.models.torch_models.chemnet_layers import Stem, InceptionResnetA, ReductionA, InceptionResnetB, ReductionB, InceptionResnetC

DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}


class ChemCeption(nn.Module):
    """
    Implements the ChemCeption model that leverages the representational capacities
    of convolutional neural networks (CNNs) to predict molecular properties.

    The model is based on the description in Goh et al., "Chemception: A Deep
    Neural Network with Minimal Chemistry Knowledge Matches the Performance of
    Expert-developed QSAR/QSPR Models" (https://arxiv.org/pdf/1706.06689.pdf).
    The authors use an image based representation of the molecule, where pixels
    encode different atomic and bond properties. More details on the image repres-
    entations can be found at https://arxiv.org/abs/1710.02238

    The model consists of a Stem Layer that reduces the image resolution for the
    layers to follow. The output of the Stem Layer is followed by a series of
    Inception-Resnet blocks & a Reduction layer. Layers in the Inception-Resnet
    blocks process image tensors at multiple resolutions and use a ResNet style
    skip-connection, combining features from different resolutions. The Reduction
    layers reduce the spatial extent of the image by max-pooling and 2-strided
    convolutions. More details on these layers can be found in the ChemCeption
    paper referenced above. The output of the final Reduction layer is subject to
    a Global Average Pooling, and a fully-connected layer maps the features to
    downstream outputs.

    In the ChemCeption paper, the authors perform real-time image augmentation by
    rotating images between 0 to 180 degrees. This can be done during model
    training by setting the augment argument to True.

    Example
    -------
    >>> import numpy as np
    >>> import torch
    >>> from deepchem.models.torch_models.ChemCeption import ChemCeption
    >>> base_filters = 16
    >>> img_size = 80
    >>> n_tasks = 10
    >>> n_classes = 2
    >>> model = ChemCeption(img_spec="std", img_size=img_size, base_filters=base_filters, inception_blocks={"A": 3, "B": 3, "C": 3}, n_tasks=n_tasks, n_classes=n_classes, augment=False, mode="classification")
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = deepchem.feat.SmilesToImage(img_size=80, img_spec='std')
    >>> images = featurizer.featurize(smiles)
    >>> image = torch.tensor(images, dtype=torch.float32)
    >>> image = image.permute(0, 3, 1, 2)  # Convert to NCHW format
    >>> output = model(image)
    >>> print(output.shape)
    torch.Size([1, 10, 2])
    """

    def __init__(self,
                 img_spec: str = "std",
                 img_size: int = 80,
                 base_filters: int = 16,
                 inception_blocks: Dict = DEFAULT_INCEPTION_BLOCKS,
                 n_tasks: int = 10,
                 n_classes: int = 2,
                 augment: bool = False,
                 mode: str = "regression",
                 **kwargs) -> None:
        """
        Parameters
        ----------
        img_spec: str, default std
            Image specification used
        img_size: int, default 80
            Image size used
        base_filters: int, default 16
            Base filters used for the different inception and reduction layers
        inception_blocks: dict,
            Dictionary containing number of blocks for every inception layer
        n_tasks: int, default 10
            Number of classification or regression tasks
        n_classes: int, default 2
            Number of classes (used only for classification)
        augment: bool, default False
            Whether to augment images
        mode: str, default regression
            Whether the model is used for regression or classification

        """
        super(ChemCeption, self).__init__()

        assert mode in ["classification", "regression"
                       ], "Mode must be 'classification' or 'regression'"

        if inception_blocks is None:
            inception_blocks = {"A": 3, "B": 3, "C": 3}

        self.img_spec = img_spec
        self.img_size = img_size
        self.base_filters = base_filters
        self.inception_blocks = inception_blocks
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.augment = augment
        self.mode = mode

        in_channels = 1 if img_spec == "std" else 4

        self.stem = Stem(in_channels=in_channels, out_channels=base_filters)
        self.inceptionA = self.build_inception_module(InceptionResnetA, "A",
                                                      base_filters,
                                                      base_filters)
        self.reductionA = ReductionA(base_filters, base_filters)
        self.inceptionB = self.build_inception_module(InceptionResnetB, "B",
                                                      4 * base_filters,
                                                      base_filters)
        self.reductionB = ReductionB(4 * base_filters, base_filters)
        self.inceptionC = self.build_inception_module(
            InceptionResnetC, "C",
            int(torch.floor(torch.tensor(7.875 * base_filters)).item()),
            base_filters)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.current_channels = int(
            torch.floor(torch.tensor(7.875 * base_filters)).item())

        if mode == "classification":
            self.output_layer = nn.Linear(self.current_channels,
                                          n_tasks * n_classes)
        else:
            self.output_layer = nn.Linear(self.current_channels, n_tasks)

    def build_inception_module(self, block_cls: Type[nn.Module], block_key: str,
                               *args, **kwargs) -> nn.Sequential:
        """
        Build a sequential stack of Inception blocks.

        Parameters
        ----------
        block_cls : class
            Inception block class (A/B/C)
        block_key : str
            Key to fetch number of blocks
        *args, **kwargs : passed to block constructor

        Returns
        -------
        nn.Sequential
        """
        n = self.inception_blocks.get(block_key, 0)
        return nn.Sequential(*[block_cls(*args, **kwargs) for _ in range(n)])

    def forward(self, x) -> torch.Tensor:
        x = self.stem(x)
        x = self.inceptionA(x)
        x = self.reductionA(x)
        x = self.inceptionB(x)
        x = self.reductionB(x)
        x = self.inceptionC(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.output_layer(x)

        if self.mode == "classification":
            x = x.view(-1, self.n_tasks, self.n_classes)
            if self.n_classes == 2:
                x = torch.sigmoid(x)
            else:
                x = F.softmax(x, dim=-1)
        else:
            x = x.view(-1, self.n_tasks)

        return x
