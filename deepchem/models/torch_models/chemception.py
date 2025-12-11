import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Type, Optional, Literal, Sequence
from deepchem.utils.typing import OneOrMany
from deepchem.models.torch_models.chemnet_layers import Stem, InceptionResnetA, ReductionA, InceptionResnetB, ReductionB, InceptionResnetC
from deepchem.models.torch_models import ModularTorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, SigmoidCrossEntropy
from deepchem.metrics import to_one_hot

logger = logging.getLogger(__name__)
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
    >>> import torch.nn as nn
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.chemnet_layers import Stem, InceptionResnetA, InceptionResnetB, InceptionResnetC, ReductionA, ReductionB
    >>> from deepchem.models.torch_models import ChemCeption
    >>> DEFAULT_INCEPTION_BLOCKS = {"A": 3, "B": 3, "C": 3}
    >>> base_filters = 16
    >>> img_spec = 'std'
    >>> img_size = 80
    >>> n_tasks = 10
    >>> n_classes = 2
    >>> in_channels = 1 if img_spec == "std" else 4
    >>> mode = 'classification'
    >>> components = {}
    >>> components['stem'] = Stem(in_channels=in_channels, out_channels=base_filters)
    >>> components['inception_resnet_A'] = nn.Sequential(*[InceptionResnetA(base_filters, base_filters) for _ in range(DEFAULT_INCEPTION_BLOCKS['A'])])
    >>> components['reduction_A'] = ReductionA(base_filters, base_filters)
    >>> components['inception_resnet_B'] = nn.Sequential(*[InceptionResnetB(4*base_filters, base_filters) for _ in range(DEFAULT_INCEPTION_BLOCKS['B'])])
    >>> components['reduction_B'] = ReductionB(4 * base_filters, base_filters)
    >>> current_channels = int(torch.floor(torch.tensor(7.875 * base_filters)).item())
    >>> components['inception_resnet_C'] = nn.Sequential(*[InceptionResnetC(current_channels, base_filters) for _ in range(DEFAULT_INCEPTION_BLOCKS['C'])])
    >>> components['global_avg_pool'] = nn.AdaptiveAvgPool2d(1)
    >>> if mode == "classification":
    ...     components['fc_classification'] = nn.Linear(current_channels, n_tasks * n_classes)
    ... else:
    ...     components['fc_regression'] = nn.Linear(current_channels,n_tasks)
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = dc.feat.SmilesToImage(img_size=img_size, img_spec='std')
    >>> images = featurizer.featurize(smiles)
    >>> image = torch.tensor(images, dtype=torch.float32)
    >>> if mode == 'classification':
    ...        output_layer = components['fc_classification']
    ... else:
    ...        output_layer = components['fc_regression']
    >>> input = image.permute(0, 3, 1, 2) # to convert from channel last  (N,H,W,C) to pytorch default channel first (N,C,H,W) representation
    >>> model = ChemCeption(stem=components['stem'],
    ...                       inception_resnet_A=components['inception_resnet_A'],
    ...                       reduction_A=components['reduction_A'],
    ...                       inception_resnet_B=components['inception_resnet_B'],
    ...                       reduction_B=components['reduction_B'],
    ...                       inception_resnet_C=components['inception_resnet_C'],
    ...                       global_avg_pool=components['global_avg_pool'],
    ...                       output_layer=output_layer,
    ...                       mode=mode,
    ...                       n_tasks=n_tasks,
    ...                       n_classes=n_classes)
    >>> output = model(input)

    References
    ----------
    .. [1] Goh et al. "Chemception: A Deep Neural Network with Minimal Chemistry Knowledge Matches
        the Performance of Expert-developed QSAR/QSPR Models" (https://arxiv.org/abs/1706.06689)
    """

    def __init__(self,
                 stem: nn.Module,
                 inception_resnet_A: nn.Module,
                 reduction_A: nn.Module,
                 inception_resnet_B: nn.Module,
                 reduction_B: nn.Module,
                 inception_resnet_C: nn.Module,
                 global_avg_pool: nn.Module,
                 output_layer: nn.Module,
                 mode: str = "classification",
                 n_tasks: int = 10,
                 n_classes: int = 2,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        stem: nn.Module
            Stem layer that serves as the initial processing block in ChemCeption.
        inception_resnet_A: nn.Module
            Inception-ResNet-A block from the Inception-ResNet architecture.
        reduction_A: nn.Module
            Reduction-A block from the Inception-ResNet architecture.
        inception_resnet_B: nn.Module
            Inception-ResNet-B block from the Inception-ResNet architecture.
        reduction_B: nn.Module
            Reduction-B block from the Inception-ResNet architecture.
        inception_resnet_C:
            Inception-ResNet-C block from the Inception-ResNet architecture.
        global_avg_pool: nn.Module
            2D Average Pooling layer
        fc_classification: nn.Module
            A fully connected neural network to be used as the prediction head for classification
        fc_regression: nn.Module
            A fully connected neural network to be used as the prediction head for regression
        mode: str, default regression
            The model type, 'classification' or 'regression'.
        n_tasks: int, default 10
            Number of classification or regression tasks
        n_classes: int, default 2
            Number of classes (used only for classification)
        """
        super(ChemCeption, self).__init__()

        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        self.stem = stem
        self.inception_resnet_A = inception_resnet_A
        self.reduction_A = reduction_A
        self.inception_resnet_B = inception_resnet_B
        self.reduction_B = reduction_B
        self.inception_resnet_C = inception_resnet_C
        self.global_avg_pool = global_avg_pool
        self.output_layer = output_layer

    def forward(self, x: torch.Tensor) -> OneOrMany[torch.Tensor]:
        x = self.stem(x)
        x = self.inception_resnet_A(x)
        x = self.reduction_A(x)
        x = self.inception_resnet_B(x)
        x = self.reduction_B(x)
        x = self.inception_resnet_C(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.output_layer(x)

        if self.mode == "classification":
            x = x.view(-1, self.n_tasks, self.n_classes)
            if self.n_classes == 2:
                prob = torch.sigmoid(x)
            else:
                prob = F.softmax(x, dim=-1)
            return prob, x
        else:
            return x.view(-1, self.n_tasks, 1)


class ChemCeptionModel(ModularTorchModel):
    """
    Modular wrapper around ChemCeption for flexible pretraining and finetuning.

    This class provides a `ModularTorchModel` interface for ChemCeption. It
    allows building, training, and fine-tuning ChemCeption with configurable
    inception blocks, tasks, and modes (regression or classification).

    Parameters
    ----------
    task : {'pretraining', 'finetuning'}, default='finetuning'
        Whether the model is used for pretraining (frozen components)
        or finetuning (trainable components).
    img_size : int, default=80
        Size of the input image (height and width).
    base_filters : int, default=16
        Number of filters
    n_tasks : int, default=10
        Number of prediction tasks.
    n_classes : int, optional, default=2
        Number of output classes per task (classification only).
    mode : {'regression', 'classification'}, default='classification'
        Determines whether the model outputs regression values or
        class probabilities.
    img_spec : str, default='std'
        Image specification, determines input channels.
        `'std'` → 1 channel, otherwise 4 channels.
    inception_blocks : dict, optional
        Dictionary controlling the number of Inception-ResNet blocks per stage.
        Example: ``{"A": 3, "B": 3, "C": 3}``.
    augment : bool, default=False
        If True, enable real-time image augmentation during training.
    **kwargs : dict
        Additional keyword arguments passed to `ModularTorchModel`.
    Examples
    --------
    Pretraining and finetuning workflow:

    >>> import numpy as np
    >>> import deepchem as dc
    >>> from deepchem.feat import SmilesToImage
    >>> from deepchem.models.torch_models.chemception import ChemCeptionModel
    >>> import tempfile
    >>> tempdir = tempfile.TemporaryDirectory()
    >>> n_samples = 6
    >>> img_size = 80
    >>> n_tasks = 10
    >>> n_classes = 2
    >>> smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "C1CCCCC1", "O=C=O"]
    >>> y_pretrain = np.random.randint(0, n_classes, (n_samples, n_tasks)).astype(np.float32)
    >>> y_finetune = np.random.randint(0, n_classes, (n_samples, n_tasks)).astype(np.float32)
    >>> featurizer = SmilesToImage(img_size=img_size, img_spec='std')
    >>> X_images = featurizer.featurize(smiles_list)
    >>> X_images = np.array([img.squeeze() for img in X_images])[:, np.newaxis, :, :]
    >>> dataset_pt = dc.data.NumpyDataset(X_images, y_pretrain)
    >>> dataset_ft = dc.data.NumpyDataset(X_images, y_finetune)
    >>> pretrain_model = ChemCeptionModel(
    ...     img_size=img_size,
    ...     n_tasks=n_tasks,
    ...     n_classes=n_classes,
    ...     mode='classification',
    ...     learning_rate=1e-4,
    ...     model_dir=tempdir.name
    ... )
    >>> pretrain_loss = pretrain_model.fit(dataset_pt, nb_epoch=2)
    >>> pretrain_model.save_checkpoint()
    >>> finetune_model = ChemCeptionModel(
    ...     img_size=img_size,
    ...     n_tasks=n_tasks,
    ...     n_classes=n_classes,
    ...     mode='regression',
    ...     learning_rate=1e-4,
    ...     model_dir=tempdir.name
    ... )
    >>> finetune_model.load_from_pretrained(source_model=pretrain_model,
    ...                                 components=[
    ...                                  'stem', 'inception_resnet_A', 'inception_resnet_B',
    ...                                  'inception_resnet_C', 'reduction_A', 'reduction_B'
    ...                              ])
    >>> finetuning_loss = finetune_model.fit(dataset_ft,nb_epoch=1)
    >>> predictions = finetune_model.predict(dataset_ft)
    """

    def __init__(
            self,
            img_spec: str = "std",
            img_size: int = 80,
            base_filters: int = 16,
            inception_blocks: Optional[Dict[str,
                                            int]] = DEFAULT_INCEPTION_BLOCKS,
            n_tasks: int = 10,
            n_classes: int = 2,
            augment: bool = False,
            mode: Literal['regression', 'classification'] = 'classification',
            **kwargs):
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
        self.img_size = img_size
        self.base_filters = base_filters
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.mode = mode
        self.img_spec = img_spec
        self.inception_blocks = inception_blocks or DEFAULT_INCEPTION_BLOCKS
        self.augment = augment
        self.kwargs = kwargs

        # Build components first
        self.components = self.build_components()

        # Build the model
        self.model = self.build_model()

        # Define output types
        if mode == 'regression':
            output_types = ['prediction']
        else:
            output_types = ['prediction', 'loss']

        super().__init__(model=self.model,
                         components=self.components,
                         output_types=output_types,
                         **kwargs)

    def build_components(self) -> Dict[str, nn.Module]:
        """
        Build and return the modular components of ChemCeption.

        Returns
        -------
        components : dict of str -> nn.Module
            Dictionary containing the model components:
            - 'stem': initial convolutional stem
            - 'inception_resnet_A': Inception-ResNet-A block stack
            - 'reduction_A': Reduction-A layer
            - 'inception_resnet_B': Inception-ResNet-B block stack
            - 'reduction_B': Reduction-B layer
            - 'inception_resnet_C': Inception-ResNet-C block stack
            - 'global_avg_pool': Global average pooling
            - 'output_layer': Final linear projection
        """
        in_channels = 1 if self.img_spec == "std" else 4

        components: Dict[str, nn.Module] = {}

        components['stem'] = Stem(in_channels=in_channels,
                                  out_channels=self.base_filters)

        components['inception_resnet_A'] = self.build_inception_module(
            InceptionResnetA, "A", self.base_filters, self.base_filters)
        components['reduction_A'] = ReductionA(self.base_filters,
                                               self.base_filters)

        components['inception_resnet_B'] = self.build_inception_module(
            InceptionResnetB, "B", 4 * self.base_filters, self.base_filters)
        components['reduction_B'] = ReductionB(4 * self.base_filters,
                                               self.base_filters)

        current_channels = int(
            torch.floor(torch.tensor(7.875 * self.base_filters)).item())
        components['inception_resnet_C'] = self.build_inception_module(
            InceptionResnetC, "C", current_channels, self.base_filters)
        components['global_avg_pool'] = nn.AdaptiveAvgPool2d(1)

        if self.mode == "classification":
            components['fc_classification'] = nn.Linear(
                current_channels, self.n_tasks * self.n_classes)
        else:
            components['fc_regression'] = nn.Linear(current_channels,
                                                    self.n_tasks)

        return components

    def build_inception_module(self, block_cls: Type[nn.Module], block_key: str,
                               *args, **kwargs) -> nn.Sequential:
        """
        Build a sequential stack of Inception blocks.

        Parameters
        ----------
        block_cls : Type[nn.Module]
            Class of the Inception-ResNet block (A, B, or C).
        block_key : str
            Key indicating which block type ('A', 'B', 'C').
        *args, **kwargs
            Arguments passed to the block constructor.

        Returns
        -------
        nn.Sequential
            Sequential module containing stacked Inception blocks.
        """
        n = self.inception_blocks[block_key]
        return nn.Sequential(*[block_cls(*args, **kwargs) for _ in range(n)])

    def build_model(self) -> nn.Module:
        """Assemble ChemCeption model from components."""
        if self.mode == 'classification':
            output_layer = self.components['fc_classification']
        else:
            output_layer = self.components['fc_regression']
        return ChemCeption(
            stem=self.components['stem'],
            inception_resnet_A=self.components['inception_resnet_A'],
            reduction_A=self.components['reduction_A'],
            inception_resnet_B=self.components['inception_resnet_B'],
            reduction_B=self.components['reduction_B'],
            inception_resnet_C=self.components['inception_resnet_C'],
            global_avg_pool=self.components['global_avg_pool'],
            output_layer=output_layer,
            mode=self.mode,
            n_tasks=self.n_tasks,
            n_classes=self.n_classes)

    def loss_func(self, inputs: OneOrMany[torch.Tensor], labels: Sequence,
                  weights: Sequence) -> torch.Tensor:
        """Compute the loss depending on the mode (regression/classification)."""
        outputs = self.model(inputs)

        if isinstance(labels, list) and len(labels) == 1:
            labels = labels[0]

        if self.mode == 'regression':
            loss_fn = L2Loss()._create_pytorch_loss()

        elif self.mode == 'classification':
            if self.n_classes == 2:
                loss_fn = SigmoidCrossEntropy()._create_pytorch_loss()
            else:
                loss_fn = SoftmaxCrossEntropy()._create_pytorch_loss()

            if self._loss_outputs is not None:
                outputs = [outputs[i] for i in self._loss_outputs]
                outputs = outputs[0]

        losses = loss_fn(outputs, labels)

        # To ensure weights and losses have the same shape and multiply losses with weights.
        w = weights[0]
        if len(w.shape) < len(losses.shape):
            if isinstance(w, torch.Tensor):
                shape = tuple(w.shape)
            else:
                shape = w.shape
            shape = tuple(-1 if x is None else x for x in shape)
            w = w.reshape(shape + (1,) * (len(losses.shape) - len(w.shape)))

        loss = losses * w
        loss = loss.mean()

        if self.regularization_loss is not None:
            loss += self.model.regularization_loss()

        return loss

    def default_generator(self,
                          dataset,
                          epochs=1,
                          mode='fit',
                          deterministic=True,
                          pad_batches=True):
        """ Create a generator that iterates batches for a dataset.

        Parameters
        ----------
        dataset: Dataset
            the data to iterate
        epochs: int
            the number of times to iterate over the full dataset
        mode: str
            allowed values are 'fit' (called during training), 'predict' (called
            during prediction), and 'uncertainty' (called during uncertainty
            prediction)
        deterministic: bool
            whether to iterate over the dataset in order, or randomly shuffle the
            data for each epoch
        pad_batches: bool
            whether to pad each batch up to this model's preferred batch size

        Returns
        -------
        a generator that iterates batches, each represented as a tuple of lists:
        ([inputs], [outputs], [weights]) """

        for epoch in range(epochs):
            if mode == "predict" or (not self.augment):
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):

                    in_channels = self.components['stem'].conv_layer.in_channels
                    if X_b.shape[1] != in_channels and X_b.shape[
                            -1] == in_channels:
                        X_b = np.transpose(X_b, (0, 3, 1, 2))

                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)

                    yield ([X_b], [y_b], [w_b])

            else:
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):

                    in_channels = self.components['stem'].conv_layer.in_channels
                    if X_b.shape[1] != in_channels and X_b.shape[
                            -1] == in_channels:
                        X_b = np.transpose(X_b, (0, 3, 1, 2))

                    N = len(X_b)
                    angles = np.random.uniform(-180, 180, size=N)
                    X_b = np.stack([
                        scipy.ndimage.rotate(x,
                                             angle=a,
                                             axes=(1, 2),
                                             reshape=False,
                                             order=1,
                                             mode='nearest')
                        for x, a in zip(X_b, angles)
                    ],
                                   axis=0)

                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)
                    yield ([X_b], [y_b], [w_b])
