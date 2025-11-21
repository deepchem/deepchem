import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import logging
from typing import Dict, Type, Optional, Literal, List
from deepchem.models.torch_models.chemnet_layers import Stem, InceptionResnetA, ReductionA, InceptionResnetB, ReductionB, InceptionResnetC
from deepchem.models.torch_models import ModularTorchModel
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, SigmoidCrossEntropy
from deepchem.metrics import to_one_hot
from deepchem.data.datasets import pad_batch

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
    >>> import numpy as np
    >>> import torch
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models.ChemCeption import ChemCeption
    >>> base_filters = 16
    >>> img_size = 80
    >>> n_tasks = 10
    >>> n_classes = 2
    >>> model = ChemCeption(img_spec="std", img_size=img_size, base_filters=base_filters, inception_blocks={"A": 3, "B": 3, "C": 3}, n_tasks=n_tasks, n_classes=n_classes, augment=False, mode="classification")
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = dc.feat.SmilesToImage(img_size=80, img_spec='std')
    >>> images = featurizer.featurize(smiles)
    >>> image = torch.tensor(images, dtype=torch.float32)
    >>> image = image.permute(0, 3, 1, 2)  # Convert to NCHW format
    >>> output = model(image)
    >>> print(output.shape)
    torch.Size([1, 10, 2])
    """

    def __init__(self,
                 stem: nn.Module,
                 inceptionA: nn.Module,
                 reductionA: nn.Module,
                 inceptionB: nn.Module,
                 reductionB: nn.Module,
                 inceptionC: nn.Module,
                 global_avg_pool: nn.Module,
                 fc_classification : Optional[nn.Module] = None,
                 fc_regression: Optional[nn.Module] = None,
                 mode: str = "classification",
                 n_tasks: int = 10,
                 n_classes: int = 2,
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

        self.mode = mode
        self.n_tasks = n_tasks
        self.n_classes = n_classes

        self.stem = stem
        self.inceptionA = inceptionA
        self.reductionA = reductionA
        self.inceptionB = inceptionB
        self.reductionB = reductionB
        self.inceptionC = inceptionC
        self.global_avg_pool = global_avg_pool
        if self.mode == 'classification':
            self.output_layer = fc_classification
        else:
            self.output_layer = fc_regression

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
                final_output = torch.sigmoid(x), x
            else:
                final_output = F.softmax(x, dim=-1), x
        else:
            final_output = x.view(-1, self.n_tasks,1)

        return final_output


class ChemCeptionModular(ModularTorchModel):
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
    >>> from deepchem.models.torch_models.chemception import ChemCeptionModular
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
    >>> pretrain_model = ChemCeptionModular(
    ...     img_size=img_size,
    ...     n_tasks=n_tasks,
    ...     n_classes=n_classes,
    ...     mode='classification',
    ...     learning_rate=1e-4,
    ...     model_dir=model_dir=tempdir.name
    ... )
    >>> pretrain_model.fit(dataset_pt, nb_epoch=2)
    >>> pretrain_model.save_checkpoint()
    >>> finetune_model = ChemCeptionModular(
    ...     img_size=img_size,
    ...     n_tasks=n_tasks,
    ...     n_classes=n_classes,
    ...     mode='regression',
    ...     learning_rate=1e-4
    ...     model_dir=tempdir.name
    ... )
    >>> finetune_model.restore()
    >>> finetuning_loss = finetune_model.fit(dataset_ft,nb_epoch=2)
    >>> finetune_model.predict(dataset_ft)
    """

    def __init__(self,
                 img_spec: str = "std",
                 img_size: int = 80,
                 base_filters: int = 16,
                 inception_blocks: Optional[Dict[str, int]] = DEFAULT_INCEPTION_BLOCKS, 
                 n_tasks: int = 10,
                 n_classes: int = 2,
                 augment: bool = False,
                 mode: Literal['regression',
                               'classification'] = 'classification',
                 **kwargs):

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
            output_types = ['prediction', 'logits']

        # Initialize parent with both model and components
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
            - 'inceptionA': Inception-ResNet-A block stack
            - 'reductionA': Reduction-A layer
            - 'inceptionB': Inception-ResNet-B block stack
            - 'reductionB': Reduction-B layer
            - 'inceptionC': Inception-ResNet-C block stack
            - 'global_avg_pool': Global average pooling
            - 'output_layer': Final linear projection
        """
        in_channels = 1 if self.img_spec == "std" else 4

        components: Dict[str, nn.Module] = {}

        components['stem'] = Stem(in_channels=in_channels, out_channels=self.base_filters)

        components['inceptionA'] = self.build_inception_module(InceptionResnetA, "A",
                                                 self.base_filters,
                                                 self.base_filters)
        components['reductionA'] = ReductionA(self.base_filters, self.base_filters)

        components['inceptionB'] = self.build_inception_module(InceptionResnetB, "B",
                                                 4 * self.base_filters,
                                                 self.base_filters)
        components['reductionB'] = ReductionB(4 * self.base_filters, self.base_filters)

        current_channels = int(
            torch.floor(torch.tensor(7.875 * self.base_filters)).item())
        components['inceptionC'] = self.build_inception_module(InceptionResnetC, "C",
                                                 current_channels,
                                                 self.base_filters)

        components['global_avg_pool'] = nn.AdaptiveAvgPool2d(1)

        if self.mode == "classification":
            components['fc_classification'] = nn.Linear(current_channels,
                                     self.n_tasks * self.n_classes)
        else:
            components['fc_regression'] = nn.Linear(current_channels, self.n_tasks)

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
        n = self.inception_blocks.get(block_key, 0)
        return nn.Sequential(*[block_cls(*args, **kwargs) for _ in range(n)])

    def build_model(self) -> nn.Module:
        """Assemble ChemCeption model from components."""
        return ChemCeption(**self.components, mode=self.mode, n_tasks=self.n_tasks, n_classes=self.n_classes)

    def loss_func(self, inputs, labels, weights) -> torch.Tensor:
        """Compute the loss depending on mode (regression/classification)."""
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
        # The following code is taken from _StandardLoss class inside torch_model.py, 
        # as the class cannot be imported here due to its absense in the __init__.py file
    
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

        for epoch in range(epochs):
            if mode == "predict" or (not self.augment):
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):

                    in_channels = self.components['stem'].conv_layer.in_channels
                    if X_b.shape[1] != in_channels and X_b.shape[-1] == in_channels:
                        X_b = np.transpose(X_b, (0, 3, 1, 2))

                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)

                    yield ([X_b], [y_b], [w_b])

            else:

                augment_fn = transforms.v2.RandomRotation(180)
                
                for (X_b, y_b, w_b,
                     ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                                   deterministic=deterministic,
                                                   pad_batches=pad_batches):

                    in_channels = self.components['stem'].conv_layer.in_channels
                    if X_b.shape[1] != in_channels and X_b.shape[-1] == in_channels:
                        X_b = np.transpose(X_b, (0, 3, 1, 2))

                    X_b = augment_fn(X_b)

                    if self.mode == 'classification':
                        y_b = to_one_hot(y_b.flatten(), self.n_classes).reshape(
                            -1, self.n_tasks, self.n_classes)
                    yield ([X_b], [y_b], [w_b])

    def restore(  # type: ignore
            self,
            components: Optional[List[str]] = None,
            checkpoint: Optional[str] = None,
            model_dir: Optional[str] = None) -> None:
        """
        Restores the state of a ModularTorchModel from a checkpoint file.

        If no checkpoint file is provided, it will use the latest checkpoint found in the model directory. 
        If a list of component names is provided, only the state of those components will be restored.
        Optimizer state dict and global step is not restored.

        Parameters
        ----------
        components: Optional[List[str]]
            A list of component names to restore. If None, all components will be restored.
        checkpoint: Optional[str]
            The path to the checkpoint file. If None, the latest checkpoint in the model directory will
            be used.
        model_dir: Optional[str]
            The path to the model directory. If None, the model directory used to initialize the model will be used.
        """
        logger.info('Restoring model')
        if checkpoint is None:
            checkpoints = sorted(self.get_checkpoints(model_dir))
            if len(checkpoints) == 0:
                raise ValueError('No checkpoint found')
            checkpoint = checkpoints[0]
        data = torch.load(checkpoint)
        for name, state_dict in data.items():
            if name != 'model' and name in self.components.keys():
                if components is None or name in components:
                    self.components[name].load_state_dict(state_dict)

        self.build_model()