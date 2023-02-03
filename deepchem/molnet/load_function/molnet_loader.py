"""
Common code for loading MoleculeNet datasets.
"""
import os
import logging
import deepchem as dc
from deepchem.data import Dataset, DiskDataset
from typing import List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class TransformerGenerator(object):
    """Create Transformers for Datasets.

    When loading molnet datasets, you cannot directly pass in Transformers
    to use because many Transformers require the Dataset they will be applied to
    as a constructor argument.  Instead you pass in TransformerGenerator objects
    which can create the Transformers once the Dataset is loaded.
    """

    def __init__(self, transformer_class: Type[dc.trans.Transformer], **kwargs):
        """Construct an object for creating Transformers.

        Parameters
        ----------
        transformer_class: Type[Transformer]
            the class of Transformer to create
        kwargs:
            any additional arguments are passed to the Transformer's constructor
        """
        self.transformer_class = transformer_class
        self.kwargs = kwargs

    def create_transformer(self, dataset: Dataset) -> dc.trans.Transformer:
        """Construct a Transformer for a Dataset."""
        return self.transformer_class(dataset=dataset, **self.kwargs)

    def get_directory_name(self) -> str:
        """Get a name for directories on disk describing this Transformer."""
        name = self.transformer_class.__name__
        for key, value in self.kwargs.items():
            if isinstance(value, list):
                continue
            name += '_' + key + '_' + str(value)
        return name


featurizers = {
    'ecfp': dc.feat.CircularFingerprint(size=1024),
    'graphconv': dc.feat.ConvMolFeaturizer(),
    'raw': dc.feat.RawFeaturizer(),
    'onehot': dc.feat.OneHotFeaturizer(),
    'smiles2img': dc.feat.SmilesToImage(img_size=80, img_spec='std'),
    'weave': dc.feat.WeaveFeaturizer(),
}

splitters = {
    'index': dc.splits.IndexSplitter(),
    'random': dc.splits.RandomSplitter(),
    'scaffold': dc.splits.ScaffoldSplitter(),
    'butina': dc.splits.ButinaSplitter(),
    'fingerprint': dc.splits.FingerprintSplitter(),
    'task': dc.splits.TaskSplitter(),
    'stratified': dc.splits.RandomStratifiedSplitter()
}

transformers = {
    'balancing':
        TransformerGenerator(dc.trans.BalancingTransformer),
    'normalization':
        TransformerGenerator(dc.trans.NormalizationTransformer,
                             transform_y=True),
    'minmax':
        TransformerGenerator(dc.trans.MinMaxTransformer, transform_y=True),
    'clipping':
        TransformerGenerator(dc.trans.ClippingTransformer, transform_y=True),
    'log':
        TransformerGenerator(dc.trans.LogTransformer, transform_y=True)
}


class _MolnetLoader(object):
    """The class provides common functionality used by many molnet loader functions.
       It is an abstract class.  Subclasses implement loading of particular datasets.
    """

    def __init__(self, featurizer: Union[dc.feat.Featurizer, str],
                 splitter: Union[dc.splits.Splitter, str, None],
                 transformer_generators: List[Union[TransformerGenerator,
                                                    str]], tasks: List[str],
                 data_dir: Optional[str], save_dir: Optional[str], **kwargs):
        """Construct an object for loading a dataset.

        Parameters
        ----------
        featurizer: Featurizer or str
            the featurizer to use for processing the data.  Alternatively you can pass
            one of the names from dc.molnet.featurizers as a shortcut.
        splitter: Splitter or str
            the splitter to use for splitting the data into training, validation, and
            test sets.  Alternatively you can pass one of the names from
            dc.molnet.splitters as a shortcut.  If this is None, all the data
            will be included in a single dataset.
        transformer_generators: list of TransformerGenerators or strings
            the Transformers to apply to the data.  Each one is specified by a
            TransformerGenerator or, as a shortcut, one of the names from
            dc.molnet.transformers.
        tasks: List[str]
            the names of the tasks in the dataset
        data_dir: str
            a directory to save the raw data in
        save_dir: str
            a directory to save the dataset in
        """
        if 'split' in kwargs:
            splitter = kwargs['split']
            logger.warning("'split' is deprecated.  Use 'splitter' instead.")
        if isinstance(featurizer, str):
            featurizer = featurizers[featurizer.lower()]
        if isinstance(splitter, str):
            splitter = splitters[splitter.lower()]
        if data_dir is None:
            data_dir = dc.utils.data_utils.get_data_dir()
        if save_dir is None:
            save_dir = dc.utils.data_utils.get_data_dir()
        self.featurizer = featurizer
        self.splitter = splitter
        self.transformers = [
            transformers[t.lower()] if isinstance(t, str) else t
            for t in transformer_generators
        ]
        self.tasks = list(tasks)
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.args = kwargs

    def load_dataset(
        self, name: str, reload: bool
    ) -> Tuple[List[str], Tuple[Dataset, ...], List[dc.trans.Transformer]]:
        """Load the dataset.

        Parameters
        ----------
        name: str
            the name of the dataset, used to identify the directory on disk
        reload: bool
            if True, the first call for a particular featurizer and splitter will cache
            the datasets to disk, and subsequent calls will reload the cached datasets.
        """
        # Build the path to the dataset on disk.

        featurizer_name = str(self.featurizer)
        splitter_name = 'None' if self.splitter is None else str(self.splitter)
        save_folder = os.path.join(self.save_dir, name + "-featurized",
                                   featurizer_name, splitter_name)
        if len(self.transformers) > 0:
            transformer_name = '_'.join(
                t.get_directory_name() for t in self.transformers)
            save_folder = os.path.join(save_folder, transformer_name)

        # Try to reload cached datasets.

        if reload:
            if self.splitter is None:
                if os.path.exists(save_folder):
                    transformers = dc.utils.data_utils.load_transformers(
                        save_folder)
                    return self.tasks, (DiskDataset(save_folder),), transformers
            else:
                loaded, all_dataset, transformers = dc.utils.data_utils.load_dataset_from_disk(
                    save_folder)
                if all_dataset is not None:
                    return self.tasks, all_dataset, transformers

        # Create the dataset

        logger.info("About to featurize %s dataset." % name)
        dataset = self.create_dataset()

        # Split and transform the dataset.

        if self.splitter is None:
            transformer_dataset: Dataset = dataset
        else:
            logger.info("About to split dataset with {} splitter.".format(
                self.splitter.__class__.__name__))
            train, valid, test = self.splitter.train_valid_test_split(dataset)
            transformer_dataset = train
        transformers = [
            t.create_transformer(transformer_dataset) for t in self.transformers
        ]
        logger.info("About to transform data.")
        if self.splitter is None:
            for transformer in transformers:
                dataset = transformer.transform(dataset)
            if reload and isinstance(dataset, DiskDataset):
                dataset.move(save_folder)
                dc.utils.data_utils.save_transformers(save_folder, transformers)
            return self.tasks, (dataset,), transformers

        for transformer in transformers:
            train = transformer.transform(train)
            valid = transformer.transform(valid)
            test = transformer.transform(test)
        if reload and isinstance(train, DiskDataset) and isinstance(
                valid, DiskDataset) and isinstance(test, DiskDataset):
            dc.utils.data_utils.save_dataset_to_disk(save_folder, train, valid,
                                                     test, transformers)
        return self.tasks, (train, valid, test), transformers

    def create_dataset(self) -> Dataset:
        """Subclasses must implement this to load the dataset."""
        raise NotImplementedError()
