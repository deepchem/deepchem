"""
Feature calculations.
"""
import inspect
import logging
import numpy as np
from joblib import Parallel, delayed
from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast
from collections.abc import Sequence as _Sequence  # runtime Sequence check

from deepchem.utils import get_print_threshold
from deepchem.utils.typing import PymatgenStructure

logger = logging.getLogger(__name__)


class Featurizer(object):
    """Abstract class for calculating a set of features for a datapoint.

    This class is abstract and cannot be invoked directly. You'll
    likely only interact with this class if you're a developer. In
    that case, you might want to make a child class which
    implements the `_featurize` method for calculating features for
    a single datapoints if you'd like to make a featurizer for a
    new datatype.
    """

    def featurize(self,
                  datapoints: Any,
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Calculate features for datapoints.

        Parameters
        ----------
        datapoints: Iterable[Any]
            A sequence of objects that you'd like to featurize. Subclassses of
            `Featurizer` should instantiate the `_featurize` method that featurizes
            objects in the sequence.
        log_every_n: int, default 1000
            Logs featurization progress every `log_every_n` steps.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        # Normalize datapoints to a list. We accept Any here to maximize
        # compatibility with subclasses that may use different signatures.
        try:
            datapoints = list(datapoints)
        except TypeError:
            # datapoints may be a single element (not iterable); wrap it.
            datapoints = [datapoints]

        # Helper that executes a single datapoint featurization safely.
        def _safe_featurize(i: int, point: Any) -> Any:
            """Helper to featurize safely with error handling.

            Builds kwargs_per_datapoint: scalar kwargs are passed through;
            sequence-like kwargs (lists/tuples/ndarrays) are indexed per i.
            Strings and bytes are treated as scalars (not sequences to index).
            """
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)
            try:
                # Build kwargs for this datapoint.
                kwargs_per_datapoint: Dict[str, Any] = {}
                for key, val in kwargs.items():
                    # Treat string/bytes as scalar even though they are sequences.
                    if isinstance(val, _Sequence) and not isinstance(
                            val, (str, bytes)):
                        # Sequence-like: try indexing, but fall back to whole value on error.
                        try:
                            kwargs_per_datapoint[key] = val[i]
                        except Exception:
                            # If indexing fails (e.g., wrong length), pass original value.
                            kwargs_per_datapoint[key] = val
                    else:
                        # Scalar: pass through.
                        kwargs_per_datapoint[key] = val

                return self._featurize(point, **kwargs_per_datapoint)
            except Exception as e:
                # Log the exception and return an empty array marker.
                logger.warning(
                    "Failed to featurize datapoint {}. Appending empty array. Error: {}"
                    .format(i, e))
                return np.array([])

        # Run in parallel if requested
        if n_jobs != 1:
            logger.info(
                "Featurizing {} datapoints using {} parallel jobs".format(
                    len(datapoints), n_jobs))
            # Use joblib.Parallel to run the safe wrapper on each datapoint.
            features = Parallel(n_jobs=n_jobs)(
                delayed(_safe_featurize)(i, point)
                for i, point in enumerate(datapoints))
        else:
            features = [
                _safe_featurize(i, point) for i, point in enumerate(datapoints)
            ]

        # Try to assemble into a homogeneous numpy array; if shapes differ,
        # fall back to an object-dtype array to preserve results.
        try:
            return np.asarray(features)
        except ValueError as e:
            logger.warning(
                "Exception while creating numpy array from features: %s", e)
            return np.asarray(features, dtype=object)

    def __call__(self, datapoints: Iterable[Any], **kwargs):
        """Calculate features for datapoints.

        `**kwargs` will get passed directly to `Featurizer.featurize`

        Parameters
        ----------
        datapoints: Iterable[Any]
            Any blob of data you like. Subclasss should instantiate this.
        """
        return self.featurize(datapoints, **kwargs)

    def _featurize(self, datapoint: Any, **kwargs):
        """Calculate features for a single datapoint.

        Parameters
        ----------
        datapoint: Any
            Any blob of data you like. Subclass should instantiate this.
        """
        raise NotImplementedError('Featurizer is not defined.')

    def __repr__(self) -> str:
        """Convert self to repr representation.

        Returns
        -------
        str
            The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> dc.feat.CircularFingerprint(size=1024, radius=4)
        CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True, features=False, sparse=False, smiles=False, is_counts_based=False]
        >>> dc.feat.CGCNNFeaturizer()
        CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != 'self']
        args_info = ''
        for arg_name in args_names:
            value = self.__dict__[arg_name]
            # for str
            if isinstance(value, str):
                value = "'" + value + "'"
            # for list
            if isinstance(value, list):
                threshold = get_print_threshold()
                value = np.array2string(np.array(value), threshold=threshold)
            args_info += arg_name + '=' + str(value) + ', '
        return self.__class__.__name__ + '[' + args_info[:-2] + ']'

    def __str__(self) -> str:
        """Convert self to str representation.

        Returns
        -------
        str
            The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
        'CircularFingerprint_radius_4_size_1024'
        >>> str(dc.feat.CGCNNFeaturizer())
        'CGCNNFeaturizer'
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != 'self']
        args_num = len(args_names)
        args_default_values = [None for _ in range(args_num)]
        if args_spec.defaults is not None:
            defaults = list(args_spec.defaults)
            args_default_values[-len(defaults):] = defaults

        override_args_info = ''
        for arg_name, default in zip(args_names, args_default_values):
            if arg_name in self.__dict__:
                arg_value = self.__dict__[arg_name]
                # validation
                # skip list
                if isinstance(arg_value, list):
                    continue
                if isinstance(arg_value, str):
                    # skip path string
                    if "\\/." in arg_value or "/" in arg_value or '.' in arg_value:
                        continue
                # main logic
                if default != arg_value:
                    override_args_info += '_' + arg_name + '_' + str(arg_value)
        return self.__class__.__name__ + override_args_info


class ComplexFeaturizer(Featurizer):
    """"
    Abstract class for calculating features for mol/protein complexes.
    """

    def featurize(self,
                  datapoints: Optional[Iterable[Tuple[str, str]]] = None,
                  log_every_n: int = 100,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """
        Calculate features for mol/protein complexes.
        Parameters
        ----------
        datapoints: Iterable[Tuple[str, str]]
            List of filenames (PDB, SDF, etc.) for ligand molecules and proteins.
            Each element should be a tuple of the form (ligand_filename,
            protein_filename).
        log_every_n: int, default 100
            Logging messages reported every `log_every_n` samples.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        features: np.ndarray
            Array of features
        """

        if 'complexes' in kwargs:
            datapoints = kwargs.get("complexes")
            raise DeprecationWarning(
                'Complexes is being phased out as a parameter, please pass "datapoints" instead.'
            )
        if not isinstance(datapoints, Iterable):
            datapoints = [cast(Tuple[str, str], datapoints)]

        # Convert to list for processing
        datapoints = list(datapoints)

        # Delegate to base class for parallel processing
        features = super().featurize(datapoints,
                                     log_every_n=log_every_n,
                                     n_jobs=n_jobs,
                                     **kwargs)

        # Handle complex-specific post-processing for failed featurizations
        # Find successful featurizations and create dummy arrays for failures
        successes = []
        failures = []
        for idx, feature in enumerate(features):
            if feature.size > 0:
                successes.append(idx)
            else:
                failures.append(idx)

        if failures and successes:
            # Find a successful featurization to use as template
            try:
                i = np.argmax([f.shape[0] for f in features[successes]])
                success_idx = successes[i]
                dtype = features[success_idx].dtype
                shape = features[success_idx].shape
                dummy_array = np.zeros(shape, dtype=dtype)
            except (AttributeError, IndexError):
                dummy_array = np.zeros(1)

            # Replace failed featurizations with appropriate array
            for idx in failures:
                features[idx] = dummy_array

        return np.asarray(features, dtype=object)

    def _featurize(self, datapoint: Optional[Tuple[str, str]] = None, **kwargs):
        """
        Calculate features for single mol/protein complex.

        Parameters
        ----------
        complex: Tuple[str, str]
            Filenames for molecule and protein.
        """
        raise NotImplementedError('Featurizer is not defined.')


class MolecularFeaturizer(Featurizer):
    """Abstract class for calculating a set of features for a
molecule.

    The defining feature of a `MolecularFeaturizer` is that it
    uses SMILES strings and RDKit molecule objects to represent
    small molecules. All other featurizers which are subclasses of
    this class should plan to process input which comes as smiles
    strings or RDKit molecules.

    Child classes need to implement the _featurize method for
    calculating features for a single molecule.

    Note
    ----
    The subclasses of this class require RDKit to be installed.
    """

    def __init__(self, use_original_atoms_order: bool = False):
        """
        Parameters
        ----------
        use_original_atoms_order: bool, default False
            Whether to use original atom ordering or canonical ordering (default)
        """
        self.use_original_atoms_order = use_original_atoms_order

    def featurize(self,
                  datapoints: Any,
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Calculate features for molecules.

        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
            RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
            strings.
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolfiles
            from rdkit.Chem import rdmolops
            from rdkit.Chem.rdchem import Mol
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if 'molecules' in kwargs:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning(
                'Molecules is being phased out as a parameter, please pass "datapoints" instead.'
            )

        # Special case handling of single molecule
        if isinstance(datapoints, str) or isinstance(datapoints, Mol):
            datapoints = [datapoints]
        else:
            # Convert iterables to list
            try:
                datapoints = list(datapoints)
            except TypeError:
                datapoints = [datapoints]

        # Build a processed list where SMILES strings are converted to RDKit Mol
        # objects and invalid SMILES produce None placeholders. We annotate
        # the list for mypy/type-checkers.
        processed: list[Optional[Mol]] = []
        for mol in datapoints:
            if isinstance(mol, str):
                # Parse SMILES into RDKit Mol; None if parsing fails.
                parsed = Chem.MolFromSmiles(mol)
                if parsed is None:
                    processed.append(None)
                    continue
                # If canonical ordering is desired, renumber atoms.
                if not getattr(self, 'use_original_atoms_order', False):
                    new_order = rdmolfiles.CanonicalRankAtoms(parsed)
                    parsed = rdmolops.RenumberAtoms(parsed, new_order)
                processed.append(parsed)
            else:
                # Already an RDKit Mol (or None), append as-is.
                processed.append(mol)

        # Delegate to base class for the core featurization loop (including
        # per-datapoint kwargs handling and parallelization). We forward
        # n_jobs explicitly so that callers can request parallel execution.
        return super().featurize(processed,
                                 log_every_n=log_every_n,
                                 n_jobs=n_jobs,
                                 **kwargs)


class MaterialStructureFeaturizer(Featurizer):
    """
    Abstract class for calculating a set of features for an
    inorganic crystal structure.

    The defining feature of a `MaterialStructureFeaturizer` is that it
    operates on 3D crystal structures with periodic boundary conditions.
    Inorganic crystal structures are represented by Pymatgen structure
    objects. Featurizers for inorganic crystal structures that are subclasses of
    this class should plan to process input which comes as pymatgen
    structure objects.

    This class is abstract and cannot be invoked directly. You'll
    likely only interact with this class if you're a developer. Child
    classes need to implement the _featurize method for calculating
    features for a single crystal structure.

    Note
    ----
    Some subclasses of this class will require pymatgen and matminer to be
    installed.
    """

    def featurize(self,
                  datapoints: Optional[Iterable[Union[Dict[
                      str, Any], PymatgenStructure]]] = None,
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Calculate features for crystal structures.

        Parameters
        ----------
        datapoints: Iterable[Union[Dict, pymatgen.core.Structure]]
            Iterable sequence of pymatgen structure dictionaries
            or pymatgen.core.Structure. Please confirm the dictionary representations
            of pymatgen.core.Structure from https://pymatgen.org/pymatgen.core.structure.html.
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` steps.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of
            `datapoints`.
        """
        try:
            from pymatgen.core import Structure
        except ModuleNotFoundError:
            raise ImportError("This class requires pymatgen to be installed.")

        if 'structures' in kwargs:
            datapoints = kwargs.get("structures")
            raise DeprecationWarning(
                'Structures is being phased out as a parameter, please pass "datapoints" instead.'
            )

        if not isinstance(datapoints, Iterable):
            datapoints = [
                cast(Union[Dict[str, Any], PymatgenStructure], datapoints)
            ]

        # Convert to list and preprocess structures
        datapoints = list(datapoints)
        processed_structures = []
        for structure in datapoints:
            if isinstance(structure, Dict):
                structure = Structure.from_dict(structure)
            processed_structures.append(structure)

        # Delegate to base class for parallel processing
        return super().featurize(processed_structures,
                                 log_every_n=log_every_n,
                                 n_jobs=n_jobs,
                                 **kwargs)


class MaterialCompositionFeaturizer(Featurizer):
    """
    Abstract class for calculating a set of features for an
    inorganic crystal composition.

    The defining feature of a `MaterialCompositionFeaturizer` is that it
    operates on 3D crystal chemical compositions.
    Inorganic crystal compositions are represented by Pymatgen composition
    objects. Featurizers for inorganic crystal compositions that are
    subclasses of this class should plan to process input which comes as
    Pymatgen composition objects.

    This class is abstract and cannot be invoked directly. You'll
    likely only interact with this class if you're a developer. Child
    classes need to implement the _featurize method for calculating
    features for a single crystal composition.

    Note
    ----
    Some subclasses of this class will require pymatgen and matminer to be
    installed.
    """

    def featurize(self,
                  datapoints: Optional[Iterable[str]] = None,
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Calculate features for crystal compositions.

        Parameters
        ----------
        datapoints: Iterable[str]
            Iterable sequence of composition strings, e.g. "MoS2".
        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` steps.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of
            `compositions`.
        """
        try:
            from pymatgen.core import Composition
        except ModuleNotFoundError:
            raise ImportError("This class requires pymatgen to be installed.")

        if 'compositions' in kwargs and datapoints is None:
            datapoints = kwargs.get("compositions")
            raise DeprecationWarning(
                'Compositions is being phased out as a parameter, please pass "datapoints" instead.'
            )

        if not isinstance(datapoints, Iterable):
            datapoints = [cast(str, datapoints)]

        # Convert to list and preprocess compositions
        datapoints = list(datapoints)
        processed_compositions = []
        for composition in datapoints:
            c = Composition(composition)
            processed_compositions.append(c)

        # Delegate to base class for parallel processing
        return super().featurize(processed_compositions,
                                 log_every_n=log_every_n,
                                 n_jobs=n_jobs,
                                 **kwargs)


class PolymerFeaturizer(Featurizer):
    """Abstract class for calculating features for polymer materials.

    The `PolymerFeaturzer` is responsibe for conversion of different
    polymer representations to features. The child classes can
    following representations for feature conversions.

    i)  Weighted Directed Graph Representation (Monomer SMILES + Fragments + Weight Distrbution)
    ii) BigSMILES String Representation

    This polymer base class is useful considering it handles batches and validates the individual data points
    before passing it for featurization. Currently it only validates the string type for above representations.

    Child classes need to implement the _featurize method for
    calculating features for a polymer.

    Example
    -------
    >>> from deepchem.feat import PolymerFeaturizer
    >>> class MyPolymerFeaturizer(PolymerFeaturizer):
    ...     def _featurize(self, datapoint):
    ...         # Implement your featurization logic here
    ...         pass
    >>> featurizer = MyPolymerFeaturizer()
    >>> features = featurizer.featurize(['CCC'])

    Note
    ----
    The subclasses of this class require RDKit to be installed.
    """

    def featurize(self,
                  datapoints: Iterable[Any],
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Calculate features for polymers.

        Parameters
        ----------
        datapoints: BigSMILES Strings /  Iterable of BigSMILES Strings
        Weighted Directed Graph Objects / Iterable of Weighted Directed Graph Objects

        log_every_n: int, default 1000
            Logging messages reported every `log_every_n` samples.
        n_jobs: int, default 1
            Number of parallel jobs to run.
            -1 means "use all available cores".

        Returns
        -------
        features: np.ndarray
            A numpy array containing a featurized representation of `datapoints`.
        """
        # converting single data point to array
        if isinstance(datapoints, str):
            datapoints = [datapoints]
        else:
            try:
                datapoints = list(datapoints)
            except TypeError:
                datapoints = [datapoints]

        # Validate and preprocess datapoints
        processed_datapoints = []
        for mol in datapoints:
            if isinstance(mol, str):
                processed_datapoints.append(mol)
            else:
                raise ValueError(
                    "The input data point has to be of string representation of "
                    "BigSMILES or Weight Distributed String Representation not {}"
                    .format(type(mol)))

        # Delegate to base class for parallel processing
        return super().featurize(processed_datapoints,
                                 log_every_n=log_every_n,
                                 n_jobs=n_jobs,
                                 **kwargs)


class UserDefinedFeaturizer(Featurizer):
    """Directs usage of user-computed featurizations."""

    def __init__(self, feature_fields):
        """Creates user-defined-featurizer."""
        self.feature_fields = feature_fields


class DummyFeaturizer(Featurizer):
    """Class that implements a no-op featurization.
    This is useful when the raw dataset has to be used without featurizing the
    examples. The Molnet loader requires a featurizer input and such datasets
    can be used in their original form by passing the raw featurizer.

    Examples
    --------
    >>> import deepchem as dc
    >>> smi_map = [["N#C[S-].O=C(CBr)c1ccc(C(F)(F)F)cc1>CCO.[K+]", "N#CSCC(=O)c1ccc(C(F)(F)F)cc1"], ["C1COCCN1.FCC(Br)c1cccc(Br)n1>CCN(C(C)C)C(C)C.CN(C)C=O.O", "FCC(c1cccc(Br)n1)N1CCOCC1"]]
    >>> Featurizer = dc.feat.DummyFeaturizer()
    >>> smi_feat = Featurizer.featurize(smi_map)
    >>> smi_feat
    array([['N#C[S-].O=C(CBr)c1ccc(C(F)(F)F)cc1>CCO.[K+]',
            'N#CSCC(=O)c1ccc(C(F)(F)F)cc1'],
           ['C1COCCN1.FCC(Br)c1cccc(Br)n1>CCN(C(C)C)C(C)C.CN(C)C=O.O',
            'FCC(c1cccc(Br)n1)N1CCOCC1']], dtype='<U55')
    """

    def featurize(self,
                  datapoints: Iterable[Any],
                  log_every_n: int = 1000,
                  n_jobs: int = 1,
                  **kwargs) -> np.ndarray:
        """Passes through dataset, and returns the datapoint.

        Parameters
        ----
        datapoints: Iterable[Any]
            A sequence of objects that you'd like to featurize.

        Returns
        ----
        datapoints: np.ndarray
            A numpy array containing a featurized representation of
            the datapoints.
        """
        return np.asarray(datapoints)
