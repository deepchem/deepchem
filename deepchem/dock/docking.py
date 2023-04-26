"""
Docks Molecular Complexes
"""
import logging
import tempfile
from typing import Generator, Optional, Tuple, Union
import numpy as np

from deepchem.utils.typing import RDKitMol
from deepchem.models import Model
from deepchem.feat import ComplexFeaturizer
from deepchem.data import NumpyDataset
from deepchem.dock import PoseGenerator

logger = logging.getLogger(__name__)
POSED_COMPLEX = Tuple[RDKitMol, RDKitMol]


class Docker(object):
    """A generic molecular docking class

    This class provides a docking engine which uses provided models for
    featurization, pose generation, and scoring. Most pieces of docking
    software are command line tools that are invoked from the shell. The
    goal of this class is to provide a python clean API for invoking
    molecular docking programmatically.

    The implementation of this class is lightweight and generic. It's
    expected that the majority of the heavy lifting will be done by pose
    generation and scoring classes that are provided to this class.
    """

    def __init__(self,
                 pose_generator: PoseGenerator,
                 featurizer: Optional[ComplexFeaturizer] = None,
                 scoring_model: Optional[Model] = None):
        """Builds model.

        Parameters
        ----------
        pose_generator: PoseGenerator
            The pose generator to use for this model
        featurizer: ComplexFeaturizer, optional (default None)
            Featurizer associated with `scoring_model`
        scoring_model: Model, optional (default None)
            Should make predictions on molecular complex.
        """
        if ((featurizer is not None and scoring_model is None) or
            (featurizer is None and scoring_model is not None)):
            raise ValueError(
                "featurizer/scoring_model must both be set or must both be None."
            )
        self.base_dir = tempfile.mkdtemp()
        self.pose_generator = pose_generator
        self.featurizer = featurizer
        self.scoring_model = scoring_model

    def dock(
        self,
        molecular_complex: Tuple[str, str],
        centroid: Optional[np.ndarray] = None,
        box_dims: Optional[np.ndarray] = None,
        exhaustiveness: int = 10,
        num_modes: int = 9,
        num_pockets: Optional[int] = None,
        out_dir: Optional[str] = None,
        use_pose_generator_scores: bool = False
    ) -> Union[Generator[POSED_COMPLEX, None, None], Generator[Tuple[
            POSED_COMPLEX, float], None, None]]:
        """Generic docking function.

        This docking function uses this object's featurizer, pose
        generator, and scoring model to make docking predictions. This
        function is written in generic style so

        Parameters
        ----------
        molecular_complex: Tuple[str, str]
            A representation of a molecular complex. This tuple is
            (protein_file, ligand_file).
        centroid: np.ndarray, optional (default None)
            The centroid to dock against. Is computed if not specified.
        box_dims: np.ndarray, optional (default None)
            A numpy array of shape `(3,)` holding the size of the box to dock. If not
            specified is set to size of molecular complex plus 5 angstroms.
        exhaustiveness: int, optional (default 10)
            Tells pose generator how exhaustive it should be with pose
            generation.
        num_modes: int, optional (default 9)
            Tells pose generator how many binding modes it should generate at
            each invocation.
        num_pockets: int, optional (default None)
            If specified, `self.pocket_finder` must be set. Will only
            generate poses for the first `num_pockets` returned by
            `self.pocket_finder`.
        out_dir: str, optional (default None)
            If specified, write generated poses to this directory.
        use_pose_generator_scores: bool, optional (default False)
            If `True`, ask pose generator to generate scores. This cannot be
            `True` if `self.featurizer` and `self.scoring_model` are set
            since those will be used to generate scores in that case.

        Returns
        -------
        Generator[Tuple[`posed_complex`, `score`]] or Generator[`posed_complex`]
            A generator. If `use_pose_generator_scores==True` or
            `self.scoring_model` is set, then will yield tuples
            `(posed_complex, score)`. Else will yield `posed_complex`.
        """
        if self.scoring_model is not None and use_pose_generator_scores:
            raise ValueError(
                "Cannot set use_pose_generator_scores=True "
                "when self.scoring_model is set (since both generator scores for complexes)."
            )

        outputs = self.pose_generator.generate_poses(
            molecular_complex,
            centroid=centroid,
            box_dims=box_dims,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            num_pockets=num_pockets,
            out_dir=out_dir,
            generate_scores=use_pose_generator_scores)
        if use_pose_generator_scores:
            complexes, scores = outputs
        else:
            complexes = outputs

        # We know use_pose_generator_scores == False in this case
        if self.scoring_model is not None:
            for posed_complex in complexes:
                # check whether self.featurizer is instance of ComplexFeaturizer or not
                assert isinstance(self.featurizer, ComplexFeaturizer)
                # TODO: How to handle the failure here?
                features = self.featurizer.featurize([molecular_complex])
                dataset = NumpyDataset(X=features)
                score = self.scoring_model.predict(dataset)
                yield (posed_complex, score)
        elif use_pose_generator_scores:
            for posed_complex, score in zip(complexes, scores):
                yield (posed_complex, score)
        else:
            for posed_complex in complexes:
                yield posed_complex
