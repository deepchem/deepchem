import numpy as np
from deepchem.feat import MolecularFeaturizer
from deepchem.utils.sascore import SAScorer
from deepchem.utils.typing import RDKitMol

class SAScoreFeaturizer(MolecularFeaturizer):
    """
    Synthetic Accessibility Score (SAScore) Featurizer.

    This featurizer computes the SAScore for a molecule, which is a 
    value between 1 (easy to synthesize) and 10 (hard to synthesize).
    It uses the rule-based fragment contributions and complexity 
    penalties defined by Ertl and Schuffenhauer.

    Note
    ----
    This featurizer requires the 'fpscores.pkl.gz' file, which will be 
    automatically downloaded to the DeepChem data directory upon first use.
    """

    def __init__(self):
        """Initialize the SAScore Featurizer."""
        self.scorer = None

    def _featurize(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Calculate SAScore for a single molecule.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit molecule object.

        Returns
        -------
        np.ndarray
            A 1D numpy array containing the SAScore. 
            Returns [np.nan] if featurization fails.
        """
        # Lazy initialization of the scorer to ensure data 
        # is only downloaded/loaded when actually needed.
        if self.scorer is None:
            self.scorer = SAScorer()

        try:
            score = self.scorer.calculate_score(datapoint)
            if score is None:
                return np.array([np.nan])
            return np.array([score], dtype=np.float32)
        except Exception as e:
            # Log the error but don't crash the entire pipeline
            import logging
            logging.getLogger(__name__).warning(f"Featurization failed: {e}")
            return np.array([np.nan])