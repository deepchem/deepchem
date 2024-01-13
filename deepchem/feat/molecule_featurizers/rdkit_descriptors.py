"""
Basic molecular features.
"""
from typing import List, Sequence, Union, Callable, Dict
import numpy as np
import scipy.stats as st
from scipy.stats import rv_continuous
import logging

from deepchem.utils.typing import RDKitMol
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.rdkit_utils import DescriptorsNormalizationParameters as DNP

logger = logging.getLogger(__name__)


class RDKitDescriptors(MolecularFeaturizer):
    """
    RDKit descriptors.

    This class computes a list of chemical descriptors like
    molecular weight, number of valence electrons, maximum and
    minimum partial charge, etc using RDKit.

    This class can also compute normalized descriptors, if required.
    (The implementation for normalization is based on `RDKit2DNormalized()` method
    in 'descriptastorus' library.)

    When the `is_normalized` option is set as True, descriptor values are normalized across the sample
    by fitting a cumulative density function. CDFs were used as opposed to simpler scaling algorithms
    mainly because CDFs have the useful property that 'each value has the same meaning: the percentage
    of the population observed below the raw feature value.'

    Warning: Currently, the normalizing cdf parameters are not available for BCUT2D descriptors.
    (BCUT2D_MWHI, BCUT2D_MWLOW, BCUT2D_CHGHI, BCUT2D_CHGLO, BCUT2D_LOGPHI, BCUT2D_LOGPLOW, BCUT2D_MRHI, BCUT2D_MRLOW)

    Note
    ----
    This class requires RDKit to be installed.

    Examples
    --------
    >>> import deepchem as dc
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = dc.feat.RDKitDescriptors()
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (210,)

    """

    def __init__(self,
                 descriptors: List[str] = [],
                 is_normalized: bool = False,
                 use_fragment: bool = True,
                 ipc_avg: bool = True,
                 use_bcut2d: bool = True,
                 labels_only: bool = False):
        """Initialize this featurizer.

        Parameters
        ----------
        descriptors: List[str] (default None)
            List of RDKit descriptors to compute properties. When None, computes values
        for descriptors which are chosen based on options set in other arguments.
        use_fragment: bool, optional (default True)
            If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
        ipc_avg: bool, optional (default True)
            If True, the IPC descriptor calculates with avg=True option.
            Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
        is_normalized: bool, optional (default False)
            If True, the return value contains normalized features.
        use_bcut2d: bool, optional (default True)
            If True, the return value includes the descriptors like 'BCUT2D_XXX'.
        labels_only: bool, optional (default False)
            Returns only the presence or absence of a group.

        Notes
        -----
        * If both `labels_only` and `is_normalized` are True, then `is_normalized` takes
            precendence and `labels_only` will not be applied.

        """
        try:
            from rdkit.Chem import Descriptors
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        self.use_fragment: bool = use_fragment
        self.use_bcut2d: bool = use_bcut2d
        self.is_normalized: bool = is_normalized
        self.ipc_avg: bool = ipc_avg
        self.labels_only = labels_only
        self.reqd_properties = {}
        self.normalized_desc: Dict[str, Callable] = {}

        all_descriptors = {name: func for name, func in Descriptors.descList}

        if not descriptors:
            # user has not specified a descriptor list
            for desc_name, function in all_descriptors.items():
                if self.use_fragment is False and desc_name.startswith('fr_'):
                    continue
                if self.use_bcut2d is False and desc_name.startswith('BCUT2D_'):
                    continue
                self.reqd_properties[desc_name] = function
        else:
            for desc_name in descriptors:
                if desc_name in all_descriptors:
                    self.reqd_properties[desc_name] = all_descriptors[desc_name]
                else:
                    logging.error("Unable to find specified property %s" %
                                  desc_name)

        # creates normalized functions dictionary if normalized features are required
        if is_normalized:
            self.normalized_desc = self._make_normalised_func_dict()
            desc_names = list(self.reqd_properties.keys())
            for desc_name in desc_names:
                if desc_name not in self.normalized_desc:
                    logger.warning("No normalization for %s. Feature removed!",
                                   desc_name)
                    self.reqd_properties.pop(desc_name)

        self.reqd_properties = dict(sorted(self.reqd_properties.items()))

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit Mol object

        Returns
        -------
        np.ndarray
            1D array of RDKit descriptors for `mol`.
            The length is `len(self.descriptors)`.

        """
        features = []
        for desc_name, function in self.reqd_properties.items():
            if desc_name == 'Ipc' and self.ipc_avg:
                feature = function(datapoint, avg=True)
            else:
                feature = function(datapoint)

            if self.is_normalized:
                # get cdf(feature) for that descriptor
                feature = self.normalized_desc[desc_name](feature)

            features.append(feature)

        np_features = np.asarray(features)
        if self.labels_only:
            np.putmask(np_features, np_features != 0, 1)
        return np_features

    def _make_normalised_func_dict(self):
        """
        Helper function to create dictionary of RDkit descriptors and
        associated cumulative density functions (CDFs) to generate normalized features.

        -------------------------------------------------------------------------------
        -------------------------------------------------------------------------------
        Copyright (c) 2018-2021, Novartis Institutes for BioMedical Research Inc.
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        * Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
            copyright notice, this list of conditions and the following
            disclaimer in the documentation and/or other materials provided
            with the distribution.
        * Neither the name of Novartis Institutes for BioMedical Research Inc.
            nor the names of its contributors may be used to endorse or promote
            products derived from this software without specific prior written
            permission.

        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        -------------------------------------------------------------------------------
        -------------------------------------------------------------------------------

        """
        normalized_desc = {}
        # get sequence of descriptor names and normalization parameters from DescriptorsNormalizationParameters class
        parameters = DNP.desc_norm_params.items()

        for desc_name, (distribution_name, params, minV, maxV, avg,
                        std) in parameters:
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # get required distribution_ from `scipy.stats` module.
            cont_distribution = getattr(st, distribution_name)

            # cdf => cumulative density functions
            # make the cdf with the parameters.
            def norm_cdf(v: Union[int, float],
                         distribution_: rv_continuous = cont_distribution,
                         arg: Sequence[float] = arg,
                         loc: float = loc,
                         scale: float = scale,
                         minV: float = minV,
                         maxV: float = maxV) -> np.ndarray:
                v = distribution_.cdf(np.clip(v, minV, maxV),
                                      loc=loc,
                                      scale=scale,
                                      *arg)
                return np.clip(v, 0., 1.)

            normalized_desc[desc_name] = norm_cdf
        return normalized_desc
