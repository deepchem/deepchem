"""
Basic molecular features.
"""

from typing import List, Sequence, Tuple, Union, Callable, Dict
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

  The neural network architecture requires that the features are appropriately scaled to prevent
  features with large ranges from dominating smaller ranged features, as well as preventing
  issues where features in the training set are not drawn from the same sample distribution as
  features in the testing set. To prevent these issues, a large sample of molecules is used to fit
  cumulative density functions (CDFs) to all features.

  CDFs were used as opposed to simpler scaling algorithms mainly because CDFs have the useful
  property that 'each value has the same meaning: the percentage of the population observed below
  the raw feature value.'

  Warning: Currently, the normalizing cdf parameters are not available for BCUT2D descriptors.
  (BCUT2D_MWHI, BCUT2D_MWLOW, BCUT2D_CHGHI, BCUT2D_CHGLO, BCUT2D_LOGPHI, BCUT2D_LOGPLOW, BCUT2D_MRHI, BCUT2D_MRLOW)

  Attributes
  ----------
  descriptors: List[str]
    List of RDKit descriptor names used in this class.

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
  (208,)
  """

  def __init__(self,
               use_fragment: bool = True,
               ipc_avg: bool = True,
               is_normalized: bool = False,
               use_bcut2d: bool = True):
    """Initialize this featurizer.

    Parameters
    ----------
    use_fragment: bool, optional (default True)
      If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
    ipc_avg: bool, optional (default True)
      If True, the IPC descriptor calculates with avg=True option.
      Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
    is_normalized: bool, optional (default False)
      If True, the return value contains normalized features.
    use_bcut2d: bool, optional (default True)
      If True, the return value includes the descriptors like 'BCUT2D_XXX'.
    """
    self.use_fragment: bool = use_fragment
    self.use_bcut2d: bool = use_bcut2d
    self.is_normalized: bool = is_normalized
    self.ipc_avg: bool = ipc_avg
    self.descriptors: List[str] = []
    self.descList: List[Tuple[str, Callable]] = []
    self.normalized_desc: Dict[str, Callable] = {}

    # initialize
    if len(self.descList) == 0:
      try:
        from rdkit.Chem import Descriptors

        desc_name: str  # descriptor name
        function: Callable  # descriptor function
        for desc_name, function in Descriptors.descList:
          if self.use_fragment is False and desc_name.startswith('fr_'):
            continue
          if self.use_bcut2d is False and desc_name.startswith('BCUT2D_'):
            continue
          self.descriptors.append(desc_name)
          self.descList.append((desc_name, function))
      except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")

    # check initialization
    assert len(self.descriptors) == len(self.descList)

    # creates normalized functions dictionary if normalized features are required
    if is_normalized:
      self._make_normalised_func_dict()

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
    if 'mol' in kwargs:
      datapoint = kwargs.get("mol")
      raise DeprecationWarning(
          'Mol is being phased out as a parameter, please pass "datapoint" instead.'
      )

    features: List[Union[int, float]] = []
    if not self.is_normalized:
      for desc_name, function in self.descList:
        if desc_name == 'Ipc' and self.ipc_avg:
          feature: Union[int, float] = function(datapoint, avg=True)
        else:
          feature = function(datapoint)
        features.append(feature)
    else:
      for desc_name, function in self.descList:

        if desc_name == 'Ipc' and self.ipc_avg:
          feature = function(datapoint, avg=True)
        else:
          feature = function(datapoint)

        try:
          feature = self.normalized_desc[desc_name](
              feature)  # get cdf(feature) for that descriptor
        except KeyError:
          logger.warning("No normalization for %s. Feature removed!", desc_name)
          self.descriptors.remove(
              desc_name
          )  # removes descriptors from the list, which cannot be normalized
          continue

        features.append(feature)
    return np.asarray(features)

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
    # get sequence of descriptor names and normalization parameters from DescriptorsNormalizationParameters class
    parameters: Sequence[Union[str, Sequence[Union[str, Sequence[float],
                                                   float]]]]
    parameters = DNP.desc_norm_params.items()

    for desc_name, (distribution_name, params, minV, maxV, avg,
                    std) in parameters:
      arg: Sequence[float] = params[:-2]
      loc: float = params[-2]
      scale: float = params[-1]

      # get required distribution_ from `scipy.stats` module.
      cont_distribution: rv_continuous = getattr(st, distribution_name)

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

      self.normalized_desc[desc_name] = norm_cdf
