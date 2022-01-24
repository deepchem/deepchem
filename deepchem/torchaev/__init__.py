# -*- coding: utf-8 -*-
"""`torchaev`_ is a PyTorch implementation of `ANI`_.It contains class like
:class:`AEVComputer` that can
be pipelined to compute molecular energies from the 3D coordinates of
molecules.  

.. _ANI:
    http://pubs.rsc.org/en/Content/ArticleLanding/2017/SC/C6SC05720A#!divAbstract

"""


from .aev import AEVComputer
from pkg_resources import get_distribution, DistributionNotFound
import warnings

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

__all__ = ['AEVComputer']


try:
    from . import data  # noqa: F401
    __all__.append('data')
except ImportError:
    warnings.warn("Dependency not satisfied, torchani.data will not be available")
