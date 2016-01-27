"""
Feature calculations.
"""
import types
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdGeometry, rdMolTransforms
from deepchem.utils.save import log

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "BSD 3-clause"


def get_featurizers():
    """Compile a dict mapping strings to featurizer classes."""

    # import all Featurizer subclasses so __subclasses__ will work
    # these have to be local imports to avoid circular imports
    from .basic import MolecularWeight, SimpleDescriptors
    from .coulomb_matrices import CoulombMatrix
    from .dragon import DragonDescriptors
    from .esp import ESP
    from .fingerprints import CircularFingerprint
    from .images import MolImage
    from .scaffolds import ScaffoldGenerator
    from .shape_grid import ShapeGrid

    featurizers = {}
    for klass in Featurizer.__subclasses__():
        assert klass.name is not None, (klass.__name__ +
                                        " 'name' attribute is None.")
        if isinstance(klass.name, list):
            for name in klass.name:
                assert name not in featurizers
                featurizers[name] = klass
        else:
            assert klass.name not in featurizers
            featurizers[klass.name] = klass
    return featurizers


def resolve_featurizer(name):
    """
    Resolve featurizer class from a string.

    Parameters
    ----------
    name : str
        Featurizer name.
    """
    return get_featurizers()[name]

class ComplexFeaturizer(object):
  """"
  Abstract class for calculating features for mol/protein complexes.

  Class Attributes
  ----------------
  name : str or list
      Name (or names) of this featurizer (used for scripting).
  """
  name = None

  def featurize_complexes(self, mol_pdbs, protein_pdbs, log_every_n=1000):
    """
    Calculate features for mol/protein complexes.

    Parameters
    ----------
    mol_pdbs: list
      List of PDBs for molecules. Each PDB should be a list of lines of the
      PDB file.
    protein_pdbs: list
      List of PDBs for proteins. Each PDB should be a list of lines of the
      PDB file.
    """

    features = []
    for i, (mol_pdb, protein_pdb) in enumerate(zip(mol_pdbs, protein_pdbs)):
      if i % log_every_n == 0:
        log("Featurizing %d / %d" % (i, len(mol_pdbs)))
      features.append(self._featurize_complex(mol_pdb, protein_pdb))
    features = np.asarray(features)
    return features

  def _featurize_complex(self, mol_pdb, complex_pdb):
    """
    Calculate features for single mol/protein complex.

    Parameters
    ----------
    mol_pdb: list
      Should be a list of lines of the PDB file.
    complex_pdb: list
      Should be a list of lines of the PDB file.
    """
    raise NotImplementedError('Featurizer is not defined.')

class Featurizer(object):
  """
  Abstract class for calculating a set of features for a molecule.

  Child classes implement the _featurize method for calculating features
  for a single molecule. The feature matrix returned by featurize has a
  shape that is inferred from the shape of the features returned by
  _featurize, and is either indexed by molecules or by molecules and
  conformers depending on the value of the conformers class attribute.

  Class Attributes
  ----------------
  conformers : bool, optional (default False)
      Whether features are calculated for conformers. If True, the first
      two axes of the feature matrix will index molecules and conformers,
      respectively. If False, only molecule-level features are calculated
      and the feature matrix will not have a separate conformer dimension.
      This is a class attribute because some featurizers take 3D
      conformation into account and others do not, and this is not
      typically an instance-level decision.
  name : str or list
      Name (or names) of this featurizer (used for scripting).
  topo_view : bool (default False)
      Whether the calculated features represent a topological view of the
      data.
  """
  conformers = False
  name = None
  topo_view = False

  def featurize(self, mols, parallel=False, client_kwargs=None,
                view_flags=None):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    parallel : bool, optional
        Whether to train subtrainers in parallel using
        IPython.parallel (default False).
    client_kwargs : dict, optional
        Keyword arguments for IPython.parallel Client.
    view_flags : dict, optional
        Flags for IPython.parallel LoadBalancedView.
    """
    if self.conformers and isinstance(mols, types.GeneratorType):
      mols = list(mols)

    if parallel:
      from IPython.parallel import Client

      if client_kwargs is None:
          client_kwargs = {}
      if view_flags is None:
          view_flags = {}
      client = Client(**client_kwargs)
      client.direct_view().use_dill()  # use dill
      view = client.load_balanced_view()
      view.set_flags(**view_flags)
      call = view.map(self._featurize, mols, block=False)
      features = call.get()

      # get output from engines
      call.display_outputs()

    else:
      features = [self._featurize(mol) for mol in mols]

    if self.conformers:
      features = self.conformer_container(mols, features)
    else:
      features = np.asarray(features)
    return features

  def _featurize(self, mol):
    """
    Calculate features for a single molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    raise NotImplementedError('Featurizer is not defined.')

  def __call__(self, mols, parallel=False, client_kwargs=None,
               view_flags=None):
    """
    Calculate features for molecules.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    parallel : bool, optional
        Whether to train subtrainers in parallel using
        IPython.parallel (default False).
    client_kwargs : dict, optional
        Keyword arguments for IPython.parallel Client.
    view_flags : dict, optional
        Flags for IPython.parallel LoadBalancedView.
    """
    return self.featurize(mols, parallel, client_kwargs, view_flags)

  def conformer_container(self, mols, features):
    """
    Put features into a container with an extra dimension for
    conformers. Conformer indices that are not used are masked.

    For example, if mols contains 3 molecules with 1, 2, 5 conformers,
    respectively, then the final container will have 3 entries on its
    first axis and 5 entries on its second axis. The remaining axes
    correspond to feature dimensions.

    Parameters
    ----------
    mols : iterable
        RDKit Mol objects.
    features : list
        Features calculated for molecule conformers. Each element
        corresponds to the features for a molecule and should be an
        ndarray with conformers on the first axis.
    """

    # get the maximum number of conformers
    max_confs = max([mol.GetNumConformers() for mol in mols])
    if not max_confs:
      max_confs = 1

    # construct the new container
    # - first axis = # mols
    # - second axis = max # conformers
    # - remaining axes = determined by feature shape
    features_shape = None
    for i in xrange(len(features)):
      for j in xrange(len(features[i])):
        if features[i][j] is not None:
          features_shape = features[i][0].shape
          break
      if features_shape is not None:
        break
    if features_shape is None:
      raise ValueError('Cannot find any features.')
    shape = (len(mols), max_confs) + features_shape
    x = np.ma.masked_all(shape)

    # fill in the container
    for i, (mol, mol_features) in enumerate(zip(mols, features)):
      n_confs = max(mol.GetNumConformers(), 1)
      try:
        x[i, :n_confs] = mol_features
      except ValueError:  # handle None conformer values
        for j in xrange(n_confs):
          x[i, j] = mol_features[j]
    return x
