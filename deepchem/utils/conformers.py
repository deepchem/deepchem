"""
Conformer generation.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np


class ConformerGenerator(object):
  """
  Generate molecule conformers.

  Notes
  -----
  Procedure
  1. Generate a pool of conformers.
  2. Minimize conformers.
  3. Prune conformers using an RMSD threshold.

  Note that pruning is done _after_ minimization, which differs from the
  protocol described in the references [1]_ [2]_.

  References
  ----------
  .. [1] http://rdkit.org/docs/GettingStartedInPython.html#working-with-3d-molecules
  .. [2] http://pubs.acs.org/doi/full/10.1021/ci2004658

  Parameters
  ----------
  max_conformers : int, optional (default 1)
      Maximum number of conformers to generate (after pruning).
  rmsd_threshold : float, optional (default 0.5)
      RMSD threshold for pruning conformers. If None or negative, no
      pruning is performed.
  force_field : str, optional (default 'uff')
      Force field to use for conformer energy calculation and
      minimization. Options are 'uff', 'mmff94', and 'mmff94s'.
  pool_multiplier : int, optional (default 10)
      Factor to multiply by max_conformers to generate the initial
      conformer pool. Since conformers are pruned after energy
      minimization, increasing the size of the pool increases the chance
      of identifying max_conformers unique conformers.
  """

  def __init__(self,
               max_conformers=1,
               rmsd_threshold=0.5,
               force_field='uff',
               pool_multiplier=10):
    self.max_conformers = max_conformers
    if rmsd_threshold is None or rmsd_threshold < 0:
      rmsd_threshold = -1.
    self.rmsd_threshold = rmsd_threshold
    self.force_field = force_field
    self.pool_multiplier = pool_multiplier

  def __call__(self, mol):
    """
    Generate conformers for a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    return self.generate_conformers(mol)

  def generate_conformers(self, mol):
    """
    Generate conformers for a molecule.

    This function returns a copy of the original molecule with embedded
    conformers.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """

    # initial embedding
    mol = self.embed_molecule(mol)
    if not mol.GetNumConformers():
      msg = 'No conformers generated for molecule'
      if mol.HasProp('_Name'):
        name = mol.GetProp('_Name')
        msg += ' "{}".'.format(name)
      else:
        msg += '.'
      raise RuntimeError(msg)

    # minimization and pruning
    self.minimize_conformers(mol)
    mol = self.prune_conformers(mol)

    return mol

  def embed_molecule(self, mol):
    """
    Generate conformers, possibly with pruning.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(mol)  # add hydrogens
    n_confs = self.max_conformers * self.pool_multiplier
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, pruneRmsThresh=-1.)
    return mol

  def get_molecule_force_field(self, mol, conf_id=None, **kwargs):
    """
    Get a force field for a molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    from rdkit.Chem import AllChem
    if self.force_field == 'uff':
      ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id, **kwargs)
    elif self.force_field.startswith('mmff'):
      AllChem.MMFFSanitizeMolecule(mol)
      mmff_props = AllChem.MMFFGetMoleculeProperties(
          mol, mmffVariant=self.force_field)
      ff = AllChem.MMFFGetMoleculeForceField(
          mol, mmff_props, confId=conf_id, **kwargs)
    else:
      raise ValueError("Invalid force_field " +
                       "'{}'.".format(self.force_field))
    return ff

  def minimize_conformers(self, mol):
    """
    Minimize molecule conformers.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    for conf in mol.GetConformers():
      ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
      ff.Minimize()

  def get_conformer_energies(self, mol):
    """
    Calculate conformer energies.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
      ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
      energy = ff.CalcEnergy()
      energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies

  def prune_conformers(self, mol):
    """
    Prune conformers from a molecule using an RMSD threshold, starting
    with the lowest energy conformer.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.

    Returns
    -------
    A new RDKit Mol containing the chosen conformers, sorted by
    increasing energy.
    """
    if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
      return mol
    energies = self.get_conformer_energies(mol)
    rmsd = self.get_conformer_rmsd(mol)

    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    discard = []
    for i in sort:
      # always keep lowest-energy conformer
      if len(keep) == 0:
        keep.append(i)
        continue

      # discard conformers after max_conformers is reached
      if len(keep) >= self.max_conformers:
        discard.append(i)
        continue

      # get RMSD to selected conformers
      this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

      # discard conformers within the RMSD threshold
      if np.all(this_rmsd >= self.rmsd_threshold):
        keep.append(i)
      else:
        discard.append(i)

    # create a new molecule to hold the chosen conformers
    # this ensures proper conformer IDs and energy-based ordering
    from rdkit import Chem
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
      conf = mol.GetConformer(conf_ids[i])
      new.AddConformer(conf, assignId=True)
    return new

  @staticmethod
  def get_conformer_rmsd(mol):
    """
    Calculate conformer-conformer RMSD.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit.Chem import AllChem
    rmsd = np.zeros(
        (mol.GetNumConformers(), mol.GetNumConformers()), dtype=float)
    for i, ref_conf in enumerate(mol.GetConformers()):
      for j, fit_conf in enumerate(mol.GetConformers()):
        if i >= j:
          continue
        rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                        fit_conf.GetId())
        rmsd[j, i] = rmsd[i, j]
    return rmsd
