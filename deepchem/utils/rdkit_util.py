import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_xyz_from_mol(mol):
  """
  returns an m x 3 np array of 3d coords
  of given rdkit molecule
  """
  xyz = np.zeros((mol.GetNumAtoms(), 3))
  conf = mol.GetConformer()
  for i in range(conf.GetNumAtoms()):
    position = conf.GetAtomPosition(i)
    xyz[i, 0] = position.x
    xyz[i, 1] = position.y
    xyz[i, 2] = position.z
  return (xyz)


# TODO(LESWING) move to util
def load_molecule(molecule_file, add_hydrogens=True,
                  calc_charges=False):
  """Converts molecule file to (xyz-coords, obmol object)

  Given molecule_file, returns a tuple of xyz coords of molecule
  and an rdkit object representing that molecule
  """
  if ".mol2" in molecule_file or ".sdf" in molecule_file:
    suppl = Chem.SDMolSupplier(molecule_file, sanitize=False)
    my_mol = suppl[0]
  elif ".pdbqt" in molecule_file:
    molecule_file = pdbqt_to_pdb(molecule_file)
    my_mol = Chem.MolFromPDBFile(molecule_file, sanitize=False)
  elif ".pdb" in molecule_file:
    my_mol = Chem.MolFromPDBFile(molecule_file, sanitize=False)
  else:
    raise ValueError("Unrecognized file type")

  if my_mol is None:
    raise ValueError("Unable to read non None Molecule Object")

  if calc_charges:
    my_mol = sanitize_mol(my_mol)
    AllChem.ComputeGasteigerChargers(my_mol)
  elif add_hydrogens:
    my_mol = sanitize_mol(my_mol)
    my_mol = Chem.AddHs(my_mol)

  xyz = get_xyz_from_mol(my_mol)

  return xyz, my_mol


def write_molecule(mol, outfile):
  if ".pdbqt" in outfile:
    # TODO (LESWING) create writer for pdbqt which includes charges
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
    pass
  elif ".pdb" in outfile:
    writer = Chem.PDBWriter(outfile)
    writer.write(mol)
    writer.close()
  else:
    raise ValueError("Unsupported Format")


def pdbqt_to_pdb(filename):
  base_filename = os.path.splitext(filename)[0]
  pdb_filename = base_filename + ".pdb"
  pdbqt_data = open(filename).readlines()
  with open(pdb_filename, 'w') as fout:
    for line in pdbqt_data:
      fout.write("%s\n" % line[:66])
  return pdb_filename


def FragIndicesToMol(oMol, indices):
  em = Chem.EditableMol(Chem.Mol())

  newIndices = {}
  for i, idx in enumerate(indices):
    em.AddAtom(oMol.GetAtomWithIdx(idx))
    newIndices[idx] = i

  for i, idx in enumerate(indices):
    at = oMol.GetAtomWithIdx(idx)
    for bond in at.GetBonds():
      if bond.GetBeginAtomIdx() == idx:
        oidx = bond.GetEndAtomIdx()
      else:
        oidx = bond.GetBeginAtomIdx()
      # make sure every bond only gets added once:
      if oidx < idx:
        continue
      em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
  res = em.GetMol()
  res.ClearComputedProps()
  Chem.GetSymmSSSR(res)
  res.UpdatePropertyCache(False)
  res._idxMap = newIndices
  return res


def _recursivelyModifyNs(mol, matches, indices=None):
  if indices is None:
    indices = []
  res = None
  while len(matches) and res is None:
    tIndices = indices[:]
    nextIdx = matches.pop(0)
    tIndices.append(nextIdx)
    nm = Chem.Mol(mol.ToBinary())
    nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
    nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
    cp = Chem.Mol(nm.ToBinary())
    try:
      Chem.SanitizeMol(cp)
    except ValueError:
      res, indices = _recursivelyModifyNs(nm, matches, indices=tIndices)
    else:
      indices = tIndices
      res = cp
  return res, indices


def AdjustAromaticNs(m, nitrogenPattern='[n&D2&H0;r5,r6]'):
  """
     default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
     to fix: O=c1ccncc1
  """
  Chem.GetSymmSSSR(m)
  m.UpdatePropertyCache(False)

  # break non-ring bonds linking rings:
  em = Chem.EditableMol(m)
  linkers = m.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
  plsFix = set()
  for a, b in linkers:
    em.RemoveBond(a, b)
    plsFix.add(a)
    plsFix.add(b)
  nm = em.GetMol()
  for at in plsFix:
    at = nm.GetAtomWithIdx(at)
    if at.GetIsAromatic() and at.GetAtomicNum() == 7:
      at.SetNumExplicitHs(1)
      at.SetNoImplicit(True)

  # build molecules from the fragments:
  fragLists = Chem.GetMolFrags(nm)
  frags = [FragIndicesToMol(nm, x) for x in fragLists]

  # loop through the fragments in turn and try to aromatize them:
  ok = True
  for i, frag in enumerate(frags):
    cp = Chem.Mol(frag.ToBinary())
    try:
      Chem.SanitizeMol(cp)
    except ValueError:
      matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
      lres, indices = _recursivelyModifyNs(frag, matches)
      if not lres:
        # print 'frag %d failed (%s)'%(i,str(fragLists[i]))
        ok = False
        break
      else:
        revMap = {}
        for k, v in frag._idxMap.iteritems():
          revMap[v] = k
        for idx in indices:
          oatom = m.GetAtomWithIdx(revMap[idx])
          oatom.SetNoImplicit(True)
          oatom.SetNumExplicitHs(1)
  if not ok:
    return None
  return m


def sanitize_mol(m):
  try:
    m.UpdatePropertyCache(False)
    cp = Chem.Mol(m.ToBinary())
    Chem.SanitizeMol(cp)
    return m
  except ValueError:
    pass
  try:
    nm = AdjustAromaticNs(m)
    if nm is not None:
      Chem.SanitizeMol(nm)
      return nm
    else:
      logging.warn("Unable To Sanitize Molecule")
      return m
  except ValueError:
    logging.warn("Unable To Sanitize Molecule")
    return m
