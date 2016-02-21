import tempfile
import os
import mdtraj as md
import numpy as np

def combine_mdtraj(protein, ligand):
  chain = protein.topology.add_chain()
  residue = protein.topology.add_residue("LIG", chain, resSeq=1)
  for atom in ligand.topology.atoms:
      protein.topology.add_atom(atom.name, atom.element, residue)
  protein.xyz = np.hstack([protein.xyz, ligand.xyz])
  protein.topology.create_standard_bonds()
  return protein

def visualize_complex(complex_mdtraj):
  ligand_atoms = [a.index for a in complex_mdtraj.topology.atoms if "LIG" in str(a.residue)]
  binding_pocket_atoms = md.compute_neighbors(complex_mdtraj, 0.5, ligand_atoms)[0]
  binding_pocket_residues = list(set([complex_mdtraj.topology.atom(a).residue.resSeq for a in binding_pocket_atoms]))
  binding_pocket_residues = [str(r) for r in binding_pocket_residues]
  binding_pocket_residues = " or ".join(binding_pocket_residues)

  traj = nglview.MDTrajTrajectory( complex_mdtraj ) # load file from RCSB PDB
  ngltraj = nglview.NGLWidget( traj )
  ngltraj.representations = [
  { "type": "cartoon", "params": {
  "sele": "protein", "color": "residueindex"
  } },
  { "type": "licorice", "params": {
  "sele": "(not hydrogen) and (resi (%s))" %  binding_pocket_residues
  } },
  { "type": "ball+stick", "params": {
  "sele": "resn LIG"
  } }
  ]
  return ngltraj

def convert_lines_to_mdtraj(molecule_lines):
  tempdir = tempfile.mkdtemp()
  molecule_file = os.path.join(tempdir, "molecule.pdb")
  with open(molecule_file, "wb") as f:
    f.writelines(molecule_lines)
  molecule_mdtraj = md.load(molecule_file)
  return molecule_mdtraj