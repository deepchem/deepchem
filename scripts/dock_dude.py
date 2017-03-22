"""
Pseudocode/TODOs

(0) create new directory with same subdirectories 

(1) prepare receptors:
  for each subdir in dud-e, find the receptor pdb, and prepare it. save it in new director

(2) prepare ligands:
  for each ligand: 
      if ligand does not already exist in new dir: prepare ligand 

      might have to use pybel to split up the mol2.gz
      cf: https://sourceforge.net/p/openbabel/mailman/message/27353258/
in between: make a pandas data frame with all ligand-protein combinations.

(3) dock:
  for each receptor, for each ligand (parallelize with iPython over rows in DF):
      find associated receptor and ligand in new directory
      do docking
      save in a new "docking" directory within receptor
      skip docking if that new docked pose already exists. 

"""
from __future__ import print_function
import os
import subprocess
from deepchem.feat.nnscore_utils import hydrogenate_and_compute_partial_charges
import glob
import numpy as np
import time
from functools import partial
from deepchem.utils import rdkit_util
from deepchem.utils import mol_xyz_util
from multiprocessing import Pool
from rdkit import Chem

VINA_EXECUTABLE = ""


def prepare_receptors(dude_dir, new_dir):
  broken_receptors = list()
  for subdir, dirs, files in os.walk(dude_dir):
    receptor_name = os.path.basename(subdir)
    print("Currently examining receptor %s " % receptor_name)
    save_dir = os.path.join(new_dir, receptor_name)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    receptor_filename = os.path.join(subdir, "receptor.pdb")
    if not os.path.exists(receptor_filename):
      continue
    prepared_filename = os.path.join(save_dir, "%s.pdb" % receptor_name)
    prepared_pdbqt = os.path.join(save_dir, "%s.pdbqt" % receptor_name)

    if os.path.exists(prepared_pdbqt):
      continue

    try:
      hydrogenate_and_compute_partial_charges(
          receptor_filename,
          ".pdb",
          hyd_output=prepared_filename,
          pdbqt_output=prepared_pdbqt,
          verbose=False)
    except ValueError:
      broken_receptors.append(receptor_filename)
      print("Unable to prepare %s" % receptor_filename)


def generate_ligand_mol2(mol, save_dir):
  mol_name = str(mol.OBMol.GetTitle())
  # print("Preparing ligand %s" % mol_name)
  filename = mol_name + ".mol2"
  filename = os.path.join(save_dir, filename)
  prepared_filename = os.path.join(save_dir, "%s_prepared.pdb" % mol_name)
  prepared_pdbqt = os.path.join(save_dir, "%s_prepared.pdbqt" % mol_name)
  if os.path.exists(prepared_pdbqt):
    return mol_name
  output = open(filename, "w")  # erase existing file, if any
  output.write(mol.write("mol2"))
  output.close()

  return mol_name


def prepare_ligand(args):
  mol_name, mol, save_dir = args[0], args[1], args[2]
  filename = str(mol_name) + ".sdf"
  filename = os.path.join(save_dir, filename)
  rdkit_util.write_molecule(mol, filename)
  prepared_filename = os.path.join(save_dir, "%s_prepared.pdb" % mol_name)
  prepared_pdbqt = os.path.join(save_dir, "%s_prepared.pdbqt" % mol_name)
  if os.path.exists(prepared_pdbqt):
    return

  hydrogenate_and_compute_partial_charges(
      filename,
      "sdf",
      hyd_output=prepared_filename,
      pdbqt_output=prepared_pdbqt,
      verbose=False,
      protein=False)


def prepare_ligands(mol2_file, save_dir, worker_pool=None):
  print("mol2_file")
  print(mol2_file)

  mol_data = [tuple(x + (save_dir,)) for x in read_mol2_file(mol2_file)]

  if worker_pool is not None:
    worker_pool.map(prepare_ligand, mol_data)
  else:
    for mol_datum in mol_data:
      prepare_ligand(mol_datum)


def prepare_ligands_in_directory(dude_dir,
                                 new_dir,
                                 receptor_name=None,
                                 worker_pool=None):
  subdirs = sorted(glob.glob(os.path.join(dude_dir, '*/')))
  print("Searching for receptor %s" % receptor_name)
  for subdir in subdirs:
    print("subdir: %s" % subdir)
    subdir = subdir.rstrip('/')
    if receptor_name == os.path.basename(subdir):
      print("Found receptor %s in subdirectory %s" % (receptor_name, subdir))
      break

  receptor_name = os.path.basename(subdir)
  print("Currently examining receptor %s " % receptor_name)
  save_dir = os.path.join(new_dir, receptor_name)
  input_mol2gz = os.path.join(subdir, "actives_final.mol2.gz")
  output_mol2 = os.path.join(subdir, "actives_final.mol2")
  try:
    subprocess.call(
        "gunzip < %s > %s" % (input_mol2gz, output_mol2), shell=True)
  except:
    pass

  print("output_mol2")
  print(output_mol2)

  if not os.path.exists(output_mol2):
    return

  prepare_ligands(output_mol2, save_dir, worker_pool=worker_pool)

  input_mol2gz = os.path.join(subdir, "decoys_final.mol2.gz")
  output_mol2 = os.path.join(subdir, "decoys_final.mol2")
  try:
    subprocess.call(
        "gunzip < %s > %s" % (input_mol2gz, output_mol2), shell=True)
  except:
    pass

  prepare_ligands(output_mol2, save_dir, worker_pool=worker_pool)


def write_conf(receptor_filename,
               ligand_filename,
               centroid,
               box_dims,
               conf_filename,
               exhaustiveness=None):
  with open(conf_filename, "w") as f:
    f.write("receptor = %s\n" % receptor_filename)
    f.write("ligand = %s\n\n" % ligand_filename)

    f.write("center_x = %f\n" % centroid[0])
    f.write("center_y = %f\n" % centroid[1])
    f.write("center_z = %f\n\n" % centroid[2])

    f.write("size_x = %f\n" % box_dims[0])
    f.write("size_y = %f\n" % box_dims[1])
    f.write("size_z = %f\n\n" % box_dims[2])

    if exhaustiveness is not None:
      f.write("exhaustiveness = %d\n" % exhaustiveness)

  return


def dock_ligand_to_receptor(ligand_file, receptor_filename, protein_centroid,
                            box_dims, subdir, exhaustiveness):
  head, tail = os.path.split(ligand_file)
  ligand_name = os.path.splitext(tail)[0]
  print("Docking ligand %s to receptor %s" % (ligand_name, receptor_filename))
  conf_filename = os.path.join(subdir, "%s_conf.txt" % ligand_name)
  write_conf(
      receptor_filename,
      ligand_file,
      protein_centroid,
      box_dims,
      conf_filename,
      exhaustiveness=exhaustiveness)

  log_filename = os.path.join(subdir, "%s_log.txt" % ligand_name)
  out_filename = os.path.join(subdir, "%s_docked.pdbqt" % ligand_name)
  if os.path.exists(out_filename):
    return out_filename

  start = time.time()
  subprocess.call(
      "%s --config %s --log %s --out %s" %
      (VINA_EXECUTABLE, conf_filename, log_filename, out_filename),
      shell=True)
  total_time = time.time() - start
  with open(log_filename, "a") as f:
    f.write("total time = %s" % (str(total_time)))

  return out_filename


def dock_ligands_to_receptors(docking_dir,
                              worker_pool=None,
                              exhaustiveness=None,
                              chosen_receptor=None,
                              restrict_box=True):
  subdirs = glob.glob(os.path.join(docking_dir, '*/'))
  for subdir in subdirs:
    subdir = subdir.rstrip('/')
    receptor_name = os.path.basename(subdir)
    if chosen_receptor is not None and chosen_receptor != receptor_name:
      continue
    print("receptor name = %s" % receptor_name)
    receptor_filename = os.path.join(subdir, "%s.pdbqt" % receptor_name)
    if not os.path.exists(receptor_filename):
      continue

    print("Examining %s" % receptor_filename)

    receptor_mol = rdkit_util.load_molecule(
        os.path.join(subdir, "%s.pdb" % receptor_name))
    protein_centroid = mol_xyz_util.get_molecule_centroid(receptor_mol[0])
    protein_range = mol_xyz_util.get_molecule_range(receptor_mol[0])

    box_dims = protein_range + 5.0

    ligands = sorted(glob.glob(os.path.join(subdir, '*_prepared.pdbqt')))
    print("Num ligands = %d" % len(ligands))

    dock_ligand_to_receptor_partial = partial(
        dock_ligand_to_receptor,
        receptor_filename=receptor_filename,
        protein_centroid=protein_centroid,
        box_dims=box_dims,
        subdir=subdir,
        exhaustiveness=exhaustiveness)

    if restrict_box:
      active_ligand = ""
      for ligand in ligands:
        if "CHEM" in ligand:
          active_ligand = ligand
          break

      print("Docking to %s first to ascertain centroid and box dimensions" %
            active_ligand)

      out_pdb_qt = dock_ligand_to_receptor_partial(active_ligand)
      ligand_pybel = rdkit_util.load_molecule(out_pdb_qt)
      ligand_centroid = mol_xyz_util.get_molecule_centroid(ligand_pybel[0])
      print("Protein centroid = %s" % (str(protein_centroid)))
      print("Ligand centroid = %s" % (str(ligand_centroid)))
      box_dims = np.array([20., 20., 20.])
      dock_ligand_to_receptor_partial = partial(
          dock_ligand_to_receptor,
          receptor_filename=receptor_filename,
          protein_centroid=ligand_centroid,
          box_dims=box_dims,
          subdir=subdir,
          exhaustiveness=exhaustiveness)

      print("Finished docking to %s, docking to remainder of ligands now." %
            active_ligand)

    if worker_pool is None:
      for i, ligand_file in enumerate(ligands):
        a = time.time()
        dock_ligand_to_receptor_partial(ligand_file)
        print("took %f seconds to dock single ligand." % (time.time() - a))
    else:
      print("parallelizing docking over worker pool")

      worker_pool.map(dock_ligand_to_receptor_partial, ligands)


def prepare_ligands_and_dock_ligands_to_receptors(dude_dir, docking_dir,
                                                  worker_pool):
  subdirs = sorted(glob.glob(os.path.join(docking_dir, '*/')))
  for subdir in subdirs:
    subdir = subdir.rstrip('/')
    receptor_name = os.path.basename(subdir)
    print("Preparing ligands and then docking to %s" % receptor_name)
    prepare_ligands_in_directory(dude_dir, docking_dir, receptor_name, None)
    dock_ligands_to_receptors(
        docking_dir, worker_pool, chosen_receptor=receptor_name)


def prepare_receptors_prepare_ligands_dock_ligands_to_receptors(
    dude_dir, docking_dir, worker_pool):
  prepare_receptors(dude_dir, docking_dir)
  prepare_ligands_and_dock_ligands_to_receptors(dude_dir, docking_dir,
                                                worker_pool)


def read_mol2_file(mol2_filename):
  mol_header = "@<TRIPOS>MOLECULE"
  data = open(mol2_filename).read()
  mol_blocks = data.split(mol_header)
  retval = []
  for mol_block in mol_blocks[1:]:
    mol_name = mol_block.split()[0]
    mol_block = mol_header + mol_block
    mol = Chem.MolFromMol2Block(mol_block)
    retval.append((mol_name, mol))
  return retval


if __name__ == "__main__":
  import sys

  if len(sys.argv) not in (4, 5):
    print("""
    python dock_dude.py <dude_directory> <desired_docked_directory> <vina_executable> <optional_num_threads>
    """)
    sys.exit(1)
  dude_dir = sys.argv[1]
  docking_dir = sys.argv[2]
  VINA_EXECUTABLE = sys.argv[3]
  if len(sys.argv) == 5:
    pool = Pool(int(sys.argv[5]))
  else:
    pool = None
  prepare_receptors_prepare_ligands_dock_ligands_to_receptors(dude_dir,
                                                              docking_dir, pool)
