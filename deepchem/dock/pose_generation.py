"""
Generates protein-ligand docked poses using Autodock Vina.
"""
import platform
import deepchem
import logging
import numpy as np
import os
import tempfile
from subprocess import call
from deepchem.utils.rdkit_util import add_hydrogens_to_mol
from subprocess import check_output
from deepchem.utils import rdkit_util
from deepchem.utils import mol_xyz_util
from deepchem.utils import geometry_utils
from deepchem.utils import vina_utils

logger = logging.getLogger(__name__)


class PoseGenerator(object):
  """A Pose Generator computes low energy conformations for molecular complexes.

  Many questions in structural biophysics reduce to that of computing
  the binding free energy of molecular complexes. A key step towards
  computing the binding free energy of two complexes is to find low
  energy "poses", that is energetically favorable conformations of
  molecules with respect to each other. One application of this
  technique is to find low energy poses for protein-ligand
  interactions.
  """

  def generate_poses(self,
                     molecular_complex,
                     centroid=None,
                     box_dims=None,
                     exhaustiveness=10, 
                     num_modes=9, 
                     num_pockets=None,
                     out_dir=None):
    """Generates a list of low energy poses for molecular complex

    Parameters
    ----------
    molecular_complexes: list
      A representation of a molecular complex.
    centroid: np.ndarray, optional
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional
      Of shape `(3,)` holding the size of the box to dock. If not
      specified is set to size of molecular complex plus 5 angstroms.
    exhaustiveness: int, optional (default 10)
      Tells Autodock Vina how exhaustive it should be with pose
      generation.
    num_modes: int, optional (default 9)
      Tells Autodock Vina how many binding modes it should generate at
      each invocation.
    num_pockets: int, optional (default None)
      If specified, `self.pocket_finder` must be set. Will only
      generate poses for the first `num_pockets` returned by
      `self.pocket_finder`.
    out_dir: str, optional
      If specified, write generated poses to this directory.

    Returns
    -------
    A list of molecular complexes in energetically favorable poses. 
    """
    raise NotImplementedError


class VinaPoseGenerator(PoseGenerator):
  """Uses Autodock Vina to generate binding poses.

  This class uses Autodock Vina to make make predictions of
  binding poses. It downloads the Autodock Vina executable for
  your system to your specified DEEPCHEM_DATA_DIR (remember this
  is an environment variable you set) and invokes the executable
  to perform pose generation for you.

  Note
  ----
  This class requires RDKit to be installed.
  """

  def __init__(self, sixty_four_bits=True, pocket_finder=None):
    """Initializes Vina Pose Generator

    Params
    ------
    sixty_four_bits: bool, optional (default True)
      Specifies whether this is a 64-bit machine. Needed to download
      the correct executable. 
    pocket_finder: object, optional (default None)
      If specified should be an instance of
      `dc.dock.BindingPocketFinder`.
    """
    data_dir = deepchem.utils.get_data_dir()
    if platform.system() == 'Linux':
      url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz"
      filename = "autodock_vina_1_1_2_linux_x86.tgz" 
      dirname = "autodock_vina_1_1_2_linux_x86"
    elif platform.system() == 'Darwin':
      if sixty_four_bits:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac_64bit.tar.gz"
        filename = "autodock_vina_1_1_2_mac_64bit.tar.gz"
        dirname = "autodock_vina_1_1_2_mac_catalina_64bit"
      else:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac.tgz"
        filename = "autodock_vina_1_1_2_mac.tgz"
        dirname = "autodock_vina_1_1_2_mac"
    else:
      raise ValueError("This class can only run on Linux or Mac. If you are on Windows, please try using a cloud platform to run this code instead.")
    self.vina_dir = os.path.join(data_dir, dirname)
    self.pocket_finder = pocket_finder 
    if not os.path.exists(self.vina_dir):
      logger.info("Vina not available. Downloading")
      wget_cmd = "wget -nv -c -T 15 %s" % url 
      check_output(wget_cmd.split())
      logger.info("Downloaded Vina. Extracting")
      untar_cmd = "tar -xzvf %s" % filename
      check_output(untar_cmd.split())
      logger.info("Moving to final location")
      mv_cmd = "mv %s %s" % (dirname, data_dir)
      check_output(mv_cmd.split())
      logger.info("Cleanup: removing downloaded vina tar.gz")
      rm_cmd = "rm %s" % filename
      call(rm_cmd.split())
    self.vina_cmd = os.path.join(self.vina_dir, "bin/vina")

  def generate_poses(self,
                     molecular_complex,
                     centroid=None,
                     box_dims=None,
                     exhaustiveness=10, 
                     num_modes=9, 
                     num_pockets=None,
                     out_dir=None):
    """Generates the docked complex and outputs files for docked complex.

    TODO: How can this work on Windows? We need to install a .msi file and invoke it correctly from Python for this to work.

    TODO: Can we extract the autodock vina computed binding free energy? Need to parse the output from Autodock vina.

    Parameters
    ----------
    molecular_complexes: list
      A representation of a molecular complex.
    centroid: np.ndarray, optional
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional
      Of shape `(3,)` holding the size of the box to dock. If not
      specified is set to size of molecular complex plus 5 angstroms.
    exhaustiveness: int, optional (default 10)
      Tells Autodock Vina how exhaustive it should be with pose
      generation.
    num_modes: int, optional (default 9)
      Tells Autodock Vina how many binding modes it should generate at
      each invocation.
    num_pockets: int, optional (default None)
      If specified, `self.pocket_finder` must be set. Will only
      generate poses for the first `num_pockets` returned by
      `self.pocket_finder`.
    out_dir: str, optional
      If specified, write generated poses to this directory.

    Returns
    -------
    List of docked molecular complexes. Each entry in this list contains a `(protein_mol, ligand_mol)` pair of RDKit molecules.

    Raises
    ------
    `ValueError` if `num_pockets` is set but `self.pocket_finder is None`.
    """
    if out_dir is None:
      out_dir = tempfile.mkdtemp()

    if num_pockets is not None and self.pocket_finder is None:
      raise ValueError("If num_pockets is specified, pocket_finder must have been provided at construction time.")

    # Parse complex
    if len(molecular_complex) > 2:
      raise ValueError("Autodock Vina can only dock protein-ligand complexes and not more general molecular complexes.")

    (protein_file, ligand_file) = molecular_complex

    # Prepare protein 
    protein_name = os.path.basename(protein_file).split(".")[0]
    protein_hyd = os.path.join(out_dir, "%s_hyd.pdb" % protein_name)
    protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % protein_name)
    protein_mol = rdkit_util.load_molecule(
        protein_file, calc_charges=True, add_hydrogens=True)

    # Get protein centroid and range
    if centroid is not None and box_dims is not None:
      centroids = [centroid]
      dimensions = [box_dims]
    else:
      if self.pocket_finder is None:
        logger.info("Pockets not specified. Will use whole protein to dock")
        rdkit_util.write_molecule(protein_mol[1], protein_hyd, is_protein=True)
        rdkit_util.write_molecule(
            protein_mol[1], protein_pdbqt, is_protein=True)
        protein_centroid = geometry_utils.compute_centroid(protein_mol[0])
        protein_range = mol_xyz_util.get_molecule_range(protein_mol[0])
        box_dims = protein_range + 5.0
        centroids, dimensions = [protein_centroid], [box_dims]
      else:
        logger.info("About to find putative binding pockets")
        pockets = self.pocket_finder.find_pockets(
            (protein_file, ligand_file))
        logger.info("%d pockets found in total" % len(pockets))
        logger.info("Computing centroid and size of proposed pockets.")
        centroids, dimensions = [], []
        for pocket in pockets:
          protein_centroid = pocket.center()
          (x_min, x_max), (y_min, y_max), (z_min, z_max) = pocket.x_range, pocket.y_range, pocket.z_range
          # TODO(rbharath: Does vina divide box dimensions by 2?
          x_box = (x_max - x_min) / 2.
          y_box = (y_max - y_min) / 2.
          z_box = (z_max - z_min) / 2.
          box_dims = (x_box, y_box, z_box)
          centroids.append(protein_centroid)
          dimensions.append(box_dims)

    if num_pockets is not None:
      logger.info("num_pockets = %d so selecting this many pockets for docking." % num_pockets)
      centroids = centroids[:num_pockets]
      dimensions = dimensions[:num_pockets]

    # Prepare protein 
    ligand_name = os.path.basename(ligand_file).split(".")[0]
    ligand_pdbqt = os.path.join(out_dir, "%s.pdbqt" % ligand_name)

    ligand_mol = rdkit_util.load_molecule(
        ligand_file, calc_charges=True, add_hydrogens=True)
    rdkit_util.write_molecule(ligand_mol[1], ligand_pdbqt)

    docked_complexes = []
    for i, (protein_centroid, box_dims) in enumerate(zip(centroids, dimensions)):
      logger.info("Docking in pocket %d/%d" % (i, len(centroids)))
      logger.info("Docking with center: %s" % str(protein_centroid))
      logger.info("Box dimensions: %s" % str(box_dims))
      # Write Vina conf file
      conf_file = os.path.join(out_dir, "conf.txt")
      vina_utils.write_vina_conf(
          protein_pdbqt,
          ligand_pdbqt,
          protein_centroid,
          box_dims,
          conf_file,
          num_modes=num_modes,
          exhaustiveness=exhaustiveness)

      # Define locations of log and output files
      log_file = os.path.join(out_dir, "%s_log.txt" % ligand_name)
      out_pdbqt = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
      logger.info("About to call Vina")
      call(
          "%s --config %s --log %s --out %s" % (self.vina_cmd, conf_file,
                                                log_file, out_pdbqt),
          shell=True)
      ligands = vina_utils.load_docked_ligands(out_pdbqt)
      docked_complexes += [(protein_mol[1], ligand) for ligand in ligands]

    return docked_complexes
