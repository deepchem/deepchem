"""
Generates protein-ligand docked poses.
"""
import platform
import logging
import os
import tempfile
import tarfile
import numpy as np
from subprocess import call, Popen, PIPE
from subprocess import check_output
from typing import List, Optional, Tuple, Union

from deepchem.dock.binding_pocket import BindingPocketFinder
from deepchem.utils.data_utils import download_url, get_data_dir
from deepchem.utils.typing import RDKitMol
from deepchem.utils.geometry_utils import compute_centroid, compute_protein_range
from deepchem.utils.rdkit_utils import load_molecule, write_molecule
from deepchem.utils.docking_utils import load_docked_ligands, write_vina_conf, write_gnina_conf, read_gnina_log

logger = logging.getLogger(__name__)
DOCKED_POSES = List[Tuple[RDKitMol, RDKitMol]]


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
                     molecular_complex: Tuple[str, str],
                     centroid: Optional[np.ndarray] = None,
                     box_dims: Optional[np.ndarray] = None,
                     exhaustiveness: int = 10,
                     num_modes: int = 9,
                     num_pockets: Optional[int] = None,
                     out_dir: Optional[str] = None,
                     generate_scores: bool = False):
    """Generates a list of low energy poses for molecular complex

    Parameters
    ----------
    molecular_complexes: Tuple[str, str]
      A representation of a molecular complex. This tuple is
      (protein_file, ligand_file).
    centroid: np.ndarray, optional (default None)
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional (default None)
      A numpy array of shape `(3,)` holding the size of the box to dock.
      If not specified is set to size of molecular complex plus 5 angstroms.
    exhaustiveness: int, optional (default 10)
      Tells pose generator how exhaustive it should be with pose
      generation.
    num_modes: int, optional (default 9)
      Tells pose generator how many binding modes it should generate at
      each invocation.
    num_pockets: int, optional (default None)
      If specified, `self.pocket_finder` must be set. Will only
      generate poses for the first `num_pockets` returned by
      `self.pocket_finder`.
    out_dir: str, optional (default None)
      If specified, write generated poses to this directory.
    generate_score: bool, optional (default False)
      If `True`, the pose generator will return scores for complexes.
      This is used typically when invoking external docking programs
      that compute scores.

    Returns
    -------
    A list of molecular complexes in energetically favorable poses.
    """
    raise NotImplementedError


class GninaPoseGenerator(PoseGenerator):
  """Use GNINA to generate binding poses.

  This class uses GNINA (a deep learning framework for molecular
  docking) to generate binding poses. It downloads the GNINA
  executable to DEEPCHEM_DATA_DIR (an environment variable you set)
  and invokes the executable to perform pose generation.

  GNINA uses pre-trained convolutional neural network (CNN) scoring
  functions to rank binding poses based on learned representations of
  3D protein-ligand interactions. It has been shown to outperform
  AutoDock Vina in virtual screening applications [1]_.

  If you use the GNINA molecular docking engine, please cite the relevant
  papers: https://github.com/gnina/gnina#citation
  The primary citation for GNINA is [1]_.

  References
  ----------
  .. [1] M Ragoza, J Hochuli, E Idrobo, J Sunseri, DR Koes.
  "Protein–Ligand Scoring with Convolutional Neural Networks."
  Journal of chemical information and modeling (2017).

  Note
  ----
  * GNINA currently only works on Linux operating systems.
  * GNINA requires CUDA >= 10.1 for fast CNN scoring.
  * Almost all dependencies are included in the most compatible way
    possible, which reduces performance. Build GNINA from source
    for production use.

  """

  def __init__(self):
    """Initialize GNINA pose generator."""

    data_dir = get_data_dir()
    if platform.system() == 'Linux':
      url = "https://github.com/gnina/gnina/releases/download/v1.0/gnina"
      filename = 'gnina'
      self.gnina_dir = data_dir
      self.gnina_cmd = os.path.join(self.gnina_dir, filename)
    else:
      raise ValueError(
          "GNINA currently only runs on Linux. Try using a cloud platform to run this code instead."
      )

    if not os.path.exists(self.gnina_cmd):
      logger.info("GNINA not available. Downloading...")
      download_url(url, data_dir)
      downloaded_file = os.path.join(data_dir, filename)
      os.chmod(downloaded_file, 755)
      logger.info("Downloaded GNINA.")

  def generate_poses(
      self,
      molecular_complex: Tuple[str, str],
      centroid: Optional[np.ndarray] = None,
      box_dims: Optional[np.ndarray] = None,
      exhaustiveness: int = 10,
      num_modes: int = 9,
      num_pockets: Optional[int] = None,
      out_dir: Optional[str] = None,
      generate_scores: bool = True,
      **kwargs) -> Union[Tuple[DOCKED_POSES, List[float]], DOCKED_POSES]:
    """Generates the docked complex and outputs files for docked complex.

    Parameters
    ----------
    molecular_complexes: Tuple[str, str]
      A representation of a molecular complex. This tuple is
      (protein_file, ligand_file).
    centroid: np.ndarray, optional (default None)
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional (default None)
      A numpy array of shape `(3,)` holding the size of the box to dock.
      If not specified is set to size of molecular complex plus 4 angstroms.
    exhaustiveness: int (default 8)
      Tells GNINA how exhaustive it should be with pose
      generation.
    num_modes: int (default 9)
      Tells GNINA how many binding modes it should generate at
      each invocation.
    out_dir: str, optional
      If specified, write generated poses to this directory.
    generate_scores: bool, optional (default True)
      If `True`, the pose generator will return scores for complexes.
      This is used typically when invoking external docking programs
      that compute scores.
    kwargs:
      Any args supported by GNINA as documented
      https://github.com/gnina/gnina#usage

    Returns
    -------
    Tuple[`docked_poses`, `scores`] or `docked_poses`
      Tuple of `(docked_poses, scores)` or `docked_poses`. `docked_poses`
      is a list of docked molecular complexes. Each entry in this list
      contains a `(protein_mol, ligand_mol)` pair of RDKit molecules.
      `scores` is an array of binding affinities (kcal/mol),
      CNN pose scores, and CNN affinities predicted by GNINA.

    """

    if out_dir is None:
      out_dir = tempfile.mkdtemp()
    if not os.path.exists(out_dir):
      os.makedirs(out_dir)

    # Parse complex
    if len(molecular_complex) > 2:
      raise ValueError(
          "GNINA can only dock protein-ligand complexes and not more general molecular complexes."
      )

    (protein_file, ligand_file) = molecular_complex

    # check filetypes
    if not protein_file.endswith('.pdb'):
      raise ValueError('Protein file must be in .pdb format.')
    if not ligand_file.endswith('.sdf'):
      raise ValueError('Ligand file must be in .sdf format.')

    protein_mol = load_molecule(
        protein_file, calc_charges=True, add_hydrogens=True)
    ligand_name = os.path.basename(ligand_file).split(".")[0]

    # Define locations of log and output files
    log_file = os.path.join(out_dir, "%s_log.txt" % ligand_name)
    out_file = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
    logger.info("About to call GNINA.")

    # Write GNINA conf file
    conf_file = os.path.join(out_dir, "conf.txt")
    write_gnina_conf(
        protein_filename=protein_file,
        ligand_filename=ligand_file,
        conf_filename=conf_file,
        num_modes=num_modes,
        exhaustiveness=exhaustiveness,
        **kwargs)

    # Run GNINA
    args = [
        self.gnina_cmd, "--config", conf_file, "--log", log_file, "--out",
        out_file
    ]
    process = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    # read output and log
    ligands, _ = load_docked_ligands(out_file)
    docked_complexes = [(protein_mol[1], ligand) for ligand in ligands]
    scores = read_gnina_log(log_file)

    if generate_scores:
      return docked_complexes, scores
    else:
      return docked_complexes


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

  def __init__(self,
               sixty_four_bits: bool = True,
               pocket_finder: Optional[BindingPocketFinder] = None):
    """Initializes Vina Pose Generator

    Parameters
    ----------
    sixty_four_bits: bool, optional (default True)
      Specifies whether this is a 64-bit machine. Needed to download
      the correct executable.
    pocket_finder: BindingPocketFinder, optional (default None)
      If specified should be an instance of
      `dc.dock.BindingPocketFinder`.
    """
    data_dir = get_data_dir()
    if platform.system() == 'Linux':
      url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz"
      filename = "autodock_vina_1_1_2_linux_x86.tgz"
      dirname = "autodock_vina_1_1_2_linux_x86"
      self.vina_dir = os.path.join(data_dir, dirname)
      self.vina_cmd = os.path.join(self.vina_dir, "bin/vina")
    elif platform.system() == 'Darwin':
      if sixty_four_bits:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac_64bit.tar.gz"
        filename = "autodock_vina_1_1_2_mac_64bit.tar.gz"
        dirname = "autodock_vina_1_1_2_mac_catalina_64bit"
      else:
        url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_mac.tgz"
        filename = "autodock_vina_1_1_2_mac.tgz"
        dirname = "autodock_vina_1_1_2_mac"
      self.vina_dir = os.path.join(data_dir, dirname)
      self.vina_cmd = os.path.join(self.vina_dir, "bin/vina")
    elif platform.system() == 'Windows':
      url = "http://vina.scripps.edu/download/autodock_vina_1_1_2_win32.msi"
      filename = "autodock_vina_1_1_2_win32.msi"
      self.vina_dir = "\\Program Files (x86)\\The Scripps Research Institute\\Vina"
      self.vina_cmd = os.path.join(self.vina_dir, "vina.exe")
    else:
      raise ValueError(
          "Unknown operating system. Try using a cloud platform to run this code instead."
      )
    self.pocket_finder = pocket_finder
    if not os.path.exists(self.vina_dir):
      logger.info("Vina not available. Downloading")
      download_url(url, data_dir)
      downloaded_file = os.path.join(data_dir, filename)
      logger.info("Downloaded Vina. Extracting")
      if platform.system() == 'Windows':
        msi_cmd = "msiexec /i %s" % downloaded_file
        check_output(msi_cmd.split())
      else:
        with tarfile.open(downloaded_file) as tar:
          tar.extractall(data_dir)
      logger.info("Cleanup: removing downloaded vina tar.gz")
      os.remove(downloaded_file)

  def generate_poses(self,
                     molecular_complex: Tuple[str, str],
                     centroid: Optional[np.ndarray] = None,
                     box_dims: Optional[np.ndarray] = None,
                     exhaustiveness: int = 10,
                     num_modes: int = 9,
                     num_pockets: Optional[int] = None,
                     out_dir: Optional[str] = None,
                     generate_scores: Optional[bool] = False
                    ) -> Union[Tuple[DOCKED_POSES, List[float]], DOCKED_POSES]:
    """Generates the docked complex and outputs files for docked complex.

    TODO: How can this work on Windows? We need to install a .msi file and
    invoke it correctly from Python for this to work.

    Parameters
    ----------
    molecular_complexes: Tuple[str, str]
      A representation of a molecular complex. This tuple is
      (protein_file, ligand_file). The protein should be a pdb file
      and the ligand should be an sdf file.
    centroid: np.ndarray, optional
      The centroid to dock against. Is computed if not specified.
    box_dims: np.ndarray, optional
      A numpy array of shape `(3,)` holding the size of the box to dock. If not
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
    generate_score: bool, optional (default False)
      If `True`, the pose generator will return scores for complexes.
      This is used typically when invoking external docking programs
      that compute scores.

    Returns
    -------
    Tuple[`docked_poses`, `scores`] or `docked_poses`
      Tuple of `(docked_poses, scores)` or `docked_poses`. `docked_poses`
      is a list of docked molecular complexes. Each entry in this list
      contains a `(protein_mol, ligand_mol)` pair of RDKit molecules.
      `scores` is a list of binding free energies predicted by Vina.

    Raises
    ------
    `ValueError` if `num_pockets` is set but `self.pocket_finder is None`.
    """
    if out_dir is None:
      out_dir = tempfile.mkdtemp()

    if num_pockets is not None and self.pocket_finder is None:
      raise ValueError(
          "If num_pockets is specified, pocket_finder must have been provided at construction time."
      )

    # Parse complex
    if len(molecular_complex) > 2:
      raise ValueError(
          "Autodock Vina can only dock protein-ligand complexes and not more general molecular complexes."
      )

    (protein_file, ligand_file) = molecular_complex

    # Prepare protein
    protein_name = os.path.basename(protein_file).split(".")[0]
    protein_hyd = os.path.join(out_dir, "%s_hyd.pdb" % protein_name)
    protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % protein_name)
    protein_mol = load_molecule(
        protein_file, calc_charges=True, add_hydrogens=True)
    write_molecule(protein_mol[1], protein_hyd, is_protein=True)
    write_molecule(protein_mol[1], protein_pdbqt, is_protein=True)

    # Get protein centroid and range
    if centroid is not None and box_dims is not None:
      centroids = [centroid]
      dimensions = [box_dims]
    else:
      if self.pocket_finder is None:
        logger.info("Pockets not specified. Will use whole protein to dock")
        protein_centroid = compute_centroid(protein_mol[0])
        protein_range = compute_protein_range(protein_mol[0])
        box_dims = protein_range + 5.0
        centroids, dimensions = [protein_centroid], [box_dims]
      else:
        logger.info("About to find putative binding pockets")
        pockets = self.pocket_finder.find_pockets(protein_file)
        logger.info("%d pockets found in total" % len(pockets))
        logger.info("Computing centroid and size of proposed pockets.")
        centroids, dimensions = [], []
        for pocket in pockets:
          protein_centroid = pocket.center()
          (x_min, x_max), (y_min, y_max), (
              z_min, z_max) = pocket.x_range, pocket.y_range, pocket.z_range
          # TODO(rbharath: Does vina divide box dimensions by 2?
          x_box = (x_max - x_min) / 2.
          y_box = (y_max - y_min) / 2.
          z_box = (z_max - z_min) / 2.
          box_dims = (x_box, y_box, z_box)
          centroids.append(protein_centroid)
          dimensions.append(box_dims)

    if num_pockets is not None:
      logger.info("num_pockets = %d so selecting this many pockets for docking."
                  % num_pockets)
      centroids = centroids[:num_pockets]
      dimensions = dimensions[:num_pockets]

    # Prepare ligand
    ligand_name = os.path.basename(ligand_file).split(".")[0]
    ligand_pdbqt = os.path.join(out_dir, "%s.pdbqt" % ligand_name)

    ligand_mol = load_molecule(
        ligand_file, calc_charges=True, add_hydrogens=True)
    write_molecule(ligand_mol[1], ligand_pdbqt)

    docked_complexes = []
    all_scores = []
    for i, (protein_centroid, box_dims) in enumerate(
        zip(centroids, dimensions)):
      logger.info("Docking in pocket %d/%d" % (i + 1, len(centroids)))
      logger.info("Docking with center: %s" % str(protein_centroid))
      logger.info("Box dimensions: %s" % str(box_dims))
      # Write Vina conf file
      conf_file = os.path.join(out_dir, "conf.txt")
      write_vina_conf(
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
      if platform.system() == 'Windows':
        args = [
            self.vina_cmd, "--config", conf_file, "--log", log_file, "--out",
            out_pdbqt
        ]
      else:
        # I'm not sure why specifying the args as a list fails on other platforms,
        # but for some reason it only works if I pass it as a string.
        # FIXME: Incompatible types in assignment
        args = "%s --config %s --log %s --out %s" % (  # type: ignore
            self.vina_cmd, conf_file, log_file, out_pdbqt)
      # FIXME: We should use `subprocess.run` instead of `call`
      call(args, shell=True)
      ligands, scores = load_docked_ligands(out_pdbqt)
      docked_complexes += [(protein_mol[1], ligand) for ligand in ligands]
      all_scores += scores

    if generate_scores:
      return docked_complexes, all_scores
    else:
      return docked_complexes
