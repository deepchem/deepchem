"""
Generates protein-ligand docked poses.
"""
import platform
import logging
import os
import tempfile
import numpy as np
from subprocess import Popen, PIPE
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

    def generate_poses(  # type: ignore
            self,
            molecular_complex: Tuple[str, str],
            centroid: Optional[np.ndarray] = None,
            box_dims: Optional[np.ndarray] = None,
            exhaustiveness: int = 10,
            num_modes: int = 9,
            num_pockets: Optional[int] = None,
            out_dir: Optional[str] = None,
            generate_scores: bool = True,
            **kwargs) -> Union[Tuple[DOCKED_POSES, np.ndarray], DOCKED_POSES]:
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

        protein_mol = load_molecule(protein_file,
                                    calc_charges=True,
                                    add_hydrogens=True)
        ligand_name = os.path.basename(ligand_file).split(".")[0]

        # Define locations of log and output files
        log_file = os.path.join(out_dir, "%s_log.txt" % ligand_name)
        out_file = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
        logger.info("About to call GNINA.")

        # Write GNINA conf file
        conf_file = os.path.join(out_dir, "conf.txt")
        write_gnina_conf(protein_filename=protein_file,
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
    binding poses.

    Example
    -------
    >> import deepchem as dc
    >> vpg = dc.dock.VinaPoseGenerator(pocket_finder=None)
    >> protein_file = '1jld_protein.pdb'
    >> ligand_file = '1jld_ligand.sdf'
    >> poses, scores = vpg.generate_poses(
    ..        (protein_file, ligand_file),
    ..        exhaustiveness=1,
    ..        num_modes=1,
    ..        out_dir=tmp,
    ..        generate_scores=True)

    Note
    ----
    This class requires RDKit and vina to be installed. As on 9-March-22,
    Vina is not available on Windows. Hence, this utility is currently
    available only on Ubuntu and MacOS.
    """

    def __init__(self, pocket_finder: Optional[BindingPocketFinder] = None):
        """Initializes Vina Pose Generator

        Parameters
        ----------
        pocket_finder: BindingPocketFinder, optional (default None)
            If specified should be an instance of
            `dc.dock.BindingPocketFinder`.
        """
        self.pocket_finder = pocket_finder

    def generate_poses(
            self,
            molecular_complex: Tuple[str, str],
            centroid: Optional[np.ndarray] = None,
            box_dims: Optional[np.ndarray] = None,
            exhaustiveness: int = 10,
            num_modes: int = 9,
            num_pockets: Optional[int] = None,
            out_dir: Optional[str] = None,
            generate_scores: Optional[bool] = False,
            **kwargs) -> Union[Tuple[DOCKED_POSES, List[float]], DOCKED_POSES]:
        """Generates the docked complex and outputs files for docked complex.

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
            Tells Autodock Vina how exhaustive it should be with pose generation. A
            higher value of exhaustiveness implies more computation effort for the
            docking experiment.
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
        kwargs:
            The kwargs - cpu, min_rmsd, max_evals, energy_range supported by VINA
            are as documented in https://autodock-vina.readthedocs.io/en/latest/vina.html

        Returns
        -------
        Tuple[`docked_poses`, `scores`] or `docked_poses` or `scores`
            Tuple of `(docked_poses, scores)`, `docked_poses`, or `scores`. `docked_poses`
            is a list of docked molecular complexes. Each entry in this list
            contains a `(protein_mol, ligand_mol)` pair of RDKit molecules.
            `scores` is a list of binding free energies predicted by Vina.

        Raises
        ------
        `ValueError` if `num_pockets` is set but `self.pocket_finder is None`.
        """
        if "cpu" in kwargs:
            cpu = kwargs["cpu"]
        else:
            cpu = 0
        if "min_rmsd" in kwargs:
            min_rmsd = kwargs["min_rmsd"]
        else:
            min_rmsd = 1.0
        if "max_evals" in kwargs:
            max_evals = kwargs["max_evals"]
        else:
            max_evals = 0
        if "energy_range" in kwargs:
            energy_range = kwargs["energy_range"]
        else:
            energy_range = 3.0

        try:
            from vina import Vina  # type: ignore
        except ModuleNotFoundError:
            raise ImportError("This function requires vina to be installed")

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
        if ".pdbqt" not in protein_file:
            protein_pdbqt = os.path.join(out_dir, "%s.pdbqt" % protein_name)
            protein_mol = load_molecule(protein_file,
                                        calc_charges=True,
                                        add_hydrogens=True)
            write_molecule(protein_mol[1], protein_hyd, is_protein=True)
            write_molecule(protein_mol[1], protein_pdbqt, is_protein=True)
        else:
            protein_pdbqt = protein_file

        # Get protein centroid and range
        if centroid is not None and box_dims is not None:
            centroids = [centroid]
            dimensions = [box_dims]
        else:
            if self.pocket_finder is None:
                logger.info(
                    "Pockets not specified. Will use whole protein to dock")
                centroids = [compute_centroid(protein_mol[0])]
                dimensions = [compute_protein_range(protein_mol[0]) + 5.0]
            else:
                logger.info("About to find putative binding pockets")
                pockets = self.pocket_finder.find_pockets(protein_file)
                logger.info("%d pockets found in total" % len(pockets))
                logger.info("Computing centroid and size of proposed pockets.")
                centroids, dimensions = [], []
                for pocket in pockets:
                    (x_min, x_max), (y_min, y_max), (
                        z_min,
                        z_max) = pocket.x_range, pocket.y_range, pocket.z_range
                    # TODO(rbharath: Does vina divide box dimensions by 2?
                    x_box = (x_max - x_min) / 2.
                    y_box = (y_max - y_min) / 2.
                    z_box = (z_max - z_min) / 2.
                    centroids.append(pocket.center())
                    dimensions.append(np.array((x_box, y_box, z_box)))

        if num_pockets is not None:
            logger.info(
                "num_pockets = %d so selecting this many pockets for docking." %
                num_pockets)
            centroids = centroids[:num_pockets]
            dimensions = dimensions[:num_pockets]

        # Prepare ligand
        ligand_name = os.path.basename(ligand_file).split(".")[0]
        if ".pdbqt" not in ligand_file:
            ligand_pdbqt = os.path.join(out_dir, "%s.pdbqt" % ligand_name)

            ligand_mol = load_molecule(ligand_file,
                                       calc_charges=True,
                                       add_hydrogens=True)
            write_molecule(ligand_mol[1], ligand_pdbqt)
        else:
            ligand_pdbqt = ligand_file

        docked_complexes = []
        all_scores = []
        vpg = Vina(sf_name='vina',
                   cpu=cpu,
                   seed=0,
                   no_refine=False,
                   verbosity=1)
        for i, (protein_centroid,
                box_dims) in enumerate(zip(centroids, dimensions)):
            logger.info("Docking in pocket %d/%d" % (i + 1, len(centroids)))
            logger.info("Docking with center: %s" % str(protein_centroid))
            logger.info("Box dimensions: %s" % str(box_dims))
            # Write Vina conf file
            conf_file = os.path.join(out_dir, "conf.txt")
            write_vina_conf(protein_pdbqt,
                            ligand_pdbqt,
                            protein_centroid,
                            box_dims,
                            conf_file,
                            num_modes=num_modes,
                            exhaustiveness=exhaustiveness)

            # Define locations of output files
            out_pdbqt = os.path.join(out_dir, "%s_docked.pdbqt" % ligand_name)
            logger.info("About to call Vina")

            vpg.set_receptor(protein_pdbqt)
            vpg.set_ligand_from_file(ligand_pdbqt)
            vpg.compute_vina_maps(center=protein_centroid, box_size=box_dims)
            vpg.dock(exhaustiveness=exhaustiveness,
                     n_poses=num_modes,
                     min_rmsd=min_rmsd,
                     max_evals=max_evals)
            vpg.write_poses(out_pdbqt,
                            n_poses=num_modes,
                            energy_range=energy_range,
                            overwrite=True)

            ligands, scores = load_docked_ligands(out_pdbqt)
            if '.pdbqt' not in protein_file:
                docked_complexes += [
                    (protein_mol[1], ligand) for ligand in ligands
                ]
                all_scores += scores
            else:
                all_scores += scores

        if '.pdbqt' not in ligand_file and '.pdbqt' not in protein_file:
            if generate_scores:
                return docked_complexes, all_scores
            else:
                return docked_complexes
        else:
            if generate_scores:
                return all_scores  # type: ignore
            else:
                raise 'PDBQT files failed to be properly converted into RDKit Mol objects'  # type: ignore
