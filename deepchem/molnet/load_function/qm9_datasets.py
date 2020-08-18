"""
qm9 dataset loader.
"""
import os
import logging
import deepchem

logger = logging.getLogger(__name__)

DEFAULT_DIR = deepchem.utils.get_data_dir()
GDB9_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"
QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"


def load_qm9(featurizer='CoulombMatrix',
             split='random',
             reload=True,
             move_mean=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load QM9 dataset

  QM9 is a comprehensive dataset that provides geometric, energetic, 
  electronic and thermodynamic properties for a subset of GDB-17 database, 
  comprising 134 thousand stable organic molecules with up to 9 heavy atoms.
  All moleucles are modeled using density functional theory
  (B3LYP/6-31G(2df,p) based DFT).

  Random splitting is recommended for this dataset.

  The source data contain:

  - qm9.sdf: molecular structures
  - qm9.sdf.csv: tables for molecular properties

    - "mol_id" - Molecule ID (gdb9 index) mapping to the .sdf file
    - "A" - Rotational constant (unit: GHz)
    - "B" - Rotational constant (unit: GHz)
    - "C" - Rotational constant (unit: GHz)
    - "mu" - Dipole moment (unit: D)
    - "alpha" - Isotropic polarizability (unit: Bohr^3)
    - "homo" - Highest occupied molecular orbital energy (unit: Hartree)
    - "lumo" - Lowest unoccupied molecular orbital energy (unit: Hartree)
    - "gap" - Gap between HOMO and LUMO (unit: Hartree)
    - "r2" - Electronic spatial extent (unit: Bohr^2)
    - "zpve" - Zero point vibrational energy (unit: Hartree)
    - "u0" - Internal energy at 0K (unit: Hartree)
    - "u298" - Internal energy at 298.15K (unit: Hartree)
    - "h298" - Enthalpy at 298.15K (unit: Hartree)
    - "g298" - Free energy at 298.15K (unit: Hartree)
    - "cv" - Heat capavity at 298.15K (unit: cal/(mol*K))
    - "u0_atom" - Atomization energy at 0K (unit: kcal/mol)
    - "u298_atom" - Atomization energy at 298.15K (unit: kcal/mol)
    - "h298_atom" - Atomization enthalpy at 298.15K (unit: kcal/mol)
    - "g298_atom" - Atomization free energy at 298.15K (unit: kcal/mol)

  "u0_atom" ~ "g298_atom" (used in MoleculeNet) are calculated from the 
  differences between "u0" ~ "g298" and sum of reference energies of all 
  atoms in the molecules, as given in
  https://figshare.com/articles/Atomref%3A_Reference_thermochemical_energies_of_H%2C_C%2C_N%2C_O%2C_F_atoms./1057643

  References
  ----------
  .. [1] Blum, Lorenz C., and Jean-Louis Reymond. "970 million druglike small
     molecules for virtual screening in the chemical universe database GDB-13."
     Journal of the American Chemical Society 131.25 (2009): 8732-8733.
  .. [2] Ramakrishnan, Raghunathan, et al. "Quantum chemistry structures and 
     properties of 134 kilo molecules." Scientific data 1 (2014): 140022.
  """
  # Featurize qm9 dataset
  logger.info("About to featurize qm9 dataset.")
  qm9_tasks = [
      "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
      "h298", "g298"
  ]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "qm9-featurized")
    if not move_mean:
      save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
    else:
      save_folder = os.path.join(save_folder, str(featurizer))

    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return qm9_tasks, all_dataset, transformers

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunctionInput', 'MP', 'Raw']:
    dataset_file = os.path.join(data_dir, "gdb9.sdf")

    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=GDB9_URL, dest_dir=data_dir)
      deepchem.utils.untargz_file(
          os.path.join(data_dir, 'gdb9.tar.gz'), data_dir)
  else:
    dataset_file = os.path.join(data_dir, "qm9.csv")
    if not os.path.exists(dataset_file):
      deepchem.utils.download_url(url=QM9_CSV_URL, dest_dir=data_dir)

  if featurizer in ['CoulombMatrix', 'BPSymmetryFunctionInput', 'MP', 'Raw']:
    if featurizer == 'CoulombMatrix':
      featurizer = deepchem.feat.CoulombMatrix(29)
    elif featurizer == 'BPSymmetryFunctionInput':
      featurizer = deepchem.feat.BPSymmetryFunctionInput(29)
    elif featurizer == 'Raw':
      featurizer = deepchem.feat.RawFeaturizer()
    elif featurizer == 'MP':
      featurizer = deepchem.feat.WeaveFeaturizer(
          graph_distance=False, explicit_H=True)
    loader = deepchem.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)
  else:
    if featurizer == 'ECFP':
      featurizer = deepchem.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
      featurizer = deepchem.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
      featurizer = deepchem.feat.WeaveFeaturizer()
    elif featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      img_size = kwargs.get("img_size", 80)
      featurizer = deepchem.feat.SmilesToImage(
          img_size=img_size, img_spec=img_spec)
    loader = deepchem.data.CSVLoader(
        tasks=qm9_tasks, smiles_field="smiles", featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  if split == None:
    raise ValueError()

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=11)
  }
  splitter = splitters[split]
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=train_dataset, move_mean=move_mean)
  ]
  for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset = transformer.transform(test_dataset)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(
        save_folder, train_dataset, valid_dataset, test_dataset, transformers)
  return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers
