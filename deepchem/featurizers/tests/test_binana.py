"""
Test NNScore Binana featurizer.

TODO(rbharath): There still isn't an example structure that exhibits
salt-bridge interactions. There might be a bug in the pi-T interaction
finger, and the H-bonds are known to miss some potential bonds with an
overly-conservative bond-angle cutoff.
"""
import os
import unittest
from deepchem.featurizers.nnscore import Binana
from deepchem.featurizers.nnscore import compute_hydrophobic_contacts
from deepchem.featurizers.nnscore import compute_electrostatic_energy
from deepchem.featurizers.nnscore import compute_ligand_atom_counts
from deepchem.featurizers.nnscore import compute_active_site_flexibility
from deepchem.featurizers.nnscore import compute_pi_t
from deepchem.featurizers.nnscore import compute_pi_cation
from deepchem.featurizers.nnscore import compute_pi_pi_stacking
from deepchem.featurizers.nnscore import compute_hydrogen_bonds
from deepchem.featurizers.nnscore import compute_contacts
from deepchem.featurizers.nnscore import compute_salt_bridges
from deepchem.featurizers.nnscore_pdb import PDB
from vs_utils.utils.tests import __file__ as test_directory

def data_dir():
  """Get location of data directory."""
  return os.path.join(os.path.dirname(test_directory), "data")

class TestBinana(unittest.TestCase):
  """
  Test Binana Binding Pose Featurizer.
  """
  def setUp(self):
    """
    Instantiate local copy of Binana object.
    """
    self.binana = Binana()

    ### 3zp9 comes from PDBBind-CN
    _3zp9_protein = PDB()
    _3zp9_protein_pdb = os.path.join(data_dir(), "3zp9_protein_hyd.pdb")
    _3zp9_protein_pdbqt = os.path.join(data_dir(), "3zp9_protein_hyd.pdbqt")
    _3zp9_protein.load_from_files(_3zp9_protein_pdb, _3zp9_protein_pdbqt)
    # The ligand is also specified by pdbbind
    _3zp9_ligand = PDB()
    _3zp9_ligand_pdb = os.path.join(data_dir(), "3zp9_ligand_hyd.pdb")
    _3zp9_ligand_pdbqt = os.path.join(data_dir(), "3zp9_ligand_hyd.pdbqt")
    _3zp9_ligand.load_from_files(_3zp9_ligand_pdb, _3zp9_ligand_pdbqt)

    ### 3bwf comes from PDBBind-CN
    _3bwf_protein = PDB()
    _3bwf_protein_pdb = os.path.join(data_dir(), "3bwf_protein_hyd.pdb")
    _3bwf_protein_pdbqt = os.path.join(data_dir(), "3bwf_protein_hyd.pdbqt")
    _3bwf_protein.load_from_files(_3bwf_protein_pdb, _3bwf_protein_pdbqt)
    # The ligand is also specified by pdbbind
    _3bwf_ligand = PDB()
    _3bwf_ligand_pdb = os.path.join(data_dir(), "3bwf_ligand_hyd.pdb")
    _3bwf_ligand_pdbqt = os.path.join(data_dir(), "3bwf_ligand_hyd.pdbqt")
    _3bwf_ligand.load_from_files(_3bwf_ligand_pdb, _3bwf_ligand_pdbqt)

    self.test_cases = [("3bwf", _3bwf_protein, _3bwf_ligand),
                       ("3zp9", _3zp9_protein, _3zp9_ligand)]

  def test_compute_hydrophobic(self):
    """
    TestBinana: Test that hydrophobic contacts are established.
    """
    hydrophobics_dict = {}
    for name, protein, ligand in self.test_cases:
      hydrophobics_dict[name] = compute_hydrophobic_contacts(
          ligand, protein)
    for name, hydrophobics in hydrophobics_dict.iteritems():
      print "Processing hydrohobics for %s" % name
      assert len(hydrophobics) == 6
      assert "BACKBONE_ALPHA" in hydrophobics
      assert "BACKBONE_BETA" in hydrophobics
      assert "BACKBONE_OTHER" in hydrophobics
      assert "SIDECHAIN_ALPHA" in hydrophobics
      assert "SIDECHAIN_BETA" in hydrophobics
      assert "SIDECHAIN_OTHER" in hydrophobics

  def test_compute_electrostatics(self):
    """
    TestBinana: Test that electrostatic energies are computed.
    """
    electrostatics_dict = {}
    for name, protein, ligand in self.test_cases:
      electrostatics_dict[name] = compute_electrostatic_energy(
          ligand, protein)
    for name, electrostatics in electrostatics_dict.iteritems():
      print "Processing electrostatics for %s" % name
      # The keys of these dicts are pairs of atomtypes, but the keys are
      # sorted so that ("C", "O") is always written as "C_O". Thus, for N
      # atom types, there are N*(N+1)/2 unique pairs.
      num_atoms = len(Binana.atom_types)
      assert len(electrostatics) == num_atoms*(num_atoms+1)/2
      # TODO(rbharath): Charges are not computed correctly for certain
      # ligands! (see 2y2h_ligand). Understand why this happens.
      #assert np.count_nonzero(np.array(electrostatics.values())) > 0

  def test_compute_flexibility(self):
    """
    TestBinana: Gather statistics about active site protein atoms.
    """
    active_site_dict = {}
    for name, protein, ligand in self.test_cases:
      active_site_dict[name] = compute_active_site_flexibility(
          ligand, protein)
    for name, active_site_flexibility in active_site_dict.iteritems():
      print "Processing active site flexibility for %s" % name
      assert len(active_site_flexibility.keys()) == 6
      assert "BACKBONE_ALPHA" in active_site_flexibility
      assert "BACKBONE_BETA" in active_site_flexibility
      assert "BACKBONE_OTHER" in active_site_flexibility
      assert "SIDECHAIN_ALPHA" in active_site_flexibility
      assert "SIDECHAIN_BETA" in active_site_flexibility
      assert "SIDECHAIN_OTHER" in active_site_flexibility

  def test_compute_hydrogen_bonds(self):
    """
    TestBinana: Compute the number of hydrogen bonds.

    TODO(rbharath): The hydrogen-bond angle cutoff seems like it's
    incorrect to me. The hydrogens are placed by openbabel and aren't
    optimized, so I'm pretty sure that this code will miss many hydrogen
    bonds.
    Here are some options:
    -) Find a method to optimize the hydrogen placement.
    -) Place a more permissive angle cutoff for hydrogens.
    -) Allow for "buckets": angles 0-20, 20-40, 40-60, etc. and count the
    number of hydrogen bonds in each bucket.
    """
    hbonds_dict = {}
    for name, protein, ligand in self.test_cases:
      hbonds_dict[name] = compute_hydrogen_bonds(
          ligand, protein)
    for name, hbonds in hbonds_dict.iteritems():
      print "Processing hydrogen bonds for %s" % name
      assert len(hbonds) == 12
      assert "HDONOR-LIGAND_BACKBONE_ALPHA" in hbonds
      assert "HDONOR-LIGAND_BACKBONE_BETA" in hbonds
      assert "HDONOR-LIGAND_BACKBONE_OTHER" in hbonds
      assert "HDONOR-LIGAND_SIDECHAIN_ALPHA" in hbonds
      assert "HDONOR-LIGAND_SIDECHAIN_BETA" in hbonds
      assert "HDONOR-LIGAND_SIDECHAIN_OTHER" in hbonds
      assert "HDONOR-RECEPTOR_BACKBONE_ALPHA" in hbonds
      assert "HDONOR-RECEPTOR_BACKBONE_BETA" in hbonds
      assert "HDONOR-RECEPTOR_BACKBONE_OTHER" in hbonds
      assert "HDONOR-RECEPTOR_SIDECHAIN_ALPHA" in hbonds
      assert "HDONOR-RECEPTOR_SIDECHAIN_BETA" in hbonds
      assert "HDONOR-RECEPTOR_SIDECHAIN_OTHER" in hbonds

  def test_compute_ligand_atom_counts(self):
    """
    TestBinana: Compute the Number of Ligand Atom Counts.
    """
    counts_dict = {}
    for name, _, ligand in self.test_cases:
      counts_dict[name] = compute_ligand_atom_counts(
          ligand)
    for name, counts in counts_dict.iteritems():
      print "Processing ligand atom counts for %s" % name
      # TODO(rbharath): This code is useful for debugging. Remove once
      # codebase is stable enough.
      #for key in Binana.atom_types:
      #  if key in counts:
      #    del counts[key]
      #print "Residual counts:"
      #print counts
      assert len(counts) == len(Binana.atom_types)

  def test_compute_contacts(self):
    """
    TestBinana: Compute contacts between Ligand and receptor.
    """
    contacts_dict = {}
    for name, protein, ligand in self.test_cases:
      contacts_dict[name] = compute_contacts(
          ligand, protein)
    num_atoms = len(Binana.atom_types)
    for name, (close_contacts, contacts) in contacts_dict.iteritems():
      print "Processing contacts for %s" % name
      print "close_contacts"
      for key, val in close_contacts.iteritems():
        if val != 0:
          print (key, val)
      print "len(close_contacts): " + str(len(close_contacts))
      print "contacts"
      for key, val in contacts.iteritems():
        if val != 0:
          print (key, val)
      print "len(contacts): " + str(len(contacts))
      print "Desired Number: " + str(num_atoms*(num_atoms+1)/2)
      # TODO(rbharath): The following code has proved very useful for
      # debugging. Remove once the code is stable enough that it's not
      # required.
      #if name == '1pi5':
      #  for first, second in itertools.product(Binana.atom_types,
      #    Binana.atom_types):
      #    key = "_".join(sorted([first, second]))
      #    if key in close_contacts:
      #      del close_contacts[key]
      #    if key in contacts:
      #      del contacts[key]
      #  print "Residuals close_contacts:"
      #  print close_contacts
      #  print "Residuals contacts:"
      #  print contacts
      assert len(close_contacts) == num_atoms*(num_atoms+1)/2
      assert len(contacts) == num_atoms*(num_atoms+1)/2

  def test_compute_pi_pi_stacking(self):
    """
    TestBinana: Compute Pi-Pi Stacking.
    """
    # 1zea is the only example that has any pi-stacking.
    pi_stacking_dict = {}
    for name, protein, ligand in self.test_cases:
      pi_stacking_dict[name] = compute_pi_pi_stacking(
          ligand, protein)
    for name, pi_stacking in pi_stacking_dict.iteritems():
      print "Processing pi-stacking for %s" % name
      assert len(pi_stacking) == 3
      print pi_stacking
      assert "STACKING_ALPHA" in pi_stacking
      assert "STACKING_BETA" in pi_stacking
      assert "STACKING_OTHER" in pi_stacking


  def test_compute_pi_t(self):
    """
    TestBinana: Compute Pi-T Interactions.

    TODO(rbharath): I believe that the imatininb-cabl complex has a pi-T
    interaction. This code has a bug since it reports that no such
    interaction is found.
    """
    pi_t_dict = {}
    for name, protein, ligand in self.test_cases:
      pi_t_dict[name] = compute_pi_t(
          ligand, protein)
    for name, pi_t in pi_t_dict.iteritems():
      print "Processing pi-T for %s" % name
      assert len(pi_t) == 3
      assert "T-SHAPED_ALPHA" in pi_t
      assert "T-SHAPED_BETA" in pi_t
      assert "T-SHAPED_OTHER" in pi_t

  def test_compute_pi_cation(self):
    """
    TestBinana: Compute Pi-Cation Interactions.
    """
    pi_cation_dict = {}
    for name, protein, ligand in self.test_cases:
      pi_cation_dict[name] = compute_pi_cation(
          ligand, protein)
    for name, pi_cation in pi_cation_dict.iteritems():
      print "Processing pi-cation for %s" % name
      assert len(pi_cation) == 6
      assert 'PI-CATION_LIGAND-CHARGED_ALPHA' in pi_cation
      assert 'PI-CATION_LIGAND-CHARGED_BETA' in pi_cation
      assert 'PI-CATION_LIGAND-CHARGED_OTHER' in pi_cation
      assert 'PI-CATION_RECEPTOR-CHARGED_ALPHA' in pi_cation
      assert 'PI-CATION_RECEPTOR-CHARGED_BETA' in pi_cation
      assert 'PI-CATION_RECEPTOR-CHARGED_OTHER' in pi_cation

  def test_compute_salt_bridges(self):
    """
    TestBinana: Compute Salt Bridges.

    TODO(bramsundar): None of the examples contain salt-bridge interactions. Find a
    complex with an actual salt-bridge interaction.
    """
    salt_bridges_dict = {}
    for name, protein, ligand in self.test_cases:
      salt_bridges_dict[name] = compute_salt_bridges(
          ligand, protein)
    for name, salt_bridges in salt_bridges_dict.iteritems():
      print "Processing salt-bridges for %s" % name
      assert len(salt_bridges) == 3
      print salt_bridges
      assert 'SALT-BRIDGE_ALPHA' in salt_bridges
      assert 'SALT-BRIDGE_BETA' in salt_bridges
      assert 'SALT-BRIDGE_OTHER' in salt_bridges

  def test_compute_input_vector(self):
    """
    TestBinana: Compute Input Vector.
    """
    features_dict = {}
    for name, protein, ligand in self.test_cases:
      features_dict[name] = self.binana.compute_input_vector(
          ligand, protein)
    num_atoms = len(Binana.atom_types)
    # Lengths:
    # ligand_receptor_close_contacts: N*(N+1)/2
    # ligand_receptor_contacts: N*(N+1)/2
    # ligand_receptor_electrostatics: N*(N+1)/2
    # ligand_atom_counts: N
    # hbonds: 12
    # hydrophobics: 6
    # stacking: 3
    # pi_cation: 6
    # t_shaped: 3
    # active_site_flexibility: 6
    # salt_bridges: 3
    # rotatable_boonds_count: 1
    total_len = (3*num_atoms*(num_atoms+1)/2 + num_atoms
                 + 12 + 6 + 3 + 6 + 3 + 6 + 3 + 1)
    for name, input_vector in features_dict.iteritems():
      print "Processing input-vector for %s" % name
      assert len(input_vector) == total_len
