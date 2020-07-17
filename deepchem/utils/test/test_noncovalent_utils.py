class TestPiInteractions(unittest.TestCase):

  def setUp(self):
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # simple flat ring
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdDepictor import Compute2DCoords
    self.cycle4 = MolFromSmiles('C1CCC1')
    #self.cycle4.Compute2DCoords()
    Compute2DCoords(self.cycle4)

    # load and sanitize two real molecules
    _, self.prot = rdkit_util.load_molecule(
        os.path.join(current_dir, '../../feat/tests/3ws9_protein_fixer_rdkit.pdb'),
        add_hydrogens=False,
        calc_charges=False,
        sanitize=True)

    _, self.lig = rdkit_util.load_molecule(
        os.path.join(current_dir, '../../feat/tests/3ws9_ligand.sdf'),
        add_hydrogens=False,
        calc_charges=False,
        sanitize=True)

  def test_compute_ring_center(self):
    self.assertTrue(
        np.allclose(rdkit_util.compute_ring_center(self.cycle4, range(4)), 0))

  def test_compute_ring_normal(self):
    normal = rdkit_util.compute_ring_normal(self.cycle4, range(4))
    self.assertTrue(
        np.allclose(np.abs(normal / np.linalg.norm(normal)), [0, 0, 1]))

  def test_is_pi_parallel(self):
    ring1_center = np.array([0.0, 0.0, 0.0])
    ring2_center_true = np.array([4.0, 0.0, 0.0])
    ring2_center_false = np.array([10.0, 0.0, 0.0])
    ring1_normal_true = np.array([1.0, 0.0, 0.0])
    ring1_normal_false = np.array([0.0, 1.0, 0.0])

    for ring2_normal in (np.array([2.0, 0, 0]), np.array([-3.0, 0, 0])):
      # parallel normals
      self.assertTrue(
          rdkit_util.is_pi_parallel(ring1_center,
                                    ring1_normal_true,
                                    ring2_center_true,
                                    ring2_normal))
      # perpendicular normals
      self.assertFalse(
          rdkit_util.is_pi_parallel(ring1_center,
                                    ring1_normal_false,
                                    ring2_center_true,
                                    ring2_normal))
      # too far away
      self.assertFalse(
          rdkit_util.is_pi_parallel(ring1_center,
                                    ring1_normal_true,
                                    ring2_center_false,
                                    ring2_normal))

  def test_is_pi_t(self):
    ring1_center = np.array([0.0, 0.0, 0.0])
    ring2_center_true = np.array([4.0, 0.0, 0.0])
    ring2_center_false = np.array([10.0, 0.0, 0.0])
    ring1_normal_true = np.array([0.0, 1.0, 0.0])
    ring1_normal_false = np.array([1.0, 0.0, 0.0])

    for ring2_normal in (np.array([2.0, 0, 0]), np.array([-3.0, 0, 0])):
      # perpendicular normals
      self.assertTrue(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_true, ring2_center_true,
                      ring2_normal))
      # parallel normals
      self.assertFalse(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_false, ring2_center_true,
                      ring2_normal))
      # too far away
      self.assertFalse(
          rdkit_util.is_pi_t(ring1_center, ring1_normal_true, ring2_center_false,
                      ring2_normal))

  def test_compute_pi_stack(self):
    # order of the molecules shouldn't matter
    dicts1 = rdkit_util.compute_pi_stack(self.prot, self.lig)
    dicts2 = rdkit_util.compute_pi_stack(self.lig, self.prot)
    for i, j in ((0, 2), (1, 3)):
      self.assertEqual(dicts1[i], dicts2[j])
      self.assertEqual(dicts1[j], dicts2[i])

    # with this criteria we should find both types of stacking
    for d in rdkit_util.compute_pi_stack(
        self.lig, self.prot, dist_cutoff=7, angle_cutoff=40.):
      self.assertGreater(len(d), 0)

  def test_is_cation_pi(self):
    cation_position = np.array([[2.0, 0.0, 0.0]])
    ring_center_true = np.array([4.0, 0.0, 0.0])
    ring_center_false = np.array([10.0, 0.0, 0.0])
    ring_normal_true = np.array([1.0, 0.0, 0.0])
    ring_normal_false = np.array([0.0, 1.0, 0.0])

    # parallel normals
    self.assertTrue(
        rdkit_util.is_cation_pi(cation_position, ring_center_true, ring_normal_true))
    # perpendicular normals
    self.assertFalse(
        rdkit_util.is_cation_pi(cation_position, ring_center_true, ring_normal_false))
    # too far away
    self.assertFalse(
        rdkit_util.is_cation_pi(cation_position, ring_center_false, ring_normal_true))

  def test_compute_cation_pi(self):
    # TODO(rbharath): find better example, currently dicts are empty
    dicts1 = rdkit_util.compute_cation_pi(self.prot, self.lig)
    dicts2 = rdkit_util.compute_cation_pi(self.lig, self.prot)

  def test_compute_binding_pocket_cation_pi(self):
    # TODO find better example, currently dicts are empty
    prot_dict, lig_dict = rdkit_util.compute_binding_pocket_cation_pi(
        self.prot, self.lig)

    exp_prot_dict, exp_lig_dict = rdkit_util.compute_cation_pi(self.prot, self.lig)
    add_lig, add_prot = rdkit_util.compute_cation_pi(self.lig, self.prot)
    for exp_dict, to_add in ((exp_prot_dict, add_prot), (exp_lig_dict,
                                                         add_lig)):
      for atom_idx, count in to_add.items():
        if atom_idx not in exp_dict:
          exp_dict[atom_idx] = count
        else:
          exp_dict[atom_idx] += count

    self.assertEqual(prot_dict, exp_prot_dict)
    self.assertEqual(lig_dict, exp_lig_dict)
