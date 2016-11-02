"""
Protein-ligand noncovalent chemistry descriptors
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import math
import re
import itertools
import tempfile
import shutil
import numpy as np
from deepchem.feat import ComplexFeaturizer
from deepchem.feat.nnscore_pdb import PDB
from deepchem.feat.nnscore_utils import Point
from deepchem.feat.nnscore_utils import angle_between_points
from deepchem.feat.nnscore_utils import angle_between_three_points
from deepchem.feat.nnscore_utils import project_point_onto_plane
from deepchem.feat.nnscore_utils import hydrogenate_and_compute_partial_charges

__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "GPL"

#ELECTROSTATIC_JOULE_PER_MOL = 138.94238460104697e4 # units?
# This is just a scaling factor, so it's set so as to keep the network
# inputs roughly contained in 0-1
ELECTROSTATIC_JOULE_PER_MOL = 10.0
# O-H distance is 0.96 A, N-H is 1.01 A. See
# http://www.science.uwaterloo.ca/~cchieh/cact/c120/bondel.html
H_BOND_DIST = 1.3 # angstroms
H_BOND_ANGLE = 40 # degrees
# If atoms are < 2.5 A apart, we count it as a close contact
CLOSE_CONTACT_CUTOFF = 2.5
# If receptor and ligand atoms are > 4 A apart, we consider them
# unable to interact with simple electrostatics.
CONTACT_CUTOFF = 4 # angstroms
# "PI-Stacking Interactions ALIVE AND WELL IN PROTEINS" says
# distance of 7.5 A is good cutoff. This seems really big to me,
# except that pi-pi interactions (parallel) are actually usually
# off centered. Interesting paper.  Note that adenine and
# tryptophan count as two aromatic rings. So, for example, an
# interaction between these two, if positioned correctly, could
# count for 4 pi-pi interactions.
PI_PI_CUTOFF = 7.5
# Cation-pi interaction cutoff based on
# "Cation-pi interactions in structural biology."
CATION_PI_CUTOFF = 6.0
# 4  is good cutoff for salt bridges according to
# "Close-Range Electrostatic Interactions in Proteins",
# but looking at complexes, I decided to go with 5.5 A
SALT_BRIDGE_CUTOFF = 5.5
# This is perhaps controversial. I noticed that often a pi-cation
# interaction or other pi interaction was only slightly off, but
# looking at the structure, it was clearly supposed to be a pi-cation
# interaction. I've decided then to artificially expand the radius of
# each pi ring. Think of this as adding in a VDW radius, or
# accounting for poor crystal-structure resolution, or whatever you
# want to justify it.
PI_PADDING = 0.75


def hashtable_entry_add_one(hashtable, key, toadd=1):
  """Increments hashtable entry if exists, else creates entry."""
  # note that dictionaries (hashtables) are passed by reference in python
  if key in hashtable:
    hashtable[key] = hashtable[key] + toadd
  else:
    hashtable[key] = toadd

def clean_atomtype(atomtype):
  """Removes extraneous charge info from atomtype

  Atomtypes occasionally have charges such as O1+ or N1-. This function
  uses regexps to replace these out.

  atomtype: String
    Raw atomtype extracted from PDB
  """
  return re.sub(r'[0-9]+[+-]?', r'', atomtype)

def compute_hydrophobic_contacts(ligand, receptor):
  """
  Compute possible hydrophobic contacts between ligand and atom.

  Returns a dictionary whose keys are atompairs of type
  "${RESIDUETYPE}_${RECEPTOR_ATOM}" where RESIDUETYPE is either "SIDECHAIN" or
  "BACKBONE" and RECEPTOR_ATOM is "O" or "C" or etc. The
  values count the number of hydrophobic contacts.

  Parameters
  ----------
  ligand: PDB
    A PDB Object describing the ligand molecule.
  receptor: PDB
    A PDB object describing the receptor protein.

  """
  # Now see if there's hydrophobic contacts (C-C contacts)
  hydrophobics = {
      'BACKBONE_ALPHA': 0, 'BACKBONE_BETA': 0, 'BACKBONE_OTHER': 0,
      'SIDECHAIN_ALPHA': 0, 'SIDECHAIN_BETA': 0, 'SIDECHAIN_OTHER': 0}
  for ligand_index in ligand.all_atoms:
    ligand_atom = ligand.all_atoms[ligand_index]
    for receptor_index in receptor.all_atoms:
      receptor_atom = receptor.all_atoms[receptor_index]
      dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
      if dist < CONTACT_CUTOFF:
        if ligand_atom.element == "C" and receptor_atom.element == "C":
          hydrophobic_key = (
              receptor_atom.side_chain_or_backbone() +
              "_" + receptor_atom.structure)
          hashtable_entry_add_one(hydrophobics, hydrophobic_key)
  return hydrophobics

def compute_electrostatic_energy(ligand, receptor):
  """
  Compute electrostatic energy between ligand and atom.

  Returns a dictionary whose keys are atompairs of type
  "${ATOMTYPE}_${ATOMTYPE}". The ATOMTYPE terms can equal "C", "O",
  etc. One ATOMTYPE belongs to receptor, the other to ligand, but
  information on which is which isn't preserved
  (i.e., C-receptor, O-ligand and C-ligand, O-receptor generate the same
  key). The values are the sum of associated coulomb energies for such
  pairs (i.e., if there are three C_O interactions with energies 1, 2,
  and 3 respectively, the total energy is 6).

  Parameters
  ----------
  ligand: PDB
    A PDB Object describing the ligand molecule.
  receptor: PDB
    A PDB object describing the receptor protein.
  """
  electrostatics = {}
  for first, second in itertools.product(
      Binana.atom_types, Binana.atom_types):
    key = "_".join(sorted([first, second]))
    electrostatics[key] = 0
  for ligand_index in ligand.all_atoms:
    ligand_atom = ligand.all_atoms[ligand_index]
    for receptor_index in receptor.all_atoms:
      receptor_atom = receptor.all_atoms[receptor_index]
      atomtypes = [clean_atomtype(atom) for atom in
                   [ligand_atom.atomtype, receptor_atom.atomtype]]
      key = "_".join(sorted(atomtypes))
      dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
      if dist < CONTACT_CUTOFF:
        ligand_charge = ligand_atom.charge
        receptor_charge = receptor_atom.charge
        # to convert into J/mol; might be nice to double check this
        # TODO(bramsundar): What are units of
        # ligand_charge/receptor_charge?
        coulomb_energy = ((ligand_charge * receptor_charge / dist)
                          * ELECTROSTATIC_JOULE_PER_MOL)
        hashtable_entry_add_one(
            electrostatics, key, coulomb_energy)
  return electrostatics


def compute_ligand_atom_counts(ligand):
  """Counts atoms of each type in given ligand.

  Returns a dictionary that maps atom types ("C", "O", etc.) to
  counts.

  Parameters
  ----------
  ligand: PDB Object
    Stores ligand information.

  Returns
  -------
  ligand_atom_types: dictionary
    Keys are atom types; values are integer counts.
  """
  ligand_atom_types = {}
  for atom_type in Binana.atom_types:
    ligand_atom_types[atom_type] = 0
  for ligand_index in ligand.all_atoms:
    hashtable_entry_add_one(
        ligand_atom_types,
        clean_atomtype(ligand.all_atoms[ligand_index].atomtype))
  return ligand_atom_types

def compute_active_site_flexibility(ligand, receptor):
  """
  Compute statistics to judge active-site flexibility

  Returns a dictionary whose keys are of type
  "${RESIDUETYPE}_${STRUCTURE}" where RESIDUETYPE is either "SIDECHAIN"
  or "BACKBONE" and STRUCTURE is either ALPHA, BETA, or OTHER and
  corresponds to the protein secondary structure of the current residue.

  Parameters
  ----------
  ligand: PDB
    A PDB Object describing the ligand molecule.
  receptor: PDB
    A PDB object describing the receptor protein.

  """
  active_site_flexibility = {
      'BACKBONE_ALPHA': 0, 'BACKBONE_BETA': 0, 'BACKBONE_OTHER': 0,
      'SIDECHAIN_ALPHA': 0, 'SIDECHAIN_BETA': 0, 'SIDECHAIN_OTHER': 0}
  for ligand_index in ligand.all_atoms:
    ligand_atom = ligand.all_atoms[ligand_index]
    for receptor_index in receptor.all_atoms:
      receptor_atom = receptor.all_atoms[receptor_index]
      dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
      if dist < CONTACT_CUTOFF:
        flexibility_key = (receptor_atom.side_chain_or_backbone() + "_"
                           + receptor_atom.structure)
        hashtable_entry_add_one(active_site_flexibility, flexibility_key)
  return active_site_flexibility


def compute_pi_t(ligand, receptor):
  """
  Computes T-shaped pi-pi interactions.

  Returns a dictionary with keys of form T-SHAPED_${STRUCTURE} where
  STRUCTURE is "ALPHA" or "BETA" or "OTHER". Values are counts of the
  number of such stacking interactions.

  Parameters
  ----------
  ligand: PDB Object.
    small molecule to dock.
  receptor: PDB Object
    protein to dock agains.
  """
  pi_t = {'T-SHAPED_ALPHA': 0, 'T-SHAPED_BETA': 0, 'T-SHAPED_OTHER': 0}
  for lig_aromatic in ligand.aromatic_rings:
    for rec_aromatic in receptor.aromatic_rings:
      lig_aromatic_norm_vector = Point(
          coords=np.array([lig_aromatic.plane_coeff[0],
                           lig_aromatic.plane_coeff[1],
                           lig_aromatic.plane_coeff[2]]))
      rec_aromatic_norm_vector = Point(
          coords=np.array([rec_aromatic.plane_coeff[0],
                           rec_aromatic.plane_coeff[1],
                           rec_aromatic.plane_coeff[2]]))
      angle_between_planes = (
          angle_between_points(
              lig_aromatic_norm_vector, rec_aromatic_norm_vector)
          * 180.0/math.pi)
      if (math.fabs(angle_between_planes-90) < 30.0
          or math.fabs(angle_between_planes-270) < 30.0):
        # so they're more or less perpendicular, it's probably a
        # pi-edge interaction having looked at many structures, I
        # noticed the algorithm was identifying T-pi reactions
        # when the two rings were in fact quite distant, often
        # with other atoms in between. Eye-balling it, requiring
        # that at their closest they be at least 5 A apart seems
        # to separate the good T's from the bad
        min_dist = 100.0
        for ligand_ind in lig_aromatic.indices:
          ligand_at = ligand.all_atoms[ligand_ind]
          for receptor_ind in rec_aromatic.indices:
            receptor_at = receptor.all_atoms[receptor_ind]
            dist = ligand_at.coordinates.dist_to(receptor_at.coordinates)
            if dist < min_dist:
              min_dist = dist

        if min_dist <= 5.0:
          # so at their closest points, the two rings come within
          # 5 A of each other.

          # okay, is the ligand pi pointing into the receptor
          # pi, or the other way around?  first, project the
          # center of the ligand pi onto the plane of the
          # receptor pi, and vs. versa

          # This could be directional somehow, like a hydrogen
          # bond.

          pt_on_receptor_plane = project_point_onto_plane(
              lig_aromatic.center, rec_aromatic.plane_coeff)
          pt_on_ligand_plane = project_point_onto_plane(
              rec_aromatic.center, lig_aromatic.plane_coeff)

          # now, if it's a true pi-T interaction, this projected
          # point should fall within the ring whose plane it's
          # been projected into.
          if ((pt_on_receptor_plane.dist_to(rec_aromatic.center)
               <= rec_aromatic.radius + PI_PADDING) or
              (pt_on_ligand_plane.dist_to(lig_aromatic.center)
               <= lig_aromatic.radius + PI_PADDING)):

            # so it is in the ring on the projected plane.
            structure = receptor.all_atoms[rec_aromatic.indices[0]].structure
            if structure == "":
              # since it could be interacting with a cofactor or something
              structure = "OTHER"
            key = "T-SHAPED_" + structure

            hashtable_entry_add_one(pi_t, key)
  return pi_t

def compute_hydrogen_bonds(ligand, receptor):
  """
  Computes hydrogen bonds between ligand and receptor.

  Returns a dictionary whose keys are of form
  HDONOR-${MOLTYPE}_${RESIDUETYPE}_${STRUCTURE} where MOLTYPE is either
  "RECEPTOR" or "LIGAND", RESIDUETYPE is "BACKBONE" or "SIDECHAIN" and
  where STRUCTURE is "ALPHA" or "BETA" or "OTHER". The values are counts
  of the numbers of hydrogen bonds associated with the given keys.

  Parameters
  ----------
  ligand: PDB
    A PDB Object describing the ligand molecule.
  receptor: PDB
    A PDB object describing the receptor protein.
  """
  hbonds = {
      'HDONOR-LIGAND_BACKBONE_ALPHA': 0,
      'HDONOR-LIGAND_BACKBONE_BETA': 0,
      'HDONOR-LIGAND_BACKBONE_OTHER': 0,
      'HDONOR-LIGAND_SIDECHAIN_ALPHA': 0,
      'HDONOR-LIGAND_SIDECHAIN_BETA': 0,
      'HDONOR-LIGAND_SIDECHAIN_OTHER': 0,
      'HDONOR-RECEPTOR_BACKBONE_ALPHA': 0,
      'HDONOR-RECEPTOR_BACKBONE_BETA': 0,
      'HDONOR-RECEPTOR_BACKBONE_OTHER': 0,
      'HDONOR-RECEPTOR_SIDECHAIN_ALPHA': 0,
      'HDONOR-RECEPTOR_SIDECHAIN_BETA': 0,
      'HDONOR-RECEPTOR_SIDECHAIN_OTHER': 0}
  for ligand_index in ligand.all_atoms:
    ligand_atom = ligand.all_atoms[ligand_index]
    for receptor_index in receptor.all_atoms:
      receptor_atom = receptor.all_atoms[receptor_index]
      # Now see if there's some sort of hydrogen bond between
      # these two atoms. distance cutoff = H_BOND_DIST, angle cutoff =
      # H_BOND_ANGLE.
      dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
      if dist < CONTACT_CUTOFF:
        electronegative_atoms = ["O", "N", "F"]
        if ((ligand_atom.element in electronegative_atoms)
            and (receptor_atom.element in electronegative_atoms)):

          hydrogens = []
          # TODO(rbharath): This is a horrible inner-loop search. Can
          # this be made more efficient?
          for atm_index in ligand.all_atoms:
            atom = ligand.all_atoms[atm_index]
            if atom.element == "H":
              # Make sure to set comment (used below)
              atom.comment = "LIGAND"
              if (atom.coordinates.dist_to(ligand_atom.coordinates)
                  < H_BOND_DIST):
                hydrogens.append(atom)

          for atm_index in receptor.all_atoms:
            atom = receptor.all_atoms[atm_index]
            if atom.element == "H":
              # Make sure to set comment (used below)
              atom.comment = "RECEPTOR"
              if (atom.coordinates.dist_to(receptor_atom.coordinates)
                  < H_BOND_DIST):
                hydrogens.append(atom)

          # now we need to check the angles
          # TODO(rbharath): Rather than using this heuristic, it seems like
          # it might be better to just report the angle in the feature
          # vector...
          for hydrogen in hydrogens:
            angle = math.fabs(180 - angle_between_three_points(
                ligand_atom.coordinates, hydrogen.coordinates,
                receptor_atom.coordinates) * 180.0 / math.pi)
            if angle <= H_BOND_ANGLE:
              hbonds_key = (
                  "HDONOR-" + hydrogen.comment + "_" +
                  receptor_atom.side_chain_or_backbone() + "_" +
                  receptor_atom.structure)
              hashtable_entry_add_one(hbonds, hbonds_key)
  return hbonds

def compute_pi_pi_stacking(ligand, receptor):
  """
  Computes pi-pi interactions.

  Returns a dictionary with keys of form STACKING_${STRUCTURE} where
  STRUCTURE is "ALPHA" or "BETA" or "OTHER". Values are counts of the
  number of such stacking interactions.

  Parameters
  ----------
  ligand: PDB Object.
    small molecule to dock.
  receptor: PDB Object
    protein to dock agains.
  """
  pi_stacking = {'STACKING_ALPHA': 0, 'STACKING_BETA': 0, 'STACKING_OTHER': 0}
  for lig_aromatic in ligand.aromatic_rings:
    for rec_aromatic in receptor.aromatic_rings:
      dist = lig_aromatic.center.dist_to(rec_aromatic.center)
      if dist < PI_PI_CUTOFF:
        # so there could be some pi-pi interactions.  Now, let's
        # check for stacking interactions. Are the two pi's roughly
        # parallel?
        lig_aromatic_norm_vector = Point(
            coords=np.array([lig_aromatic.plane_coeff[0],
                             lig_aromatic.plane_coeff[1],
                             lig_aromatic.plane_coeff[2]]))
        rec_aromatic_norm_vector = Point(
            coords=np.array([rec_aromatic.plane_coeff[0],
                             rec_aromatic.plane_coeff[1],
                             rec_aromatic.plane_coeff[2]]))
        angle_between_planes = (
            angle_between_points(
                lig_aromatic_norm_vector, rec_aromatic_norm_vector)
            * 180.0/math.pi)

        if (math.fabs(angle_between_planes-0) < 30.0
            or math.fabs(angle_between_planes-180) < 30.0):
          # so they're more or less parallel, it's probably pi-pi
          # stacking now, since pi-pi are not usually right on
          # top of each other. They're often staggered. So I don't
          # want to just look at the centers of the rings and
          # compare. Let's look at each of the atoms.  do atom of
          # the atoms of one ring, when projected onto the plane of
          # the other, fall within that other ring?

          # start by assuming it's not a pi-pi stacking interaction
          pi_pi = False
          for ligand_ring_index in lig_aromatic.indices:
            # project the ligand atom onto the plane of the receptor ring
            pt_on_receptor_plane = project_point_onto_plane(
                ligand.all_atoms[ligand_ring_index].coordinates,
                rec_aromatic.plane_coeff)
            if (pt_on_receptor_plane.dist_to(rec_aromatic.center)
                <= rec_aromatic.radius + PI_PADDING):
              pi_pi = True
              break

          # TODO(rbharath): This if-else is confusing.
          if pi_pi == False:
            for receptor_ring_index in rec_aromatic.indices:
              # project the ligand atom onto the plane of the receptor ring
              pt_on_ligand_plane = project_point_onto_plane(
                  receptor.all_atoms[receptor_ring_index].coordinates,
                  lig_aromatic.plane_coeff)
              if (pt_on_ligand_plane.dist_to(lig_aromatic.center)
                  <= lig_aromatic.radius + PI_PADDING):
                pi_pi = True
                break

          if pi_pi == True:
            structure = receptor.all_atoms[rec_aromatic.indices[0]].structure
            if structure == "":
              # since it could be interacting with a cofactor or something
              structure = "OTHER"
            key = "STACKING_" + structure
            hashtable_entry_add_one(pi_stacking, key)
  return pi_stacking

def compute_pi_cation(ligand, receptor):
  """
  Computes number of pi-cation interactions.

  Returns a dictionary whose keys are of form
  ${MOLTYPE}-CHARGED_${STRUCTURE} where MOLTYPE is either "LIGAND" or
  "RECEPTOR" and STRUCTURE is "ALPHA" or "BETA" or "OTHER".

  Parameters
  ----------
  ligand: PDB Object
    small molecule to dock.
  receptor: PDB Object
    protein to dock agains.
  """
  pi_cation = {
      'PI-CATION_LIGAND-CHARGED_ALPHA': 0,
      'PI-CATION_LIGAND-CHARGED_BETA': 0,
      'PI-CATION_LIGAND-CHARGED_OTHER': 0,
      'PI-CATION_RECEPTOR-CHARGED_ALPHA': 0,
      'PI-CATION_RECEPTOR-CHARGED_BETA': 0,
      'PI-CATION_RECEPTOR-CHARGED_OTHER': 0}
  for aromatic in receptor.aromatic_rings:
    for charged in ligand.charges:
      if charged.positive == True: # so only consider positive charges
        if charged.coordinates.dist_to(aromatic.center) < CATION_PI_CUTOFF:

          # project the charged onto the plane of the aromatic
          charge_projected = project_point_onto_plane(
              charged.coordinates, aromatic.plane_coeff)
          if (charge_projected.dist_to(aromatic.center)
              < aromatic.radius + PI_PADDING):
            structure = receptor.all_atoms[aromatic.indices[0]].structure
            if structure == "":
              # since it could be interacting with a cofactor or something
              structure = "OTHER"
            key = "PI-CATION_LIGAND-CHARGED_" + structure

            hashtable_entry_add_one(pi_cation, key)

  for aromatic in ligand.aromatic_rings:
    # now it's the ligand that has the aromatic group
    for charged in receptor.charges:
      if charged.positive: # so only consider positive charges
        if charged.coordinates.dist_to(aromatic.center) < CATION_PI_CUTOFF:
          charge_projected = project_point_onto_plane(
              charged.coordinates, aromatic.plane_coeff)
          if (charge_projected.dist_to(aromatic.center)
              < aromatic.radius + PI_PADDING):
            structure = receptor.all_atoms[charged.indices[0]].structure
            if structure == "":
              # since it could be interacting with a cofactor or something
              structure = "OTHER"
            key = "PI-CATION_RECEPTOR-CHARGED_" + structure

            hashtable_entry_add_one(pi_cation, key)
  return pi_cation

def compute_contacts(ligand, receptor):
  """Compute distance measurements for ligand-receptor atom pairs.

  Returns two dictionaries, each of whose keys are of form
  ATOMTYPE_ATOMTYPE.

  Parameters
  ----------
  ligand: PDB object
    Should be loaded with the ligand in question.
  receptor: PDB object.
    Should be loaded with the receptor in question.
  """
  ligand_receptor_contacts, ligand_receptor_close_contacts = {}, {}
  for first, second in itertools.product(
      Binana.atom_types, Binana.atom_types):
    key = "_".join(sorted([first, second]))
    ligand_receptor_contacts[key] = 0
    ligand_receptor_close_contacts[key] = 0
  for ligand_index in ligand.all_atoms:
    for receptor_index in receptor.all_atoms:
      ligand_atom = ligand.all_atoms[ligand_index]
      receptor_atom = receptor.all_atoms[receptor_index]

      dist = ligand_atom.coordinates.dist_to(receptor_atom.coordinates)
      key = "_".join(
          sorted([clean_atomtype(atom) for atom in
                  [ligand_atom.atomtype, receptor_atom.atomtype]]))
      if dist < CONTACT_CUTOFF:
        hashtable_entry_add_one(
            ligand_receptor_contacts, key)
      if dist < CLOSE_CONTACT_CUTOFF:
        hashtable_entry_add_one(
            ligand_receptor_close_contacts, key)
  return ligand_receptor_close_contacts, ligand_receptor_contacts

def compute_salt_bridges(ligand, receptor):
  """
  Computes number of ligand-receptor salt bridges.

  Returns a dictionary with keys of form SALT-BRIDGE_${STRUCTURE} where
  STRUCTURE is "ALPHA" or "BETA" or "OTHER."

  Parameters
  ----------
  ligand: PDB Object
    small molecule to dock.
  receptor: PDB Object
    protein to dock agains.
  """
  salt_bridges = {'SALT-BRIDGE_ALPHA': 0, 'SALT-BRIDGE_BETA': 0,
                  'SALT-BRIDGE_OTHER': 0}
  for receptor_charge in receptor.charges:
    for ligand_charge in ligand.charges:
      if ligand_charge.positive != receptor_charge.positive:
        # so they have oppositve charges
        if (ligand_charge.coordinates.dist_to(
            receptor_charge.coordinates) < SALT_BRIDGE_CUTOFF):
          structure = receptor.all_atoms[receptor_charge.indices[0]].structure
          key = "SALT-BRIDGE_" + structure
          hashtable_entry_add_one(salt_bridges, key)
  return salt_bridges


class Binana:
  """
  Binana extracts a fingerprint from a provided binding pose.

  TODO(rbharath): Write a function that extracts the binding-site residues
  and their numbers. This will prove useful when debugging the fingerprint
  for correct binding-pocket interactions.

  TODO(rbharath): Write a function that lists charged groups in
  binding-site residues.

  TODO(rbharath): Write a function that aromatic groups in
  binding-site residues.

  TODO(rbharath) Write a function that lists charged groups in ligand.

  TODO(rbharath): Write a function that lists aromatic groups in ligand.

  The Binana feature vector transforms a ligand-receptor binding pose
  into a feature vector. The feature vector has the following
  components:

    -vina_output: Components of vina's score function.
    -ligand_receptor_contacts: List of contacts between ligand and
       receptor atoms (< 4 A)
    -ligand_receptor_electrostatics: Coulomb energy between contacting
       ligand and receptor atoms.
    -ligand_atom_counts: The atom types in the ligand.
    -ligand_receptor_close_contacts: List of close contacts between
       ligand and receptor (< 2.5 A)
    -hbonds: List of hydrogen bonds.
    -hydrophobic: List of hydrophobic contacts.
    -stacking: List of pi-pi stacking.
    -pi_cation: List of pi-cation interactions.
    -t_shaped: List of T-shaped interactions.
      The pi-cloud concentrates negative charge, leaving the edges of the
      aromatic ring with some positive charge. Hence, T-shaped interactions
      align the positive exterior of one ring with the negative interior of
      another. See wikipedia for details.
    -active_site_flexibility: Considers whether the receptor atoms are
       backbone or sidechain and whether they are part of
       alpha-helices or beta-sheets.
    -salt_bridges: List of salt-bridges between ligand and receptor.
    -rotatable_bonds_count: Count of (ligand(?), receptor(?))
       rotatable bonds.
  """
  # TODO(rbharath): What is atom type A here?
  atom_types = [
      "A", "AL", "AS", "B", "BE", "BR", "C", "CA", "CD", "CO", "CL", "CU",
      "F", "FE", "H", "HG", "HD", "I", "IR", "MG", "MN", "N", "NA", "NI",
      "O", "OA", "OS", "P", "PT", "RE", "RH", "RU", "S", "SA", "SE", "SI",
      "SR", "V", "ZN"]

  @staticmethod
  def num_features():
    """Returns the length of Binana's feature vectors."""
    num_atoms = len(Binana.atom_types)
    feature_len = (
        3*num_atoms*(num_atoms+1)/2 + num_atoms
        + 12 + 6 + 3 + 6 + 3 + 6 + 3 + 1)
    return feature_len

  def compute_input_vector_from_files(
      self, ligand_pdb_filename, receptor_pdb_filename, line_header):
    """Computes feature vector for ligand-receptor pair.

    Parameters
    ----------
    ligand_pdb_filename: string
      path to ligand's pdb file.
    receptor_pdb_filename: string
      path to receptor pdb file.
    line_header: string
      line separator in PDB files
    """
    # Load receptor and ligand from file.
    receptor = PDB()
    receptor.load_from_files(receptor_pdb_filename, line_header)
    receptor.assign_secondary_structure()
    ligand = PDB()
    ligand.load_from_files(ligand_pdb_filename, line_header)
    self.compute_input_vector(ligand, receptor)

  def compute_input_vector(self, ligand, receptor):
    """Computes feature vector for ligand-receptor pair.

    Parameters
    ----------
    ligand: PDB object
      Contains loaded ligand.
    receptor: PDB object
      Contains loaded receptor.
    """

    rotatable_bonds_count = {'rot_bonds': ligand.rotatable_bonds_count}
    ligand_receptor_close_contacts, ligand_receptor_contacts = (
        compute_contacts(ligand, receptor))
    ligand_receptor_electrostatics = (
        compute_electrostatic_energy(ligand, receptor))
    ligand_atom_counts = compute_ligand_atom_counts(ligand)
    hbonds = compute_hydrogen_bonds(ligand, receptor)
    hydrophobics = compute_hydrophobic_contacts(ligand, receptor)
    stacking = compute_pi_pi_stacking(ligand, receptor)
    pi_cation = compute_pi_cation(ligand, receptor)
    t_shaped = compute_pi_t(ligand, receptor)
    active_site_flexibility = (
        compute_active_site_flexibility(ligand, receptor))
    salt_bridges = compute_salt_bridges(ligand, receptor)

    input_vector = []
    for features in [ligand_receptor_contacts,
                     ligand_receptor_electrostatics, ligand_atom_counts,
                     ligand_receptor_close_contacts, hbonds, hydrophobics,
                     stacking, pi_cation, t_shaped,
                     active_site_flexibility, salt_bridges,
                     rotatable_bonds_count]:
      for key in sorted(features.keys()):
        input_vector.append(features[key])
    if len(input_vector) != Binana.num_features():
      raise ValueError("Feature length incorrect.")
    return input_vector

class NNScoreComplexFeaturizer(ComplexFeaturizer):
  """
  Compute NNScore fingerprints for complexes.
  """

  def __init__(self):
    self.binana = Binana()

  def _featurize_complex(self, mol_pdb, protein_pdb):
    """
    Compute Binana fingerprint for complex.
    """
    print("In _featurize_complex")
    mol_pdb_file = tempfile.NamedTemporaryFile(suffix="pdb")
    with open(mol_pdb_file.name, "w") as mol_f:
      mol_f.writelines(mol_pdb)
    protein_pdb_file = tempfile.NamedTemporaryFile(suffix="pdb")
    with open(protein_pdb_file.name, "w") as protein_f:
      protein_f.writelines(protein_pdb)
    print("Written temp pdb files")

    mol_hyd_file = tempfile.NamedTemporaryFile(suffix="pdb")
    mol_pdbqt_file = tempfile.NamedTemporaryFile(suffix="pdbqt")
    hydrogenate_and_compute_partial_charges(
        mol_pdb_file.name, "pdb", mol_hyd_file.name,
        mol_pdbqt_file.name)
    print("Hydrogenated mol file")

    protein_hyd_file = tempfile.NamedTemporaryFile(suffix="pdb")
    protein_pdbqt_file = tempfile.NamedTemporaryFile(suffix="pdbqt")
    hydrogenate_and_compute_partial_charges(
        protein_pdb_file.name, "pdb", protein_hyd_file.name,
        protein_pdbqt_file.name)
    print("Hydrogenated protein file")

    mol_pdb_obj = PDB()
    mol_pdb_obj.load_from_files(mol_pdb_file.name, mol_pdbqt_file.name)
    print("Loaded mol pdb object")

    protein_pdb_obj = PDB()
    protein_pdb_obj.load_from_files(
        protein_pdb_file.name, protein_pdbqt_file.name)
    print("Loaded protein pdb object")

    features = self.binana.compute_input_vector(mol_pdb_obj, protein_pdb_obj)
    print("Computed binana features.")

    return features
