# QM7 Examples

QM7 is a subset of GDB-13 (a database of nearly 1 billion
stable and synthetically accessible organic molecules)
containing up to 7 heavy atoms C, N, O, and S. The 3D
Cartesian coordinates of the most stable conformations and
their atomization energies were determined using ab-initio
density functional theory (PBE0/tier2 basis set).This dataset
also provided Coulomb matrices as calculated in [Rupp et al.
PRL, 2012]:

- C_ii = 0.5 * Z^2.4
- C_ij = Z_i * Z_j/abs(R_i − R_j)
- Z_i - nuclear charge of atom i
- R_i - cartesian coordinates of atom i

The data file (.mat format, we recommend using `scipy.io.loadmat` for python users to load this original data) contains five arrays:
- "X" - (7165 x 23 x 23), Coulomb matrices
- "T" - (7165), atomization energies (unit: kcal/mol)
- "P" - (5 x 1433), cross-validation splits as used in [Montavon et al. NIPS, 2012]
- "Z" - (7165 x 23), atomic charges
- "R" - (7165 x 23 x 3), cartesian coordinate (unit: Bohr) of each atom in the molecules

Reference:
Rupp, Matthias, et al. "Fast and accurate modeling of molecular atomization energies with machine learning." Physical review letters 108.5 (2012): 058301.
Montavon, Grégoire, et al. "Learning invariant representations of molecules for atomization energy prediction." Advances in Neural Information Processing Systems. 2012.

# QM7B Examples

QM7b is an extension for the QM7 dataset with additional
properties predicted at different levels (ZINDO, SCS, PBE0, GW).
In total 14 tasks are included for 7211 molecules with up to 7
heavy atoms.

The dataset in .mat format(for python users, we recommend using `scipy.io.loadmat`) includes two arrays:
- "X" - (7211 x 23 x 23), Coulomb matrices
- "T" - (7211 x 14), properties
	Atomization energies E (PBE0, unit: kcal/mol)
	Excitation of maximal optimal absorption E_max (ZINDO, unit: eV)
	Absorption Intensity at maximal absorption I_max (ZINDO)
	Highest occupied molecular orbital HOMO (ZINDO, unit: eV)
	Lowest unoccupied molecular orbital LUMO (ZINDO, unit: eV)
	First excitation energy E_1st (ZINDO, unit: eV)
	Ionization potential IP (ZINDO, unit: eV)
	Electron affinity EA (ZINDO, unit: eV)
	Highest occupied molecular orbital HOMO (PBE0, unit: eV)
	Lowest unoccupied molecular orbital LUMO (PBE0, unit: eV)
	Highest occupied molecular orbital HOMO (GW, unit: eV)
	Lowest unoccupied molecular orbital LUMO (GW, unit: eV)
	Polarizabilities α (PBE0, unit: Å^3)
	Polarizabilities α (SCS, unit: Å^3)

Reference:
- Blum, Lorenz C., and Jean-Louis Reymond. "970 million druglike small molecules for virtual screening in the chemical universe database GDB-13." Journal of the American Chemical Society 131.25 (2009): 8732-8733.
- Montavon, Grégoire, et al. "Machine learning of molecular electronic properties in chemical compound space." New Journal of Physics 15.9 (2013): 095003.
