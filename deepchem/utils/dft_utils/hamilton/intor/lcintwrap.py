from __future__ import annotations
from contextlib import contextmanager
from typing import List, Tuple, Iterator, Optional, Dict
import copy
import torch
import numpy as np
from deepchem.utils import memoize_method
from deepchem.utils.dft_utils import AtomCGTOBasis, CGTOBasis, Lattice
from deepchem.utils.dft_utils.hamilton.intor.utils import np2ctypes, int2ctypes, NDIM, CINT

# Terminology:
# * gauss: one gaussian element (multiple gaussian becomes one shell)
# * shell: one contracted basis (the same contracted gaussian for different atoms
#          counted as different shells)
# * ao: shell that has been splitted into its components,
#       e.g. p-shell is splitted into 3 components for cartesian (x, y, z)

PTR_RINV_ORIG = 4  # from libcint/src/cint_const.h


class LibcintWrapper(object):
    """ A class to wrap the libcint environment and parameters for the
    integrals calculation. This class is used to store the environment and
    parameters for the integrals calculation. It also provides the slicing
    functionality to get the subset of the shells.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> wrap.ao_idxs()
    (0, 6)

    """

    def __init__(self,
                 atombases: List[AtomCGTOBasis],
                 spherical: bool = True,
                 lattice: Optional[Lattice] = None) -> None:
        """Initialize the LibcintWrapper object.

        Parameters
        ----------
        atombases: List[AtomCGTOBasis]
            List of AtomCGTOBasis objects that contains the basis set for each atom
        spherical: bool, optional (default=True)
            Whether the basis is in spherical coordinate (otherwise, it is in
            cartesian coordinate)
        lattice: Optional[Lattice], optional (default=None)
            The lattice of the system (if any)

        """
        self._atombases = atombases
        self._spherical = spherical
        self._fracz = False
        self._natoms = len(atombases)
        self._lattice = lattice

        # get dtype and device for torch's tensors
        self.dtype = atombases[0].bases[0].alphas.dtype
        self.device = atombases[0].bases[0].alphas.device

        # construct _atm, _bas, and _env as well as the parameters
        ptr_env = 20  # initial padding from libcint
        atm_list: List[List[int]] = []
        env_list: List[float] = [0.0] * ptr_env
        bas_list: List[List[int]] = []
        allpos: List[torch.Tensor] = []
        allalphas: List[torch.Tensor] = []
        allcoeffs: List[torch.Tensor] = []
        allangmoms: List[int] = []
        shell_to_atom: List[int] = []
        ngauss_at_shell: List[int] = []
        gauss_to_shell: List[int] = []

        # constructing the triplet lists and also collecting the parameters
        nshells = 0
        ishell = 0
        for iatom, atombasis in enumerate(atombases):
            # construct the atom environment
            assert atombasis.pos.numel(
            ) == NDIM, "Please report this bug in Github"
            atomz = atombasis.atomz
            #                charge    ptr_coord, nucl model (unused for standard nucl model)
            atm_list.append([int(atomz), ptr_env, 1, ptr_env + NDIM, 0, 0])
            env_list.extend(atombasis.pos.detach())
            env_list.append(0.0)
            ptr_env += NDIM + 1

            # check if the atomz is fractional
            if isinstance(atomz, float) or \
                    (isinstance(atomz, torch.Tensor) and atomz.is_floating_point()):
                self._fracz = True

            # add the atom position into the parameter list
            # TODO: consider moving allpos into shell
            allpos.append(atombasis.pos.unsqueeze(0))

            nshells += len(atombasis.bases)
            shell_to_atom.extend([iatom] * len(atombasis.bases))

            # then construct the basis
            for shell in atombasis.bases:
                assert shell.alphas.shape == shell.coeffs.shape and shell.alphas.ndim == 1,\
                    "Please report this bug in Github"
                shell.wfnormalize_()
                ngauss = len(shell.alphas)
                #                iatom, angmom,       ngauss, ncontr, kappa, ptr_exp
                bas_list.append([
                    iatom,
                    shell.angmom,
                    ngauss,
                    1,
                    0,
                    ptr_env,
                    # ptr_coeffs,           unused
                    ptr_env + ngauss,
                    0
                ])
                env_list.extend(shell.alphas.detach())
                env_list.extend(shell.coeffs.detach())
                ptr_env += 2 * ngauss

                # add the alphas and coeffs to the parameters list
                allalphas.append(shell.alphas)
                allcoeffs.append(shell.coeffs)
                allangmoms.extend([shell.angmom] * ngauss)
                ngauss_at_shell.append(ngauss)
                gauss_to_shell.extend([ishell] * ngauss)
                ishell += 1

        # compile the parameters of this object
        self._allpos_params = torch.cat(allpos, dim=0)  # (natom, NDIM)
        self._allalphas_params = torch.cat(allalphas, dim=0)  # (ntot_gauss)
        self._allcoeffs_params = torch.cat(allcoeffs, dim=0)  # (ntot_gauss)
        self._allangmoms = torch.tensor(allangmoms,
                                        dtype=torch.int32,
                                        device=self.device)  # (ntot_gauss)
        self._gauss_to_shell = torch.tensor(gauss_to_shell,
                                            dtype=torch.int32,
                                            device=self.device)

        # convert the lists to numpy to make it contiguous (Python lists are not contiguous)
        self._atm = np.array(atm_list, dtype=np.int32, order="C")
        self._bas = np.array(bas_list, dtype=np.int32, order="C")
        self._env = np.array(env_list, dtype=np.float64, order="C")

        # construct the full shell mapping
        shell_to_aoloc = [0]
        ao_to_shell: List[int] = []
        ao_to_atom: List[int] = []
        for i in range(nshells):
            nao_at_shell_i = self._nao_at_shell(i)
            shell_to_aoloc_i = shell_to_aoloc[-1] + nao_at_shell_i
            shell_to_aoloc.append(shell_to_aoloc_i)
            ao_to_shell.extend([i] * nao_at_shell_i)
            ao_to_atom.extend([shell_to_atom[i]] * nao_at_shell_i)

        self._ngauss_at_shell_list = ngauss_at_shell
        self._shell_to_aoloc = np.array(shell_to_aoloc, dtype=np.int32)
        self._shell_idxs = (0, nshells)
        self._ao_to_shell = torch.tensor(ao_to_shell,
                                         dtype=torch.long,
                                         device=self.device)
        self._ao_to_atom = torch.tensor(ao_to_atom,
                                        dtype=torch.long,
                                        device=self.device)

    @property
    def parent(self) -> LibcintWrapper:
        """parent is defined as the full LibcintWrapper where it takes the
        full shells for the integration (without the need for subsetting)

        Returns
        -------
        LibcintWrapper
            The parent of this object

        """
        return self

    @property
    def natoms(self) -> int:
        """Number of atoms in the environment

        Returns
        -------
        int
            Number of atoms in the environment

        """
        return self._natoms

    @property
    def fracz(self) -> bool:
        """Indicating whether we are working with fractional z

        Returns
        -------
        bool
            Whether we are working with fractional z

        """
        return self._fracz

    @property
    def lattice(self) -> Optional[Lattice]:
        """The lattice of the system

        Returns
        -------
        Optional[Lattice]
            The lattice of the system

        """
        return self._lattice

    @property
    def spherical(self) -> bool:
        """returns whether the basis is in spherical coordinate (otherwise, it
        is in cartesian coordinate)

        Returns
        -------
        bool
            Whether the basis is in spherical coordinate

        """
        return self._spherical

    @property
    def atombases(self) -> List[AtomCGTOBasis]:
        """List of AtomCGTOBasis objects that contains the basis set for each atom

        Returns
        -------
        List[AtomCGTOBasis]
            List of AtomCGTOBasis objects that contains the basis set for each atom

        """
        return self._atombases

    @property
    def atm_bas_env(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """returns the triplet lists, i.e. atm, bas, env
        this shouldn't change in the sliced wrapper

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The triplet lists, i.e. atm, bas, env

        """
        return self._atm, self._bas, self._env

    @property
    def full_angmoms(self) -> torch.Tensor:
        """Returns the full angular momentums tensor.

        Returns
        -------
        torch.Tensor
            The full angular momentums tensor.

        """
        return self._allangmoms

    @property
    def params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns all the parameters of this object.

        This shouldn't change in the sliced wrapper.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            All the parameters of this object.

        """

        return self._allcoeffs_params, self._allalphas_params, self._allpos_params

    @property
    def shell_idxs(self) -> Tuple[int, int]:
        """Returns the lower and upper indices of the shells of this object
        in the absolute index (upper is exclusive).

        Returns
        -------
        Tuple[int, int]
            The lower and upper indices of the shells.
        """
        return self._shell_idxs

    @property
    def full_shell_to_aoloc(self) -> np.ndarray:
        """Returns the full array mapping from shell index to absolute AO location.

        Returns
        -------
        np.ndarray
            The full array mapping from shell index to absolute AO location.
        """
        return self._shell_to_aoloc

    @property
    def full_gauss_to_shell(self) -> torch.Tensor:
        """Returns the full index mapping from gaussian to shell tensor.

        Returns
        -------
        torch.Tensor
            The full index mapping from gaussian to shell tensor.
        """
        return self._gauss_to_shell

    @property
    def full_ao_to_atom(self) -> torch.Tensor:
        """Returns the full array mapping from atomic orbital index to the atom location.

        Returns
        -------
        torch.Tensor
            The full array mapping from atomic orbital index to the atom location.
        """
        return self._ao_to_atom

    @property
    def full_ao_to_shell(self) -> torch.Tensor:
        """Returns the full array mapping from atomic orbital index to the shell location.

        Returns
        -------
        torch.Tensor
            The full array mapping from atomic orbital index to the shell location.
        """
        return self._ao_to_shell

    @property
    def ngauss_at_shell(self) -> List[int]:
        """Returns the number of gaussian basis at the given shell.

        Returns
        -------
        List[int]
            The number of gaussian basis at the given shell.
        """
        return self._ngauss_at_shell_list

    @memoize_method
    def __len__(self) -> int:
        """Returns the total number of shells.

        Returns
        -------
        int
            The total number of shells.
        """
        return self.shell_idxs[-1] - self.shell_idxs[0]

    @memoize_method
    def nao(self) -> int:
        """Returns the number of atomic orbitals.

        Returns
        -------
        int
            The number of atomic orbitals.
        """
        shell_idxs = self.shell_idxs
        return self.full_shell_to_aoloc[
            shell_idxs[-1]] - self.full_shell_to_aoloc[shell_idxs[0]]

    @memoize_method
    def ao_idxs(self) -> Tuple[int, int]:
        """Returns the lower and upper indices of the atomic orbitals.

        Returns
        -------
        Tuple[int, int]
            The lower and upper indices of the atomic orbitals.
        """
        shell_idxs = self.shell_idxs
        return self.full_shell_to_aoloc[
            shell_idxs[0]], self.full_shell_to_aoloc[shell_idxs[1]]

    @memoize_method
    def ao_to_atom(self) -> torch.Tensor:
        """Returns the relative mapping from atomic orbital relative index to the absolute atom position.

        Returns
        -------
        torch.Tensor
            The relative mapping from atomic orbital relative index to the absolute atom position.
        """
        return self.full_ao_to_atom[slice(*self.ao_idxs())]

    @memoize_method
    def ao_to_shell(self) -> torch.Tensor:
        """Returns the relative mapping from atomic orbital relative index to the absolute shell position.

        Returns
        -------
        torch.Tensor
            The relative mapping from atomic orbital relative index to the absolute shell position.
        """
        return self.full_ao_to_shell[slice(*self.ao_idxs())]

    def __getitem__(self, inp) -> LibcintWrapper:
        """Gets the subset of the shells, but keeping the environment and parameters the same.

        Returns
        -------
        LibcintWrapper
            The subset of the shells with the same environment and parameters.
        """
        assert isinstance(inp, slice)
        assert inp.step is None or inp.step == 1
        assert inp.start is not None or inp.stop is not None

        # complete the slice
        nshells = self.shell_idxs[-1]
        if inp.start is None and inp.stop is not None:
            inp = slice(0, inp.stop)
        elif inp.start is not None and inp.stop is None:
            inp = slice(inp.start, nshells)

        # make the slice positive
        if inp.start < 0:
            inp = slice(inp.start + nshells, inp.stop)
        if inp.stop < 0:
            inp = slice(inp.start, inp.stop + nshells)

        return SubsetLibcintWrapper(self, inp)

    @memoize_method
    def get_uncontracted_wrapper(self) -> Tuple[LibcintWrapper, torch.Tensor]:
        """Returns the uncontracted LibcintWrapper and the mapping from uncontracted to contracted AO indices.

        Returns
        -------
        Tuple[LibcintWrapper, torch.Tensor]
            The uncontracted LibcintWrapper and the mapping from uncontracted to contracted AO indices.
        """
        new_atombases = []
        for atombasis in self.atombases:
            atomz = atombasis.atomz
            pos = atombasis.pos
            new_bases = []
            for shell in atombasis.bases:
                angmom = shell.angmom
                alphas = shell.alphas
                coeffs = shell.coeffs
                normalized = shell.normalized
                new_bases.extend([
                    CGTOBasis(angmom,
                              alpha[None],
                              coeff[None],
                              normalized=normalized)
                    for (alpha, coeff) in zip(alphas, coeffs)
                ])
            new_atombases.append(
                AtomCGTOBasis(atomz=atomz, bases=new_bases, pos=pos))
        uncontr_wrapper = LibcintWrapper(new_atombases,
                                         spherical=self.spherical)

        # get the mapping uncontracted ao to the contracted ao
        uao2ao: List[int] = []
        idx_ao = 0
        # iterate over shells
        for i in range(len(self)):
            nao = self._nao_at_shell(i)
            uao2ao += list(range(idx_ao,
                                 idx_ao + nao)) * self.ngauss_at_shell[i]
            idx_ao += nao
        uao2ao_res = torch.tensor(uao2ao, dtype=torch.long, device=self.device)
        return uncontr_wrapper, uao2ao_res

    @staticmethod
    def concatenate(*wrappers: LibcintWrapper) \
            -> Tuple[LibcintWrapper, ...]:
        """
        Concatenate the parents of wrappers, then returns the subsets
        corresponds to the wrappers.
        This function returns an environment that contains all the atoms from
        the parents of all the wrappers.
        If all the wrappers are from the same parent, then this function does
        not do anything.

        Parameters
        ----------
        wrappers : Tuple[LibcintWrapper, ...]
            The wrappers to concatenate.

        Returns
        -------
        Tuple[LibcintWrapper, ...]
            The subsets corresponding to the input wrappers.

        """

        # construct the parent mapping
        unique_parents: List[LibcintWrapper] = []
        unique_pids: Dict[int, int] = {}
        w2pidx: List[int] = []
        shell_idxs: List[Tuple[int, int]] = []
        cumsum_plen: List[int] = [0]
        for w in wrappers:
            parent = w.parent
            pid = id(parent)
            shell_idxs.append(w.shell_idxs)
            if pid not in unique_pids:
                idx = len(unique_parents)
                unique_pids[pid] = idx
                unique_parents.append(parent)
                cumsum_plen.append(cumsum_plen[-1] + len(parent))
            w2pidx.append(unique_pids[pid])

        # check the length of the unique parents, if there is only 1 unique parent,
        # then just return as it is
        assert len(unique_parents) > 0
        if len(unique_parents) == 1:
            return (*wrappers,)

        # check if the unique parents have the same options (except for the atombases)
        p0 = unique_parents[0]
        sph = p0.spherical
        latt = p0.lattice
        atombases = copy.copy(p0.atombases)  # shallow copy
        for p in unique_parents[1:]:
            assert p.spherical == sph
            assert p.lattice == latt
            atombases.extend(p.atombases)

        # get the grand environment
        grandp = LibcintWrapper(atombases, spherical=sph, lattice=latt)

        # get the subsets which correspond to the input wrappers
        res: List[LibcintWrapper] = []
        for i, w in enumerate(wrappers):
            sh_idx0, sh_idx1 = shell_idxs[i]
            offset = cumsum_plen[w2pidx[i]]
            sh_idx0 = sh_idx0 + offset
            sh_idx1 = sh_idx1 + offset
            res.append(grandp[sh_idx0:sh_idx1])

        return (*res,)

    # misc functions
    @contextmanager
    def centre_on_r(self, r: torch.Tensor) -> Iterator:
        """Sets the centre of coordinate to r.

        Parameters
        ----------
        r : torch.Tensor
            The coordinates to set the centre to.

        Yields
        ------
        Iterator
            A context manager.
        """
        try:
            env = self.atm_bas_env[-1]
            prev_centre = env[PTR_RINV_ORIG:PTR_RINV_ORIG + NDIM]
            env[PTR_RINV_ORIG:PTR_RINV_ORIG + NDIM] = r.detach().numpy()
            yield
        finally:
            env[PTR_RINV_ORIG:PTR_RINV_ORIG + NDIM] = prev_centre

    def _nao_at_shell(self, sh: int) -> int:
        """Returns the number of atomic orbital at the given shell index.

        Parameters
        ----------
        sh : int
            The shell index.

        Returns
        -------
        int
            The number of atomic orbital at the given shell index.
        """
        if self.spherical:
            op = CINT().CINTcgto_spheric
        else:
            op = CINT().CINTcgto_cart
        bas = self.atm_bas_env[1]
        return op(int2ctypes(sh), np2ctypes(bas))


class SubsetLibcintWrapper(LibcintWrapper):
    """
    A class to represent the subset of LibcintWrapper.
    If put into integrals or evaluations, this class will only evaluate
        the subset of the shells from its parent.
    The environment will still be the same as its parent.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import AtomCGTOBasis, LibcintWrapper, loadbasis
    >>> dtype = torch.double
    >>> d = 1.0
    >>> pos_requires_grad = True
    >>> pos1 = torch.tensor([0.1 * d,  0.0 * d,  0.2 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos2 = torch.tensor([0.0 * d,  1.0 * d, -0.4 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> pos3 = torch.tensor([0.2 * d, -1.4 * d, -0.9 * d], dtype=dtype, requires_grad=pos_requires_grad)
    >>> poss = [pos1, pos2, pos3]
    >>> atomzs = [1, 1, 1]
    >>> allbases = [
    ...     loadbasis("%d:%s" % (max(atomz, 1), "3-21G"), dtype=dtype, requires_grad=False)
    ...     for atomz in atomzs
    ... ]
    >>> atombases = [
    ...     AtomCGTOBasis(atomz=atomzs[i], bases=allbases[i], pos=poss[i])
    ...     for i in range(len(allbases))
    ... ]
    >>> wrap = LibcintWrapper(atombases, True, None)
    >>> subset = wrap[1:3]
    >>> subset.ao_idxs()
    (1, 3)

    """

    def __init__(self, parent: LibcintWrapper, subset: slice):
        """
        Initializes the SubsetLibcintWrapper.

        Parameters
        ----------
        parent : LibcintWrapper
            The parent LibcintWrapper.
        subset : slice
            The slice representing the subset of shells.
        """
        self._parent = parent
        self._shell_idxs = subset.start, subset.stop

    @property
    def parent(self) -> LibcintWrapper:
        """
        Returns the parent LibcintWrapper.

        Returns
        -------
        LibcintWrapper
            The parent LibcintWrapper.
        """
        return self._parent

    @property
    def shell_idxs(self) -> Tuple[int, int]:
        """
        Returns the shell indices of the subset.

        Returns
        -------
        Tuple[int, int]
            The shell indices of the subset.
        """
        return self._shell_idxs

    @memoize_method
    def get_uncontracted_wrapper(self):
        """
        Returns the uncontracted LibcintWrapper and the mapping from
        uncontracted atomic orbital (relative index) to the relative index
        of the atomic orbital of the contracted wrapper.

        Returns
        -------
        Tuple[LibcintWrapper, torch.Tensor]
            The uncontracted LibcintWrapper and the mapping from uncontracted
            atomic orbital (relative index) to the relative index of the atomic
            orbital of the contracted wrapper.
        """
        pu_wrapper, p_uao2ao = self._parent.get_uncontracted_wrapper()

        # determine the corresponding shell indices in the new uncontracted wrapper
        shell_idxs = self.shell_idxs
        gauss_idx0 = sum(self._parent.ngauss_at_shell[:shell_idxs[0]])
        gauss_idx1 = sum(self._parent.ngauss_at_shell[:shell_idxs[1]])
        u_wrapper = pu_wrapper[gauss_idx0:gauss_idx1]

        # construct the uao (relative index) mapping to the absolute index
        # of the atomic orbital in the contracted basis
        uao2ao = []
        idx_ao = 0
        for i in range(shell_idxs[0], shell_idxs[1]):
            nao = self._parent._nao_at_shell(i)
            uao2ao += list(range(
                idx_ao, idx_ao + nao)) * self._parent.ngauss_at_shell[i]
            idx_ao += nao
        uao2ao_res = torch.tensor(uao2ao, dtype=torch.long, device=self.device)
        return u_wrapper, uao2ao_res

    def __getitem__(self, inp):
        """
        Raises NotImplementedError as indexing of SubsetLibcintWrapper is not implemented.

        Parameters
        ----------
        inp : Any
            The input for indexing.

        Raises
        ------
        NotImplementedError
            Indexing of SubsetLibcintWrapper is not implemented.
        """
        raise NotImplementedError(
            "Indexing of SubsetLibcintWrapper is not implemented")

    def __getattr__(self, name):
        """
        Delegate attribute access to the parent LibcintWrapper.

        Parameters
        ----------
        name : str
            The name of the attribute.

        Returns
        -------
        Any
            The attribute value from the parent LibcintWrapper.
        """
        return getattr(self._parent, name)
