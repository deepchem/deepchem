from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple, List, Sequence
import re
import copy
from deepchem.utils.dft_utils.hamilton.intor.symmetry import BaseSymmetry, S1Symmetry, S4Symmetry


class IntorNameManager(object):
    """
    A class for integral name manipulation.

    This class is supposed to be string-manipulation only, so no array calculation
    should be performed here.

    Examples
    --------
    >>> mgr = IntorNameManager("int1e", "r0")
    >>> mgr.fullname
    'int1e_r0'
    >>> mgr.get_intgl_name(True)
    'int1e_r0_sph'
    >>> mgr.get_ft_intgl_name(True)
    'GTO_ft_r0_sph'
    >>> mgr.get_intgl_symmetry([0, 1, 2, 0]).code
    's1'
    >>> mgr.get_intgl_components_shape()
    (3,)
    """

    # Operators name must not contain separator name
    ops_name = ["ip", "rr"]  # Name of basis operators
    sep_name = ["a", "b"]  # Separator of basis (other than the middle operator)

    # Components shape of raw operator and basis operators
    # should be a tuple with AT MOST 1 element
    rawop_comp = defaultdict(
        tuple,  # type: ignore
        {
            "r0": (3,),
            "r0r0": (9,),
            "r0r0r0": (27,),
        })
    op_comp = defaultdict(
        tuple,  # type: ignore
        {
            "ip": (3,),
        })

    # The number of new dimensions added with the operators
    rawop_ndim = defaultdict(int, {k: len(v) for (k, v) in rawop_comp.items()})
    op_ndim = defaultdict(int, {k: len(v) for (k, v) in op_comp.items()})

    def __init__(self, int_type: str, shortname: str):
        """
        Initialize the IntorNameManager.

        Parameters
        ----------
        int_type : str
            Integral type.
        shortname : str
            Short name representing the integral.
        """
        self._int_type = int_type
        self._shortname = shortname
        self._rawop, self._ops = self.split_name(int_type, shortname)
        self._nbasis = len(self._ops)
        self._imid = (self._nbasis +
                      1) // 2  # Middle index (where the rawops should be)

    @property
    def fullname(self) -> str:
        """
        Get the full name of the integral.

        Returns
        -------
        str
            Full name of the integral.
        """
        return self._int_type + "_" + self._shortname

    @property
    def rawopname(self) -> str:
        """
        Get the raw operator name of the integral.

        Returns
        -------
        str
            Raw operator name of the integral.
        """
        return self._rawop

    @property
    def int_type(self) -> str:
        """
        Get the integral type.

        Returns
        -------
        str
            Integral type.
        """
        return self._int_type

    @property
    def shortname(self) -> str:
        """
        Get the short name representing the integral.

        Returns
        -------
        str
            Short name representing the integral.
        """
        return self._shortname

    def get_intgl_name(self, spherical: bool) -> str:
        """
        Get the full name of the integral in libcint library.

        Parameters
        ----------
        spherical : bool
            Flag indicating whether the integral is in spherical or cartesian coordinates.

        Returns
        -------
        str
            Full name of the integral in libcint library.
        """
        cartsph = "sph" if spherical else "cart"
        return f"{self.fullname}_{cartsph}"

    def get_ft_intgl_name(self, spherical: bool) -> str:
        """
        Get the full name of the fourier transform integral in libcint library.

        Parameters
        ----------
        spherical : bool
            Flag indicating whether the integral is in spherical or cartesian coordinates.

        Returns
        -------
        str
            Full name of the fourier transform integral in libcint library.
        """
        cartsph = "sph" if spherical else "cart"
        int_type = self._int_type
        if int_type == "int1e":
            return "GTO_ft_%s_%s" % (self._shortname, cartsph)
        else:
            raise NotImplementedError("Unimplemented FT integral for %s" %
                                      int_type)

    def get_intgl_deriv_namemgr(self, derivop: str,
                                ibasis: int) -> IntorNameManager:
        """
        Get the name manager of a new integral when a derivative operation is applied.

        Parameters
        ----------
        derivop : str
            String of the derivative operation.
        ibasis : int
            Index of the basis where the derivative operation should be performed (0-based).

        Returns
        -------
        IntorNameManager
            Name manager of the new integral.
        """
        assert derivop in self.ops_name
        assert ibasis < self._nbasis

        ops = copy.copy(self._ops)
        ops[ibasis] = [derivop] + ops[ibasis]
        sname = self.join_name(self._int_type, self._rawop, ops)
        return IntorNameManager(self._int_type, sname)

    def get_intgl_deriv_newaxispos(self, derivop: str,
                                   ibasis: int) -> Optional[int]:
        """
        Get the new axis position in the new integral name.

        Parameters
        ----------
        derivop : str
            Derivative operation.
        ibasis : int
            Index of the basis where the derivative operation should be performed (0-based).

        Returns
        -------
        Optional[int]
            New axis position (if any) or None if no new axis is inserted.
        """
        op_ndim = self.op_ndim[derivop]
        if op_ndim == 0:
            return None

        ops_flat: List[str] = sum(self._ops[:ibasis], [])
        new_ndim = sum([self.op_ndim[op] for op in ops_flat])

        # check if rawsname should also be included
        include_rname = ibasis >= self._imid
        if include_rname:
            new_ndim += self.rawop_ndim[self._rawop]

        return new_ndim

    def get_intgl_components_shape(self) -> Tuple[int, ...]:
        """
        Get the component shape of the array of the integral.

        Returns
        -------
        Tuple[int, ...]
            Component shape of the array of the integral.
        """
        ops_flat_l: List[str] = sum(self._ops[:self._imid], [])
        ops_flat_r: List[str] = sum(self._ops[self._imid:], [])
        comp_shape = sum([self.op_comp[op] for op in ops_flat_l], ()) + \
            self.rawop_comp[self._rawop] + \
            sum([self.op_comp[op] for op in ops_flat_r], ())
        return comp_shape

    def get_intgl_symmetry(self, unique: Sequence[int]) -> BaseSymmetry:
        """
        Get the symmetry of the integral.

        Parameters
        ----------
        unique : Sequence[int]
            Sequence representing the uniqueness of each dimension.

        Returns
        -------
        BaseSymmetry
            Symmetry of the integral.
        """
        all_same = all([u == 0 for u in unique])
        if all_same:
            if self._int_type == "int2e":
                if self._shortname == "ar12b":
                    return S4Symmetry()
        return S1Symmetry()

    def get_transpose_path_to(
            self, other: IntorNameManager) -> Optional[List[Tuple[int, int]]]:
        """
        Check if the integration `other` can be achieved by transposing `self`.

        Parameters
        ----------
        other : IntorNameManager
            IntorNameManager instance representing the other integral.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of two dims if it can for the transpose-path of `self`
            to get the same result as `other`, None if it cannot.
        """
        nbasis = self._nbasis
        # get the basis transpose paths
        if nbasis == 2:
            transpose_paths: List[List[Tuple[int, int]]] = [
                [],
                [(-1, -2)],
            ]
        elif nbasis == 3:
            # note: the third basis is usually an auxiliary basis which typically
            # different from the first two
            transpose_paths = [
                [],
                [(-2, -3)],
            ]
        elif nbasis == 4:
            transpose_paths = [
                [],
                [(-3, -4)],
                [(-1, -2)],
                [(-1, -3), (-2, -4)],
                [(-1, -3), (-2, -4), (-2, -1)],
                [(-1, -3), (-2, -4), (-3, -4)],
            ]
        else:
            raise self._nbasis_error(nbasis)

        def _swap(p: List[List[str]],
                  path: List[Tuple[int, int]]) -> List[List[str]]:
            # swap the pattern according to the given transpose path
            r = p[:]  # make a copy
            for i0, i1 in path:
                r[i0], r[i1] = r[i1], r[i0]
            return r

        # try all the transpose path until gets a match
        for transpose_path in transpose_paths:
            if _swap(self._ops, transpose_path) == other._ops:
                return transpose_path
        return None

    def get_comp_permute_path(
            self, transpose_path: List[Tuple[int, int]]) -> List[int]:
        """
        Get the component permute path given the basis transpose path.

        Parameters
        ----------
        transpose_path : List[Tuple[int, int]]
            Basis transpose path.

        Returns
        -------
        List[int]
            Component permute path.
        """
        flat_ops: List[str] = sum(self._ops, [])
        _ = flat_ops.count("ip")  # number of ip operators

        # get the positions of the axes
        dim_pos = []
        ioffset = 0
        for i, ops in enumerate(self._ops):
            if i == self._imid:
                naxes = self.rawop_ndim[self._rawop]
                dim_pos.append(list(range(ioffset, ioffset + naxes)))
                ioffset += naxes
            naxes = sum([self.op_ndim[op] for op in ops])
            dim_pos.append(list(range(ioffset, ioffset + naxes)))
            ioffset += naxes

        # add the bases' axes (assuming each basis only occupy one axes)
        for i in range(self._nbasis):
            dim_pos.append([ioffset])
            ioffset += 1

        # swap the axes
        for t0, t1 in transpose_path:
            s0 = t0 + self._nbasis
            s1 = t1 + self._nbasis
            s0 += 1 if s0 >= self._imid else 0
            s1 += 1 if s1 >= self._imid else 0
            dim_pos[s0], dim_pos[s1] = dim_pos[s1], dim_pos[s0]

        # flatten the list to get the permutation path
        dim_pos_flat: List[int] = sum(dim_pos, [])
        return dim_pos_flat

    @classmethod
    def split_name(cls, int_type: str,
                   shortname: str) -> Tuple[str, List[List[str]]]:
        """
        Split the shortname into operator per basis and return the raw shortname as well.

        Parameters
        ----------
        int_type : str
            Integral type.
        shortname : str
            Short name representing the integral.

        Returns
        -------
        Tuple[str, List[List[str]]]
            Raw shortname and list of basis operators.
        """
        deriv_ops = cls.ops_name
        deriv_pattern = re.compile("(" + ("|".join(deriv_ops)) + ")")

        # get the raw shortname (i.e. shortname without derivative operators)
        rawsname = shortname
        for op in deriv_ops:
            rawsname = rawsname.replace(op, "")

        nbasis = cls.get_nbasis(int_type)
        if nbasis == 2:
            ops_str = shortname.split(rawsname)
        elif nbasis == 3:
            assert rawsname.startswith("a"), rawsname
            rawsname = rawsname[1:]
            ops_l, ops_r = shortname.split(rawsname)
            ops_l1, ops_l2 = ops_l.split("a")
            ops_str = [ops_l1, ops_l2, ops_r]
        elif nbasis == 4:
            assert rawsname.startswith("a") and rawsname.endswith("b"), rawsname
            rawsname = rawsname[1:-1]
            ops_l, ops_r = shortname.split(rawsname)
            ops_l1, ops_l2 = ops_l.split("a")
            ops_r1, ops_r2 = ops_r.split("b")
            ops_str = [ops_l1, ops_l2, ops_r1, ops_r2]
        else:
            raise cls._nbasis_error(nbasis)

        ops = [re.findall(deriv_pattern, op_str) for op_str in ops_str]
        assert len(ops) == nbasis
        return rawsname, ops

    @classmethod
    def join_name(cls, int_type: str, rawsname: str,
                  ops: List[List[str]]) -> str:
        """
        Get the shortname given rawsname and list of basis ops.

        Parameters
        ----------
        int_type : str
            Integral type.
        rawsname : str
            Raw shortname.
        ops : List[List[str]]
            List of basis operators.

        Returns
        -------
        str
            Shortname representing the integral.
        """
        nbasis = cls.get_nbasis(int_type)
        ops_str = ["".join(op) for op in ops]
        assert len(ops_str) == nbasis

        if nbasis == 2:
            return ops_str[0] + rawsname + ops_str[1]
        elif nbasis == 3:
            return ops_str[0] + cls.sep_name[0] + ops_str[
                1] + rawsname + ops_str[2]
        elif nbasis == 4:
            return ops_str[0] + cls.sep_name[0] + ops_str[1] + rawsname + ops_str[2] + \
                cls.sep_name[1] + ops_str[3]
        else:
            raise cls._nbasis_error(nbasis)

    @classmethod
    def get_nbasis(cls, int_type: str) -> int:
        """
        Get the number of basis according to the integral type.

        Parameters
        ----------
        int_type : str
            Integral type.

        Returns
        -------
        int
            Number of basis.
        """
        if int_type == "int1e" or int_type == "int2c2e":
            return 2
        elif int_type == "int3c2e":
            return 3
        elif int_type == "int2e":
            return 4
        else:
            raise RuntimeError(f"Unknown integral type: {int_type}")

    @classmethod
    def _nbasis_error(cls, nbasis: int):
        return RuntimeError(f"Unknown integral with {nbasis} basis")
