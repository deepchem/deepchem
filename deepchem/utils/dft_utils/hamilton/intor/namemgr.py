from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple, List, Sequence
import re
import copy
from deepchem.utils.dft_utils.hamilton.intor.symmetry import BaseSymmetry, S1Symmetry, S4Symmetry

# name manager of the integrals

class IntorNameManager(object):
    # a class for integral name manipulation
    # this class is supposed to be string-manipulation only, so no array calculation
    # should be performed here

    # ops name must not contain sep name
    ops_name = ["ip", "rr"]  # name of basis operators
    sep_name = ["a", "b"]  # separator of basis (other than the middle operator)

    # components shape of raw operator and basis operators
    # should be a tuple with AT MOST 1 element
    rawop_comp = defaultdict(tuple, {  # type: ignore
        "r0": (3,),
        "r0r0": (9,),
        "r0r0r0": (27,),
    })
    op_comp = defaultdict(tuple, {  # type: ignore
        "ip": (3,),
    })

    # the number of new dimensions added with the operators
    rawop_ndim = defaultdict(int, {k: len(v) for (k, v) in rawop_comp.items()})
    op_ndim = defaultdict(int, {k: len(v) for (k, v) in op_comp.items()})

    def __init__(self, int_type: str, shortname: str):
        self._int_type = int_type
        self._shortname = shortname
        self._rawop, self._ops = self.split_name(int_type, shortname)
        self._nbasis = len(self._ops)
        self._imid = (self._nbasis + 1) // 2  # middle index (where the rawops should be)

    @property
    def fullname(self):
        return self._int_type + "_" + self._shortname

    @property
    def rawopname(self):
        return self._rawop

    @property
    def int_type(self):
        return self._int_type

    @property
    def shortname(self):
        return self._shortname

    def get_intgl_name(self, spherical: bool) -> str:
        # returns the full name of the integral in libcint library
        cartsph = "sph" if spherical else "cart"
        return f"{self.fullname}_{cartsph}"

    def get_ft_intgl_name(self, spherical: bool) -> str:
        # returns the full name of the fourier transform integral in libcint library
        cartsph = "sph" if spherical else "cart"
        int_type = self._int_type
        if int_type == "int1e":
            return "GTO_ft_%s_%s" % (self._shortname, cartsph)
        else:
            raise NotImplementedError("Unimplemented FT integral for %s" % int_type)

    def get_intgl_deriv_namemgr(self, derivop: str, ibasis: int) -> IntorNameManager:
        # get the name manager of a new integral when derivop is applied to
        # ibasis-th basis
        # derivop: string of the derivative operation
        # ibasis: which basis the derivative operation should be performed (0-based)
        # returns the shortname of the integral

        assert derivop in self.ops_name
        assert ibasis < self._nbasis

        ops = copy.copy(self._ops)
        ops[ibasis] = [derivop] + ops[ibasis]
        sname = self.join_name(self._int_type, self._rawop, ops)
        return IntorNameManager(self._int_type, sname)

    def get_intgl_deriv_newaxispos(self, derivop: str, ibasis: int) -> Optional[int]:
        # get the new axis position in the new integral name
        # derivop and basis like get_intgl_deriv_shortname
        # returns an int of the new axis position (if any) or
        # None if no new axis is inserted

        # get how many new axes the operator is going to add
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
        # returns the component shape of the array of the given integral
        ops_flat_l: List[str] = sum(self._ops[:self._imid], [])
        ops_flat_r: List[str] = sum(self._ops[self._imid:], [])
        comp_shape = sum([self.op_comp[op] for op in ops_flat_l], ()) + \
            self.rawop_comp[self._rawop] + \
            sum([self.op_comp[op] for op in ops_flat_r], ())
        return comp_shape

    def get_intgl_symmetry(self, unique: Sequence[int]) -> BaseSymmetry:
        # get the symmetry of the integral
        all_same = all([u == 0 for u in unique])
        if all_same:
            if self._int_type == "int2e":
                if self._shortname == "ar12b":
                    return S4Symmetry()
        return S1Symmetry()

    def get_transpose_path_to(self, other: IntorNameManager) -> Optional[List[Tuple[int, int]]]:
        # check if the integration `other` can be achieved by transposing `self`
        # returns None if it cannot.
        # returns the list of two dims if it can for the transpose-path of `self`
        # to get the same result as `other`

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

        def _swap(p: List[List[str]], path: List[Tuple[int, int]]) -> List[List[str]]:
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

    def get_comp_permute_path(self, transpose_path: List[Tuple[int, int]]) -> List[int]:
        # get the component permute path given the basis transpose path
        flat_ops: List[str] = sum(self._ops, [])
        n_ip = flat_ops.count("ip")

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
    def split_name(cls, int_type: str, shortname: str) -> Tuple[str, List[List[str]]]:
        # split the shortname into operator per basis and return the raw shortname as well
        # the first returned element is the raw shortname (i.e. the middle operator)
        # while the second returned element is the list of basis-operator shortname

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
    def join_name(cls, int_type: str, rawsname: str, ops: List[List[str]]) -> str:
        # get the shortname given rawsname and list of basis ops
        nbasis = cls.get_nbasis(int_type)
        ops_str = ["".join(op) for op in ops]
        assert len(ops_str) == nbasis

        if nbasis == 2:
            return ops_str[0] + rawsname + ops_str[1]
        elif nbasis == 3:
            return ops_str[0] + cls.sep_name[0] + ops_str[1] + rawsname + ops_str[2]
        elif nbasis == 4:
            return ops_str[0] + cls.sep_name[0] + ops_str[1] + rawsname + ops_str[2] + \
                cls.sep_name[1] + ops_str[3]
        else:
            raise cls._nbasis_error(nbasis)

    @classmethod
    def get_nbasis(cls, int_type: str) -> int:
        # get the number of basis according to the integral type
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
