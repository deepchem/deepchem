import torch
from collections import defaultdict
from deepchem.utils import get_option
from typing import Union, List, Optional, Mapping, Callable, Dict
from deepchem.utils.periodic_table_utils import atom_bragg_radii, atom_expected_radii, get_period
from deepchem.utils.dft_utils import DasguptaTrunc, NWChemTrunc, NoTrunc, Lattice, BaseGrid, RadialGrid, LogM3Transformation, TreutlerM4Transformation, DE2Transformation, LebedevGrid, TruncatedLebedevGrid, BeckeGrid, PBCBeckeGrid

# list of alphas for de2 transformation from https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.24761
__sg2_dasgupta_alphas = defaultdict(
    lambda: 1.0, {
        1: 2.6,
        3: 3.2,
        4: 2.4,
        5: 2.4,
        6: 2.2,
        7: 2.2,
        8: 2.2,
        9: 2.2,
        11: 3.2,
        12: 2.4,
        13: 2.5,
        14: 2.3,
        15: 2.5,
        16: 2.5,
        17: 2.5,
    })
__sg3_dasgupta_alphas = defaultdict(
    lambda: 1.0, {
        1: 2.7,
        3: 3.0,
        4: 2.4,
        5: 2.4,
        6: 2.4,
        7: 2.4,
        8: 2.6,
        9: 2.1,
        11: 3.2,
        12: 2.6,
        13: 2.6,
        14: 2.8,
        15: 2.4,
        16: 2.4,
        17: 2.6,
    })

# list of optimized xi for M4 transformation from Treutler Table I
# https://doi.org/10.1063/1.469408
__treutler_xi = defaultdict(
    lambda: 1.0, {
        1: 0.8,
        2: 0.9,
        3: 1.8,
        4: 1.4,
        5: 1.3,
        6: 1.1,
        7: 0.9,
        8: 0.9,
        9: 0.9,
        10: 0.9,
        11: 1.4,
        12: 1.3,
        13: 1.3,
        14: 1.2,
        15: 1.1,
        16: 1.0,
        17: 1.0,
        18: 1.0,
        19: 1.5,
        20: 1.4,
        21: 1.3,
        22: 1.2,
        23: 1.2,
        24: 1.2,
        25: 1.2,
        26: 1.2,
        27: 1.2,
        28: 1.1,
        29: 1.1,
        30: 1.1,
        31: 1.1,
        32: 1.0,
        33: 0.9,
        34: 0.9,
        35: 0.9,
        36: 0.9,
    })

# number of angular points to precision
__nang2prec = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}

_dtype = torch.double
_device = torch.device("cpu")


def get_grid(atomzs: Union[List[int], torch.Tensor],
             atompos: torch.Tensor,
             *,
             lattice: Optional[Lattice] = None,
             nr: Union[int, Callable[[int], int]] = 99,
             nang: Union[int, Callable[[int], int]] = 590,
             radgrid_generator: str = "uniform",
             radgrid_transform: str = "sg2-dasgupta",
             atom_radii: str = "expected",
             multiatoms_scheme: str = "becke",
             truncate: Optional[str] = "dasgupta",
             dtype: torch.dtype = _dtype,
             device: torch.device = _device) -> BaseGrid:
    """Get the Molecular Grid.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_grid
    >>> grid = get_grid(torch.tensor([1]), torch.tensor([[0,0,0]], dtype=torch.float64))
    >>> grid.get_rgrid().shape
    torch.Size([16710, 3])

    Parameters
    ----------
    atomzs: Union[List[int], torch.Tensor]
        Atomic Number of the atoms in the molecule.
    atompos: torch.Tensor
        Positions of the atoms in 3d. (natoms, ndim)
    lattice: Optional[Lattice] (Default None)
        Lattice Object descibing periodicity.
    nr: Union[int, Callable[[int], int]] (default 99)
        Number of radial points.
    nang: Union[int, Callable[[int], int]] = 590
        Number of angular points.
    radgrid_generator: str (default "uniform")
        Radial grid generator.
    radgrid_transform: str (default "sg2-dasgupta")
        Radial grid transformation.
    atom_radii: str (default "expected")
        Atom radii to use.
    multiatoms_scheme: str (default "becke")
        Multiatoms scheme to use.
    truncate: Optional[str] (default "dasgupta")
        Truncation rule to use.
    dtype: torch.dtype (default _dtype)
        Data type to use.
    device: torch.device (default _device)
        Device to use.

    Returns
    -------
    BaseGrid
        Molecular Grid.

    """
    # atompos: (natoms, ndim)
    assert atompos.ndim == 2
    assert atompos.shape[-2] == len(atomzs)

    # convert the atomzs to a list of integers
    if isinstance(atomzs, torch.Tensor):
        assert atomzs.ndim == 1
        atomzs_list = [a.item() for a in atomzs]
    else:
        atomzs_list = list(atomzs)

    # get the atom radii list
    atom_radii_options: Mapping[str, Union[List[float]]] = {
        "expected": atom_expected_radii,
        "bragg": atom_bragg_radii,
    }
    atom_radii_list = get_option("atom radii", atom_radii, atom_radii_options)
    atomradii = torch.tensor([atom_radii_list[atomz] for atomz in atomzs_list],
                             dtype=dtype,
                             device=device)

    # construct the radial grid transformation as a function of atom z
    radgrid_tf_options = {
        "sg2-dasgupta":
            lambda atz: DE2Transformation(alpha=__sg2_dasgupta_alphas[atz],
                                          rmin=1e-7,
                                          rmax=15 * atom_radii_list[atz]),
        "sg3-dasgupta":
            lambda atz: DE2Transformation(alpha=__sg3_dasgupta_alphas[atz],
                                          rmin=1e-7,
                                          rmax=15 * atom_radii_list[atz]),
        "logm3":
            lambda atz: LogM3Transformation(ra=atom_radii_list[atz]),
        "treutlerm4":
            lambda atz: TreutlerM4Transformation(xi=__treutler_xi[atz],
                                                 alpha=0.6),
    }
    radgrid_tf = get_option("radial grid transformation", radgrid_transform,
                            radgrid_tf_options)

    # get the precisions
    if isinstance(nang, int):
        prec: Union[int, Callable[[int],
                                  int]] = get_option("number of angular points",
                                                     nang, __nang2prec)
    else:

        def _prec_fcn(atz: int) -> int:
            assert callable(nang)
            return get_option("number of angular points", nang(atz),
                              __nang2prec)

        prec = _prec_fcn

    # wrap up a function to get the nr
    def _get_nr(nr: Union[int, Callable[[int], int]], atz: int) -> int:
        """Get the number of radial points for the atom z.

        Parameters
        ----------
        nr: Union[int, Callable[[int], int]]
            Number of radial points.
        atz: int
            Atomic Number of the atom.

        Returns
        -------
        int
            Number of radial points.

        """
        if isinstance(nr, int):
            return nr
        else:
            return nr(atz)

    # get the truncation rule as a function to avoid unnecessary evaluation
    trunc_options = {
        "dasgupta":
            lambda: DasguptaTrunc(nr),
        "nwchem":
            lambda: NWChemTrunc(atom_radii_list,
                                prec,
                                list(__nang2prec.values()),
                                dtype=dtype,
                                device=device),
        "no":
            lambda: NoTrunc(),
    }
    truncate_str = truncate if truncate is not None else "no"
    trunc = get_option("truncation rule", truncate_str, trunc_options)()

    sphgrids: List[BaseGrid] = []
    sphgrids_dict: Dict[int, BaseGrid] = {}
    for atz in atomzs_list:
        if atz in sphgrids_dict:
            sphgrids.append(sphgrids_dict[atz])
            continue

        nr_value = _get_nr(nr, atz)
        radgrid = RadialGrid(nr_value,
                             grid_integrator=radgrid_generator,
                             grid_transform=radgrid_tf(atz),
                             dtype=dtype,
                             device=device)
        if trunc.to_truncate(atz):
            rad_slices = trunc.rad_slices(atz, radgrid)
            radgrids: List[BaseGrid] = [radgrid[sl] for sl in rad_slices]
            precs = trunc.precs(atz, radgrid)
            sphgrid = TruncatedLebedevGrid(radgrids, precs)  # type: ignore
        else:
            sphgrid = LebedevGrid(
                radgrid,
                prec=_get_nr(
                    prec,  # type: ignore
                    atz))
        sphgrids_dict[atz] = sphgrid
        sphgrids.append(sphgrid)

    # get the multi atoms grid
    # the values are a function to avoid constructing it unnecessarily
    if lattice is None:
        multiatoms_options: Mapping[str, Callable[[], BaseGrid]] = {
            "becke":
                lambda: BeckeGrid(
                    sphgrids,  # type: ignore
                    atompos,
                    atomradii=atomradii),
            "treutler":
                lambda: BeckeGrid(
                    sphgrids,  # type: ignore
                    atompos,
                    atomradii=atomradii,
                    ratom_adjust="treutler"),
        }
    else:
        assert isinstance(lattice, Lattice)
        multiatoms_options = {
            "becke":
                lambda: PBCBeckeGrid(
                    sphgrids,  # type: ignore
                    atompos,
                    lattice=lattice),
            "treutler":
                lambda: PBCBeckeGrid(
                    sphgrids,  # type: ignore
                    atompos,
                    lattice=lattice,
                    ratom_adjust="treutler"),
        }
    grid = get_option("multiatoms scheme", multiatoms_scheme,
                      multiatoms_options)()
    return grid


def get_predefined_grid(grid_inp: Union[int, str],
                        atomzs: Union[List[int], torch.Tensor],
                        atompos: torch.Tensor,
                        *,
                        lattice: Optional[Lattice] = None,
                        dtype: torch.dtype = _dtype,
                        device: torch.device = _device) -> BaseGrid:
    """Returns the predefined grid object given the grid name.

    Examples
    --------
    >>> from deepchem.utils.dft_utils import get_predefined_grid
    >>> grid = get_predefined_grid(3, torch.tensor([2]), torch.tensor([[0,0,0]], dtype=torch.float64))
    >>> grid.get_rgrid().shape
    torch.Size([8608, 3])

    Parameters
    ----------
    grid_inp: Union[int, str]
        Grid name or number of angular points.
    atomzs: Union[List[int], torch.Tensor]
        Atomic Number of the atoms in the molecule.
    atompos: torch.Tensor
        Positions of the atoms in 3d. (natoms, ndim)
    lattice: Optional[Lattice] (Default None)
        Lattice Object descibing periodicity.
    dtype: torch.dtype (default _dtype)
        Data type to use.
    device: torch.device (default _device)
        Device to use.

    Returns
    -------
    BaseGrid
        Molecular Grid.

    """
    if isinstance(grid_inp, str):
        if grid_inp == "sg2":
            return get_grid(
                atomzs,
                atompos,
                lattice=lattice,
                nr=75,
                nang=302,
                radgrid_generator="uniform",
                radgrid_transform="sg2-dasgupta",
                # using expected from de2 ref: DOI 10.1007/s00214-012-1169-z
                atom_radii="expected",
                multiatoms_scheme="becke",
                truncate="dasgupta",
                dtype=dtype,
                device=device)
        elif grid_inp == "sg3":
            return get_grid(
                atomzs,
                atompos,
                lattice=lattice,
                nr=99,
                nang=590,
                radgrid_generator="uniform",
                radgrid_transform="sg3-dasgupta",
                # using expected from de2 ref: DOI 10.1007/s00214-012-1169-z
                atom_radii="expected",
                multiatoms_scheme="becke",
                truncate="dasgupta",
                dtype=dtype,
                device=device)
        else:
            raise ValueError(f"Unknown grid name: {grid_inp}")
    elif isinstance(grid_inp, int):
        # the numbers are taken from pyscf
        # https://github.com/pyscf/pyscf/blob/21a0b8d19ec58d8e70901c77783fa72ac47c1745/pyscf/dft/gen_grid.py#L596
        # period:     1,   2,   3,   4,   5,   6,   7    # level
        nr_list = (
            (10, 15, 20, 30, 35, 40, 50),  # 0
            (30, 40, 50, 60, 65, 70, 75),  # 1
            (40, 60, 65, 75, 80, 85, 90),  # 2
            (50, 75, 80, 90, 95, 100, 105),  # 3
            (60, 90, 95, 105, 110, 115, 120),  # 4
            (70, 105, 110, 120, 125, 130, 135),  # 5
            (80, 120, 125, 135, 140, 145, 150),  # 6
            (90, 135, 140, 150, 155, 160, 165),  # 7
            (100, 150, 155, 165, 170, 175, 180),  # 8
            (200, 200, 200, 200, 200, 200, 200),  # 9
        )

        # period:        1,    2,    3,    4,    5,    6,    7    # level
        nang_list = [
            [50, 86, 110, 110, 110, 110, 110],  # 0
            [110, 194, 194, 194, 194, 194, 194],  # 1
            [194, 302, 302, 302, 302, 302, 302],  # 2
            [302, 302, 434, 434, 434, 434, 434],  # 3
            [434, 590, 590, 590, 590, 590, 590],  # 4
            [590, 770, 770, 770, 770, 770, 770],  # 5
            [770, 974, 974, 974, 974, 974, 974],  # 6
            [974, 1202, 1202, 1202, 1202, 1202, 1202],  # 7
            [1202, 1202, 1202, 1202, 1202, 1202, 1202],  # 8
            [1454, 1454, 1454, 1454, 1454, 1454, 1454],  # 9
        ]

        nr_list2 = nr_list[grid_inp]
        nang_list2 = nang_list[grid_inp]

        def get_nr(atz: int) -> int:
            period = get_period(atz)
            return nr_list2[period - 1]

        def get_nang(atz: int) -> int:
            period = get_period(atz)
            return nang_list2[period - 1]

        return get_grid(atomzs,
                        atompos,
                        lattice=lattice,
                        nr=get_nr,
                        nang=get_nang,
                        radgrid_generator="chebyshev2",
                        radgrid_transform="treutlerm4",
                        atom_radii="bragg",
                        multiatoms_scheme="treutler",
                        truncate="nwchem",
                        dtype=dtype,
                        device=device)
    else:
        raise TypeError("Unknown type of grid_inp: %s" % type(grid_inp))
