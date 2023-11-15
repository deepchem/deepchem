import os
import torch
from typing import List
from deepchem.utils.dft_utils import CGTOBasis

__all__ = ["loadbasis"]

_dtype = torch.double
_device = torch.device("cpu")


def loadbasis(cmd: str,
              dtype: torch.dtype = _dtype,
              device: torch.device = _device,
              requires_grad: bool = False) -> List[CGTOBasis]:
    """Load basis from a file and return the list of CGTOBasis.

    Examples
    --------
    >>> basis = loadbasis("7:3-21G")
    >>> basis[0]
    CGTOBasis(angmom=0, alphas=tensor([242.7660,  36.4851,   7.8145], dtype=torch.float64), coeffs=tensor([ 9.3022, 13.2380,  8.3428], dtype=torch.float64), normalized=True)

    Parameters
    ----------
    cmd: str
        This can be a file path where the basis is stored or a
        string in format ``"atomz:basis"``, e.g. ``"1:6-311++G**"``.
    dtype: torch.dtype
        Tensor data type for ``alphas`` and ``coeffs`` of the GTO basis
    device: torch.device
        Tensor device for ``alphas`` and ``coeffs``
    requires_grad: bool
        If ``True``, the ``alphas`` and ``coeffs`` tensors become differentiable

    Returns
    -------
    list of CGTOBasis
        List of GTO basis loaded from the given file

    """
    res = []
    if not os.path.exists(cmd):
        file = _get_basis_file(cmd)
    else:
        file = cmd

    # read the content
    with open(file, "r") as f:
        lines = f.read().split("\n")

    # skip the header
    while True:
        line = lines.pop(0)
        if line == "":
            continue
        if line.startswith("!"):
            continue
        break

    # now it is at the orbital description
    while len(lines) > 0:
        line = lines.pop(0)
        if line.startswith("**"):
            break
        desc = line.split()
        nlines = int(desc[1])
        if nlines == 0:
            raise RuntimeError("Zero line on basis %s" % file)

        # read the exponents and the coefficients
        alphas = []
        coeffsT = []
        for i in range(nlines):
            alphacoeff = [_read_float(f) for f in lines.pop(0).split()]
            alphas.append(alphacoeff[0])
            coeffsT.append(alphacoeff[1:])
        # coeffsT: list with shape (nbasis, ncontr)
        # coeffs: list with shape (ncontr, nbasis)
        coeffs = list(zip(*coeffsT))
        ncoeffs = len(coeffs)
        angmoms = _expand_angmoms(desc[0], ncoeffs)

        # convert to tensor
        alpha = torch.tensor(alphas,
                             dtype=dtype,
                             device=device,
                             requires_grad=requires_grad)
        for i in range(ncoeffs):
            coeff = torch.tensor(coeffs[i],
                                 dtype=dtype,
                                 device=device,
                                 requires_grad=requires_grad)
            basis = CGTOBasis(angmom=angmoms[i], alphas=alpha, coeffs=coeff)
            basis.wfnormalize_()
            res.append(basis)
    return res


def _read_float(s: str) -> float:
    """Read a float from a string, replacing D with E.

    Parameters
    ----------
    s: str
        String to be read.

    Returns
    -------
    float
        Float value read from the string.

    """
    s = s.replace("D", "E")
    return float(s)


def _get_basis_file(cmd: str) -> str:
    """parse the string command, check if the basis has already been downloaded
    (download if not), and return the file name

    Parameters
    ----------
    cmd: str
        String in format ``"atomz:basis"``, e.g. ``"1:6-311++G**"``.

    Returns
    -------
    str
        File path to the basis file

    """

    # parse to get the atomz and the basisname
    atomz_str, raw_basisname = cmd.split(":")
    raw_basisname = raw_basisname.strip()
    atomz = int(atomz_str)

    # get the path to the database
    basisname = _normalize_basisname(raw_basisname)
    thisdir = os.path.dirname(os.path.realpath(__file__))
    fname = "%02d.gaussian94" % atomz
    fdir = os.path.join(thisdir, ".database", basisname)
    fpath = os.path.join(fdir, fname)

    # if the file does not exist, download it
    if not os.path.exists(fpath):
        print(
            "The %s basis for atomz %d does not exist, but we will download it"
            % (raw_basisname, atomz))
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        _download_basis(fpath, atomz, raw_basisname)

    return fpath


def _normalize_basisname(basisname: str) -> str:
    """Convert the basisname into a normalized form.

    Examples
    --------
    >>> _normalize_basisname("6-311++G**")
    '6-311ppgss'

    Parameters
    ----------
    basisname: str
        Basis name to be normalized, e.g. ``"6-311++G**"``.

    Returns
    -------
    str
        Normalized basis name, e.g. ``"6-311ppgss"``.

    """
    b = basisname.lower()
    b = b.replace("+", "p")
    b = b.replace("*", "s")
    b = b.replace("(", "_")
    b = b.replace(")", "_")
    b = b.replace(",", "_")
    return b


def _download_basis(fname: str, atomz: int, basisname: str) -> None:
    """Download the basis from the basis set exchange and save it to a file.

    Parameters
    ----------
    fname: str
        File path to save the basis.
    atomz: int
        Atomic number of the atom.
    basisname: str
        Basis name to be downloaded, e.g. ``"6-311++G**"``.

    """
    import basis_set_exchange as bse
    s = bse.get_basis(basisname, elements=[atomz], fmt="gaussian94")
    with open(fname, "w") as f:
        f.write(s)
    print("Downloaded to %s" % fname)


def _expand_angmoms(s: str, n: int) -> List[int]:
    """convert the angular momentum characters into angmom and returns a list
    of n integer containing the angular momentums.

    Examples
    --------
    >>> _expand_angmoms("SP", 2)
    [0, 1]

    Parameters
    ----------
    s: str
        String containing the angular momentum characters, e.g. ``"SP"``.
    n: int
        Number of angular momentum characters in ``s``.

    Returns
    -------
    list of int
        List of angular momentums, e.g. ``[0, 1]``.

    """
    if len(s) == n:
        pass
    elif n % len(s) == 0:
        s = s * (n // len(s))
    else:
        raise RuntimeError(
            "Do not know how to read orbital %s with %d coefficient columns" %
            (s, n))
    s = s.lower()
    spdfmap = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
        "h": 5,
        "i": 6,
    }
    angmoms = [spdfmap[c] for c in s]
    return angmoms
