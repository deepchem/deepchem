from typing import Union, Tuple
import torch
from deepchem.utils.dft_utils import AtomZsType, AtomPosType
from deepchem.utils import get_atomz

__all__ = ["parse_moldesc"]


def parse_moldesc(
    moldesc: Union[str, Tuple[AtomZsType, AtomPosType]],
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse the string of molecular descriptor and returns tensors of atomzs and
    atom positions.
    Examples
    --------
    >>> from deepchem.utils.dft_utils import parse_moldesc
    >>> system = {
    ...     'type': 'mol',
    ...     'kwargs': {
    ...         'moldesc': 'H 0.86625 0 0; F -0.86625 0 0',
    ...         'basis': '6-311++G(3df,3pd)'
    ...     }
    ... }
    >>> atomzs, atomposs = parse_moldesc(system["kwargs"]["moldesc"])
    >>> atomzs
    tensor([1., 9.], dtype=torch.float64)
    >>> atomposs
    tensor([[ 0.8662,  0.0000,  0.0000],
            [-0.8662,  0.0000,  0.0000]], dtype=torch.float64)

    Parameters
    ----------
    moldesc: Union[str, Tuple[AtomZsType, AtomPosType]]
        String that describes the system, e.g. ``"H -1 0 0; H 1 0 0"`` for H2
        molecule separated by 2 Bohr.
    dtype: torch.dtype (default torch.float64)
        The datatype of the returned atomic positions.
    device: torch.device (default torch.device('cpu'))
        The device to store the returned tensors.
    Returns
    -------
    atomzs: torch.Tensor
        The tensor of atomzs [Atom Number].
    atompos: torch.Tensor
        The tensor of atomic positions [Bohr].
    """
    if isinstance(moldesc, str):
        # TODO: use regex!
        elmts = [[
            get_atomz(c.strip()) if i == 0 else float(c.strip())
            for i, c in enumerate(line.split())
        ]
                 for line in moldesc.split(";")]
        atomzs = torch.tensor([line[0] for line in elmts], device=device)
        atompos = torch.tensor([line[1:] for line in elmts],
                               dtype=dtype,
                               device=device)

    else:  # tuple of atomzs, atomposs
        atomzs_raw, atompos_raw = moldesc
        assert len(atomzs_raw) == len(
            atompos_raw), "Mismatch length of atomz and atompos"
        assert len(atomzs_raw) > 0, "Empty atom list"

        # convert the atomz to tensor
        if not isinstance(atomzs_raw, torch.Tensor):
            atomzs = torch.tensor([get_atomz(at) for at in atomzs_raw],
                                  device=device)
        else:
            atomzs = atomzs_raw.to(device)  # already a tensor

        # convert the atompos to tensor
        if not isinstance(atompos_raw, torch.Tensor):
            atompos = torch.as_tensor(atompos_raw, dtype=dtype, device=device)
        else:
            atompos = atompos_raw.to(dtype).to(device)  # already a tensor

    # convert to dtype if atomzs is a floating point tensor, not an integer tensor
    atomzs = atomzs.to(dtype)

    return atomzs, atompos
