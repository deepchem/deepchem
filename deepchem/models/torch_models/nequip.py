"""Native PyTorch implementation of the original NequIP backbone.

The architecture follows Batzner et al., *E(3)-Equivariant Graph Neural
Networks for Data-Efficient and Accurate Interatomic Potentials*, in
particular equations 4--8 and the interaction block in Figure 2.

The implementation is organized in the same order as the model computation:

1. represent node features by O(3) irreducible representations;
2. expand edge distances with a Bessel basis and polynomial cutoff;
3. build equivariant messages from spherical harmonics and tensor products;
4. apply interaction blocks, scalar readout, and graph-wise energy summation;
5. differentiate total energy with respect to positions when forces are needed.

For an irrep ``(l, parity)``, features have shape
``(num_atoms, multiplicity, 2*l + 1)``.  Parity is ``+1`` for even irreps and
``-1`` for odd irreps.  This explicit parity bookkeeping complements
DeepChem's :class:`Fiber`, which records angular degree but not parity.
"""

import math
from dataclasses import dataclass
from typing import cast, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from deepchem.feat.graph_data import BatchGraphData, GraphData
from deepchem.models.torch_models.layers import Fiber, SE3SelfInteraction
from deepchem.utils.equivariance_utils import (basis_transformation_Q_J,
                                               get_spherical_from_cartesian,
                                               precompute_sh)

# O(3) representations and selection rules
# -----------------------------------------

# An O(3) irrep is identified by angular degree l and parity (+1 even, -1 odd).
_Irrep = Tuple[int, int]
_Features = Dict[_Irrep, torch.Tensor]

# Unit-second-moment factors used by the paper-era NequIP gate.
_SILU_NORMALIZATION = 1.6791767923989418
_TANH_NORMALIZATION = 1.5937334472592695


def _normalized_silu(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU normalized to unit second moment."""
    return F.silu(x) * _SILU_NORMALIZATION


def _normalized_tanh(x: torch.Tensor) -> torch.Tensor:
    """Apply tanh normalized to unit second moment."""
    return torch.tanh(x) * _TANH_NORMALIZATION


def _irrep_name(irrep: _Irrep) -> str:
    """Return a stable module key for an ``(l, parity)`` pair."""
    return f"l{irrep[0]}_{'e' if irrep[1] == 1 else 'o'}"


def _filter_parity(filter_degree: int) -> int:
    """Return the natural O(3) parity of a spherical harmonic."""
    return 1 if filter_degree % 2 == 0 else -1


def _tensor_product_path_allowed(input_irrep: _Irrep, filter_degree: int,
                                 output_irrep: _Irrep) -> bool:
    """Test the triangle and parity selection rules from NequIP equation 7."""
    l_in, parity_in = input_irrep
    l_out, parity_out = output_irrep
    return (abs(l_in - filter_degree) <= l_out <= l_in + filter_degree and
            parity_out == parity_in * _filter_parity(filter_degree))


def _path_exists(input_irreps: Mapping[_Irrep, int], output_irrep: _Irrep,
                 l_max: int) -> bool:
    """Return whether any input and edge irrep can produce ``output_irrep``."""
    return any(
        _tensor_product_path_allowed(input_irrep, filter_degree, output_irrep)
        for input_irrep in input_irreps
        for filter_degree in range(l_max + 1))


def _reachable_hidden_irreps(input_irreps: Mapping[_Irrep, int], l_max: int,
                             hidden_channels: int) -> Dict[_Irrep, int]:
    """Return outputs reachable using edge harmonics through degree ``l_max``."""
    candidates = [
        (degree, parity) for degree in range(l_max + 1) for parity in (1, -1)
    ]
    return {
        irrep: hidden_channels
        for irrep in candidates
        if _path_exists(input_irreps, irrep, l_max)
    }


# Radial representation
# ---------------------


class _BesselBasis(nn.Module):
    """Trainable Bessel basis from NequIP equation 6.

    Distances of shape ``(num_edges,)`` map to ``(num_edges, num_bessel)``.
    """

    def __init__(self, cutoff: float, num_bessel: int) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        frequencies = torch.arange(
            1, num_bessel + 1, dtype=torch.get_default_dtype()) * math.pi
        self.frequencies = nn.Parameter(frequencies)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # sin(w r / rc) / r = (w / rc) sinc(w r / (pi rc)); the latter is
        # finite and differentiable at r=0.
        scaled = distances.unsqueeze(-1) * self.frequencies / self.cutoff
        sin_over_r = (self.frequencies / self.cutoff) * torch.sinc(
            scaled / math.pi)
        return (2.0 / self.cutoff) * sin_over_r


class _PolynomialCutoff(nn.Module):
    """Polynomial cutoff envelope used by the original NequIP model.

    The envelope and its first two derivatives vanish at the cutoff; values at
    and beyond the cutoff are set to zero.
    """

    def __init__(self, cutoff: float, p: int = 6) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.p = int(p)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        x = distances / self.cutoff
        p = self.p
        envelope = (1.0 - ((p + 1) * (p + 2) / 2.0) * x.pow(p) + p *
                    (p + 2) * x.pow(p + 1) - (p * (p + 1) / 2.0) * x.pow(p + 2))
        return envelope * (distances < self.cutoff).to(distances.dtype)


class _RadialMLP(nn.Module):
    """Map radial bases to independent tensor-product path weights."""

    def __init__(self, input_size: int, output_size: int, hidden_width: int,
                 hidden_depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_features = input_size
        for _ in range(hidden_depth):
            layers.append(_VarianceScaledLinear(in_features, hidden_width))
            layers.append(_NormalizedSiLU())
            in_features = hidden_width
        layers.append(_VarianceScaledLinear(in_features, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, radial_basis: torch.Tensor) -> torch.Tensor:
        return self.layers(radial_basis)


class _NormalizedSiLU(nn.Module):
    """Module form of the normalized SiLU used by radial networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _normalized_silu(x)


class _VarianceScaledLinear(nn.Linear):
    """Bias-free linear map with NequIP's unit-variance initialization.

    Parameters are sampled from a standard normal distribution and divided by
    ``sqrt(in_features)`` during evaluation, matching the paper-era radial
    network parameterization.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(input_size, output_size, bias=False)
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight / math.sqrt(self.in_features))


# Equivariant feature operations
# ------------------------------


class _ParitySelfInteraction(nn.Module):
    """Mix channels within each ``(l, parity)`` using DeepChem's linear map.

    :class:`SE3SelfInteraction` performs the required channel transformation,
    shared across the ``2*l + 1`` components.  Separate modules for even and
    odd features preserve parity without changing DeepChem's ``Fiber`` API.
    """

    def __init__(self, input_irreps: Mapping[_Irrep, int],
                 output_irreps: Mapping[_Irrep, int]) -> None:
        super().__init__()
        self.input_irreps = dict(input_irreps)
        self.interactions = nn.ModuleDict()

        for parity in (1, -1):
            input_degrees = {
                degree: channels
                for (degree, irrep_parity), channels in input_irreps.items()
                if irrep_parity == parity
            }
            output_degrees = {
                degree: channels
                for (degree, irrep_parity), channels in output_irreps.items()
                if irrep_parity == parity
            }
            if not output_degrees:
                continue
            missing = set(output_degrees).difference(input_degrees)
            if missing:
                raise ValueError(
                    "An equivariant self-interaction cannot create a new irrep")
            self.interactions[str(parity)] = SE3SelfInteraction(
                Fiber(dictionary=input_degrees),
                Fiber(dictionary=output_degrees))

    def forward(self, features: _Features) -> _Features:
        output: _Features = {}
        for parity in (1, -1):
            key = str(parity)
            if key not in self.interactions:
                continue
            parity_features = {
                str(degree): features[(degree, parity)]
                for degree, irrep_parity in self.input_irreps
                if irrep_parity == parity and (degree, parity) in features
            }
            transformed = self.interactions[key](parity_features)
            output.update({(int(degree), parity): value
                           for degree, value in transformed.items()})
        return output


class _SpeciesSelfConnection(nn.Module):
    """Species-conditioned equivariant residual used by NequIP.

    A one-hot species attribute is a collection of even scalars.  Its fully
    connected tensor product with an irrep is therefore exactly a
    species-indexed channel matrix applied identically to every m component.
    """

    def __init__(self, input_irreps: Mapping[_Irrep, int],
                 output_irreps: Mapping[_Irrep, int], num_species: int) -> None:
        super().__init__()
        self.num_species = num_species
        self.weights = nn.ParameterDict()
        for irrep, output_channels in output_irreps.items():
            if irrep not in input_irreps:
                continue
            input_channels = input_irreps[irrep]
            weight = torch.randn(num_species, output_channels, input_channels)
            self.weights[_irrep_name(irrep)] = nn.Parameter(weight)

    def forward(self, features: _Features, species: torch.Tensor) -> _Features:
        output: _Features = {}
        for irrep, feature in features.items():
            key = _irrep_name(irrep)
            if key not in self.weights:
                continue
            # The one-hot species axis contributes to the tensor-product fan-in.
            normalization = math.sqrt(feature.shape[1] * self.num_species)
            weight = self.weights[key][species] / normalization
            output[irrep] = torch.einsum("noi,nim->nom", weight, feature)
        return output


@dataclass(frozen=True)
class _TensorProductPath:
    """Metadata for one input, edge-filter, and output irrep coupling."""

    input_irrep: _Irrep
    filter_degree: int
    output_irrep: _Irrep
    weight_start: int
    weight_stop: int
    basis_name: str


class _TensorProductConvolution(nn.Module):
    """Evaluate NequIP equation 8 with distinct channels for each path.

    Each feature tensor has shape ``(num_atoms, multiplicity, 2*l + 1)``.
    Radial weights are generated per edge, and compatible paths remain
    separate until the post-convolution self-interaction.
    """

    def __init__(self, input_irreps: Mapping[_Irrep, int],
                 output_irreps: Mapping[_Irrep, int], l_max: int,
                 num_bessel: int, radial_hidden_width: int,
                 radial_hidden_depth: int, avg_num_neighbors: float) -> None:
        super().__init__()
        self.output_irreps = dict(output_irreps)
        self.avg_num_neighbors = float(avg_num_neighbors)
        self.paths: List[_TensorProductPath] = []
        basis_names: Dict[Tuple[int, int, int], str] = {}
        mid_irreps: Dict[_Irrep, int] = {irrep: 0 for irrep in output_irreps}
        weight_offset = 0

        # One path corresponds to one e3nn ``uvu`` instruction: the input
        # multiplicity is preserved, while the edge harmonic has multiplicity
        # one.  It therefore consumes one radial weight per input channel.
        # Paths with the same output irrep remain separate in ``mid_irreps``.
        for input_irrep, input_channels in input_irreps.items():
            l_in, _ = input_irrep
            for filter_degree in range(l_max + 1):
                for output_irrep in output_irreps:
                    if not _tensor_product_path_allowed(
                            input_irrep, filter_degree, output_irrep):
                        continue
                    l_out, _ = output_irrep
                    basis_key = (filter_degree, l_in, l_out)
                    if basis_key not in basis_names:
                        basis_name = f"q_{filter_degree}_{l_in}_{l_out}"
                        q_matrix = basis_transformation_Q_J(
                            filter_degree, l_in, l_out).to(
                                dtype=torch.get_default_dtype()).T.contiguous()
                        self.register_buffer(basis_name, q_matrix)
                        basis_names[basis_key] = basis_name
                    self.paths.append(
                        _TensorProductPath(input_irrep=input_irrep,
                                           filter_degree=filter_degree,
                                           output_irrep=output_irrep,
                                           weight_start=weight_offset,
                                           weight_stop=weight_offset +
                                           input_channels,
                                           basis_name=basis_names[basis_key]))
                    weight_offset += input_channels
                    mid_irreps[output_irrep] += input_channels

        if weight_offset == 0:
            raise ValueError(
                "The requested irreps have no tensor-product paths")
        self.radial_mlp = _RadialMLP(num_bessel, weight_offset,
                                     radial_hidden_width, radial_hidden_depth)
        self.post_self_interaction = _ParitySelfInteraction(
            mid_irreps, output_irreps)

    def forward(self, features: _Features, radial_basis: torch.Tensor,
                spherical_harmonics: Mapping[int, torch.Tensor],
                center: torch.Tensor, neighbor: torch.Tensor,
                num_nodes: int) -> _Features:
        """Convolve neighbor features along all allowed tensor-product paths.

        For each edge and path, the contraction implements

        ``neighbor feature x Y_l(edge direction) x radial weight``.

        The Clebsch--Gordan basis maps the input components to output
        components.  Messages are summed into central atoms, divided by
        ``sqrt(avg_num_neighbors)``, concatenated by output irrep, and mixed by
        the post-convolution self-interaction.

        ``features[(l, p)]`` has shape ``(N, C, 2*l + 1)``;
        ``radial_basis`` has shape ``(E, num_bessel)``; and each spherical
        harmonic tensor is indexed by the same ``E`` directed edges.
        """
        weights = self.radial_mlp(radial_basis)
        path_outputs: Dict[_Irrep, List[torch.Tensor]] = {
            irrep: [] for irrep in self.output_irreps
        }

        for path in self.paths:
            l_in, _ = path.input_irrep
            l_out, _ = path.output_irrep
            q_matrix = getattr(self, path.basis_name)
            angular = torch.matmul(spherical_harmonics[path.filter_degree],
                                   q_matrix)
            # Convert DeepChem's integral-normalized SH and unit-norm Q_J to
            # the component normalization used by the original NequIP model.
            angular = angular * math.sqrt(4 * math.pi * (2 * l_out + 1))
            angular = angular.view(-1, 2 * l_out + 1, 2 * l_in + 1)
            neighbor_features = features[path.input_irrep][neighbor]
            message = torch.einsum("eoi,eui->euo", angular, neighbor_features)
            path_weight = weights[:, path.weight_start:path.weight_stop]
            message = message * path_weight.unsqueeze(-1)
            aggregated = message.new_zeros(
                (num_nodes, message.shape[1], message.shape[2]))
            aggregated.index_add_(0, center, message)
            path_outputs[path.output_irrep].append(aggregated)

        normalization = math.sqrt(self.avg_num_neighbors)
        concatenated = {
            irrep: torch.cat(outputs, dim=1) / normalization
            for irrep, outputs in path_outputs.items()
        }
        return self.post_self_interaction(concatenated)


class _EquivariantGate(nn.Module):
    """Apply parity-aware scalar activations and higher-irrep gates.

    Even and odd scalars use normalized SiLU and tanh, respectively.  Each
    higher-irrep channel is multiplied by one normalized even scalar gate.
    """

    def __init__(self, output_irreps: Mapping[_Irrep, int]) -> None:
        super().__init__()
        self.output_irreps = dict(output_irreps)
        self.scalar_irreps = {
            irrep: channels
            for irrep, channels in output_irreps.items()
            if irrep[0] == 0
        }
        self.gated_irreps = {
            irrep: channels
            for irrep, channels in output_irreps.items()
            if irrep[0] > 0
        }
        self.num_even_scalars = self.scalar_irreps.get((0, 1), 0)
        self.num_gates = sum(self.gated_irreps.values())

    @property
    def input_irreps(self) -> Dict[_Irrep, int]:
        """Return convolution outputs, including scalar gate channels."""
        irreps = dict(self.output_irreps)
        irreps[(0, 1)] = self.num_even_scalars + self.num_gates
        return irreps

    def forward(self, features: _Features) -> _Features:
        output: _Features = {}
        even_and_gates = features[(0, 1)]
        if self.num_even_scalars:
            output[(0, 1)] = _normalized_silu(
                even_and_gates[:, :self.num_even_scalars])
        if (0, -1) in self.scalar_irreps:
            output[(0, -1)] = _normalized_tanh(features[(0, -1)])

        gates = _normalized_silu(even_and_gates[:, self.num_even_scalars:, 0])
        offset = 0
        for irrep, channels in self.gated_irreps.items():
            gate = gates[:, offset:offset + channels].unsqueeze(-1)
            output[irrep] = features[irrep] * gate
            offset += channels
        return output


# Interaction block and energy model
# ----------------------------------


class NequIPInteractionBlock(nn.Module):
    """One interaction block from the original NequIP architecture.

    The operation order follows Figure 2 of the paper::

        residual = species self connection(x)
        x = pre self-interaction(x)
        x = tensor-product convolution(x)  # includes post self-interaction
        output = gate(x + residual)

    The convolution includes the post self-interaction that mixes distinct
    tensor-product paths.  Paths satisfy the angular-momentum triangle and
    parity selection rules.

    Feature tensors are keyed by ``(l, parity)`` and have shape
    ``(num_atoms, multiplicity, 2*l + 1)``.  Messages read features from the
    neighbor endpoint and are summed into the center endpoint before division
    by ``sqrt(avg_num_neighbors)``.

    Parameters
    ----------
    input_irreps : mapping
        Multiplicity of each input ``(l, parity)`` irrep.
    hidden_channels : int
        Multiplicity assigned to each reachable output irrep.
    num_species : int
        Size of the atomic-number embedding table.
    l_max : int
        Maximum degree of node and edge spherical harmonics.
    num_bessel : int
        Number of radial Bessel functions.
    radial_hidden_width : int
        Width of each radial MLP hidden layer.
    radial_hidden_depth : int
        Number of radial MLP hidden layers.
    avg_num_neighbors : float
        Neighbor statistic used to normalize aggregated messages.

    References
    ----------
    .. [1] Batzner, S. et al. E(3)-equivariant graph neural networks for
       data-efficient and accurate interatomic potentials. Nat Commun 13,
       2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
    """

    def __init__(self, input_irreps: Mapping[_Irrep, int], hidden_channels: int,
                 num_species: int, l_max: int, num_bessel: int,
                 radial_hidden_width: int, radial_hidden_depth: int,
                 avg_num_neighbors: float) -> None:
        super().__init__()
        if avg_num_neighbors <= 0:
            raise ValueError("avg_num_neighbors must be positive")
        self.output_irreps = _reachable_hidden_irreps(input_irreps, l_max,
                                                      hidden_channels)
        self.gate = _EquivariantGate(self.output_irreps)
        # The convolution must also emit one even scalar for every gated
        # higher-irrep channel; the gate removes these auxiliary scalars.
        convolution_irreps = self.gate.input_irreps
        self.pre_self_interaction = _ParitySelfInteraction(
            input_irreps, input_irreps)
        self.convolution = _TensorProductConvolution(
            input_irreps=input_irreps,
            output_irreps=convolution_irreps,
            l_max=l_max,
            num_bessel=num_bessel,
            radial_hidden_width=radial_hidden_width,
            radial_hidden_depth=radial_hidden_depth,
            avg_num_neighbors=avg_num_neighbors)
        self.self_connection = _SpeciesSelfConnection(input_irreps,
                                                      convolution_irreps,
                                                      num_species)

    def forward(self, features: _Features, radial_basis: torch.Tensor,
                spherical_harmonics: Mapping[int, torch.Tensor],
                edge_index: torch.Tensor, species: torch.Tensor) -> _Features:
        """Apply one interaction block.

        Parameters
        ----------
        features : dict
            Input tensors keyed by ``(l, parity)``.
        radial_basis : torch.Tensor
            Cutoff-weighted radial basis, shape ``(num_edges, num_bessel)``.
        spherical_harmonics : mapping
            Edge spherical harmonics keyed by degree.
        edge_index : torch.Tensor
            Directed edges with shape ``(2, num_edges)``.
        species : torch.Tensor
            Atomic-number indices with shape ``(num_atoms,)``.

        Returns
        -------
        dict
            Gated output tensors keyed by ``(l, parity)``.
        """
        # Row 0 indexes receiving centers; row 1 indexes source neighbors.
        center, neighbor = edge_index
        residual = self.self_connection(features, species)
        convolved = self.convolution(self.pre_self_interaction(features),
                                     radial_basis, spherical_harmonics, center,
                                     neighbor, species.shape[0])
        for irrep, value in residual.items():
            convolved[irrep] = convolved[irrep] + value
        return self.gate(convolved)


class NequIP(nn.Module):
    """Native nonperiodic NequIP energy backbone.

    Interaction blocks predict even scalar atomic energies, which are summed
    for each structure using ``GraphData.graph_index``.

    Parameters
    ----------
    cutoff : float
        Radius cutoff used by equation 6.  Input topology must use the same
        cutoff.
    num_species : int
        Atomic-number embedding size.  Atomic numbers are used directly as
        indices, so this must be larger than the largest supported atomic
        number.
    hidden_channels : int
        Multiplicity of every hidden ``(l, parity)`` feature type.
    num_interaction_blocks : int
        Number of NequIP interaction blocks.
    l_max : int
        Maximum node and edge spherical-harmonic degree.  Edge filters use
        degrees ``0..l_max`` with natural parity ``(-1)**l``.
    num_bessel : int
        Number of trainable Bessel frequencies.
    radial_hidden_width : int
        Width of each radial MLP hidden layer.
    radial_hidden_depth : int
        Number of SiLU hidden layers in each radial MLP.
    avg_num_neighbors : float
        Training-set neighbor statistic used for ``sqrt(avg)`` normalization.
    readout_hidden_channels : int
        Number of even scalar channels in the first output self-interaction.
    compute_forces : bool, optional
        If ``True``, return forces obtained from the energy gradient by default.

    Notes
    -----
    ``forward`` accepts a torch-converted :class:`GraphData` or
    :class:`BatchGraphData`.  Static edge vectors and distances are ignored:
    for an edge ``center -> neighbor`` geometry is recomputed as
    ``pos[neighbor] - pos[center]`` so forces remain connected to positions.
    ``node_features`` must contain one integer-valued atomic number per node;
    values are cast to integer indices before embedding because graph tensor
    conversion may produce floating-point features.

    References
    ----------
    .. [1] Batzner, S. et al. E(3)-equivariant graph neural networks for
       data-efficient and accurate interatomic potentials. Nat Commun 13,
       2453 (2022). https://doi.org/10.1038/s41467-022-29939-5
    """

    def __init__(self,
                 cutoff: float,
                 num_species: int = 119,
                 hidden_channels: int = 32,
                 num_interaction_blocks: int = 3,
                 l_max: int = 2,
                 num_bessel: int = 8,
                 radial_hidden_width: int = 64,
                 radial_hidden_depth: int = 2,
                 avg_num_neighbors: float = 1.0,
                 readout_hidden_channels: int = 16,
                 compute_forces: bool = False) -> None:
        super().__init__()
        integer_parameters = {
            "num_species": num_species,
            "hidden_channels": hidden_channels,
            "num_interaction_blocks": num_interaction_blocks,
            "num_bessel": num_bessel,
            "radial_hidden_width": radial_hidden_width,
            "radial_hidden_depth": radial_hidden_depth,
            "readout_hidden_channels": readout_hidden_channels,
        }
        if cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if l_max < 0:
            raise ValueError("l_max must be nonnegative")
        if avg_num_neighbors <= 0:
            raise ValueError("avg_num_neighbors must be positive")
        if any(value <= 0 for value in integer_parameters.values()):
            raise ValueError("NequIP size parameters must be positive")

        self.cutoff = float(cutoff)
        self.num_species = int(num_species)
        self.l_max = int(l_max)
        self.compute_forces = bool(compute_forces)
        self.embedding = nn.Embedding(num_species, hidden_channels)
        self.bessel_basis = _BesselBasis(cutoff, num_bessel)
        self.cutoff_envelope = _PolynomialCutoff(cutoff, p=6)

        input_irreps: Dict[_Irrep, int] = {(0, 1): hidden_channels}
        blocks: List[NequIPInteractionBlock] = []
        for _ in range(num_interaction_blocks):
            block = NequIPInteractionBlock(
                input_irreps=input_irreps,
                hidden_channels=hidden_channels,
                num_species=num_species,
                l_max=l_max,
                num_bessel=num_bessel,
                radial_hidden_width=radial_hidden_width,
                radial_hidden_depth=radial_hidden_depth,
                avg_num_neighbors=avg_num_neighbors)
            blocks.append(block)
            input_irreps = block.output_irreps
        self.interaction_blocks = nn.ModuleList(blocks)

        scalar_input = {(0, 1): input_irreps[(0, 1)]}
        scalar_hidden = {(0, 1): readout_hidden_channels}
        scalar_output = {(0, 1): 1}
        self.output_hidden = _ParitySelfInteraction(scalar_input, scalar_hidden)
        self.output_energy = _ParitySelfInteraction(scalar_hidden,
                                                    scalar_output)

    def _validate_graph(
        self, graph: Union[GraphData, BatchGraphData]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate and return species, edges, and graph membership."""
        node_features = graph.node_features
        positions = graph.node_pos_features
        edge_index = graph.edge_index
        graph_index = getattr(graph, "graph_index", None)
        if not all(
                isinstance(value, torch.Tensor)
                for value in (node_features, positions, edge_index)):
            raise TypeError("NequIP requires a torch-converted GraphData")
        node_features = cast(torch.Tensor, node_features)
        positions = cast(torch.Tensor, positions)
        edge_index = cast(torch.Tensor, edge_index)
        if node_features.ndim != 2 or node_features.shape[1] != 1:
            raise ValueError("node_features must have shape (num_atoms, 1)")
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("node_pos_features must have shape (num_atoms, 3)")
        if not positions.is_floating_point():
            raise TypeError("node positions must be floating point")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        species_values = node_features[:, 0]
        if species_values.is_floating_point() and not torch.equal(
                species_values, species_values.round()):
            raise ValueError("atomic numbers must be integer-valued")
        species = species_values.long()
        if torch.any(species < 0) or torch.any(species >= self.num_species):
            raise ValueError("atomic number is outside the embedding range")

        if graph_index is None:
            graph_index = torch.zeros(species.shape[0],
                                      dtype=torch.long,
                                      device=species.device)
        elif not isinstance(graph_index, torch.Tensor):
            raise TypeError("graph_index must be a torch tensor")
        else:
            graph_index = graph_index.long()
        if graph_index.shape != species.shape:
            raise ValueError("graph_index must have shape (num_atoms,)")
        edge_index = edge_index.long()
        if edge_index.numel() and torch.any(
                graph_index[edge_index[0]] != graph_index[edge_index[1]]):
            raise ValueError("edges may not connect different batch graphs")
        return species, edge_index, graph_index

    def _energy(self, graph: Union[GraphData, BatchGraphData],
                positions: torch.Tensor) -> torch.Tensor:
        """Run embedding, edge encoding, interaction blocks, and readout."""
        species, edge_index, graph_index = self._validate_graph(graph)
        center, neighbor = edge_index
        # Reuse topology but rebuild all geometry from differentiable positions.
        edge_vectors = positions[neighbor] - positions[center]
        distances = torch.linalg.vector_norm(edge_vectors, dim=-1)
        if distances.numel() and torch.any(distances <= 0):
            raise ValueError("NequIP does not support zero-length edges")
        directions = edge_vectors / distances.clamp_min(
            torch.finfo(positions.dtype).eps).unsqueeze(-1)

        radial_basis = self.bessel_basis(distances)
        radial_basis = radial_basis * self.cutoff_envelope(distances).unsqueeze(
            -1)
        spherical = get_spherical_from_cartesian(directions)
        spherical_harmonics = precompute_sh(spherical, self.l_max)

        # Atomic numbers enter the network as even scalar (0e) features.
        features: _Features = {
            (0, 1): (self.embedding(species) /
                     math.sqrt(self.num_species)).unsqueeze(-1)
        }
        for block in self.interaction_blocks:
            features = block(features, radial_basis, spherical_harmonics,
                             edge_index, species)

        # Equivariant projection selects 0e features for atomic energies.
        scalar_features = {(0, 1): features[(0, 1)]}
        atomic_energies = self.output_energy(
            self.output_hidden(scalar_features))[(0, 1)][:, 0, 0]
        batch_size = int(graph_index.max().item()) + 1
        total_energy = atomic_energies.new_zeros(batch_size)
        # graph_index assigns each atomic contribution to its structure.
        total_energy.index_add_(0, graph_index, atomic_energies)
        return total_energy.unsqueeze(-1)

    def forward(
        self,
        graph: Union[GraphData, BatchGraphData],
        compute_forces: Optional[bool] = None,
        create_graph: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict total energies and optionally conservative forces.

        Parameters
        ----------
        graph : GraphData or BatchGraphData
            Torch-converted graph containing atomic numbers, positions,
            directed edges, and optional graph membership.
        compute_forces : bool, optional
            Override the constructor setting for this call.
        create_graph : bool, default False
            Preserve the force derivative graph for force-loss backpropagation.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Graph energies with shape ``(batch_size, 1)``.  When forces are
            requested, also returns ``-dE/dR`` with shape ``(num_atoms, 3)``.

        Notes
        -----
        Force calls create a differentiable position tensor when needed;
        caller-owned graph tensors are not modified.
        """
        do_forces = self.compute_forces if compute_forces is None else compute_forces
        positions = graph.node_pos_features
        if not isinstance(positions, torch.Tensor):
            raise TypeError("NequIP requires a torch-converted GraphData")
        if not do_forces:
            return self._energy(graph, positions)

        with torch.enable_grad():
            if not positions.requires_grad:
                # Avoid setting requires_grad on the caller-owned graph tensor.
                positions = positions.detach().requires_grad_(True)
            total_energy = self._energy(graph, positions)
            forces = -torch.autograd.grad(total_energy.sum(),
                                          positions,
                                          create_graph=create_graph,
                                          retain_graph=create_graph)[0]
        return total_energy, forces
