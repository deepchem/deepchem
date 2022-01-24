#include <aev.h>
#include <torch/extension.h>
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

AEVScalarParams::AEVScalarParams(
    float Rcr_,
    float Rca_,
    Tensor EtaR_t_,
    Tensor ShfR_t_,
    Tensor EtaA_t_,
    Tensor Zeta_t_,
    Tensor ShfA_t_,
    Tensor ShfZ_t_,
    int num_species_)
    : Rcr(Rcr_),
      Rca(Rca_),
      radial_sublength(EtaR_t_.size(0) * ShfR_t_.size(0)),
      angular_sublength(EtaA_t_.size(0) * Zeta_t_.size(0) * ShfA_t_.size(0) * ShfZ_t_.size(0)),
      num_species(num_species_),
      EtaR_t(EtaR_t_),
      ShfR_t(ShfR_t_),
      EtaA_t(EtaA_t_),
      Zeta_t(Zeta_t_),
      ShfA_t(ShfA_t_),
      ShfZ_t(ShfZ_t_) {
  radial_length = radial_sublength * num_species;
  angular_length = angular_sublength * (num_species * (num_species + 1) / 2);
}

Result::Result(
    Tensor aev_t_,
    Tensor tensor_Rij_,
    Tensor tensor_radialRij_,
    Tensor tensor_angularRij_,
    int64_t total_natom_pairs_,
    int64_t nRadialRij_,
    int64_t nAngularRij_,
    Tensor tensor_centralAtom_,
    Tensor tensor_numPairsPerCenterAtom_,
    Tensor tensor_centerAtomStartIdx_,
    int64_t maxnbrs_per_atom_aligned_,
    int64_t angular_length_aligned_,
    int64_t ncenter_atoms_,
    Tensor coordinates_t_,
    Tensor species_t_)
    : aev_t(aev_t_),
      tensor_Rij(tensor_Rij_),
      tensor_radialRij(tensor_radialRij_),
      tensor_angularRij(tensor_angularRij_),
      total_natom_pairs(total_natom_pairs_),
      nRadialRij(nRadialRij_),
      nAngularRij(nAngularRij_),
      tensor_centralAtom(tensor_centralAtom_),
      tensor_numPairsPerCenterAtom(tensor_numPairsPerCenterAtom_),
      tensor_centerAtomStartIdx(tensor_centerAtomStartIdx_),
      maxnbrs_per_atom_aligned(maxnbrs_per_atom_aligned_),
      angular_length_aligned(angular_length_aligned_),
      ncenter_atoms(ncenter_atoms_),
      coordinates_t(coordinates_t_),
      species_t(species_t_) {}

Result::Result(tensor_list tensors)
    : aev_t(tensors[0]), // aev_t will be a undefined tensor
      tensor_Rij(tensors[1]),
      tensor_radialRij(tensors[2]),
      tensor_angularRij(tensors[3]),
      total_natom_pairs(tensors[4].item<int>()),
      nRadialRij(tensors[5].item<int>()),
      nAngularRij(tensors[6].item<int>()),
      tensor_centralAtom(tensors[7]),
      tensor_numPairsPerCenterAtom(tensors[8]),
      tensor_centerAtomStartIdx(tensors[9]),
      maxnbrs_per_atom_aligned(tensors[10].item<int>()),
      angular_length_aligned(tensors[11].item<int>()),
      ncenter_atoms(tensors[12].item<int>()),
      coordinates_t(tensors[13]),
      species_t(tensors[14]) {}

CuaevComputer::CuaevComputer(
    double Rcr,
    double Rca,
    const Tensor& EtaR_t,
    const Tensor& ShfR_t,
    const Tensor& EtaA_t,
    const Tensor& Zeta_t,
    const Tensor& ShfA_t,
    const Tensor& ShfZ_t,
    int64_t num_species)
    : aev_params(Rcr, Rca, EtaR_t, ShfR_t, EtaA_t, Zeta_t, ShfA_t, ShfZ_t, num_species) {}

Tensor CuaevDoubleAutograd::forward(
    AutogradContext* ctx,
    Tensor grad_e_aev,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer,
    tensor_list result_tensors) {
  Tensor grad_coord = cuaev_computer->backward(grad_e_aev, result_tensors);

  if (grad_e_aev.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->save_for_backward(result_tensors);
  }

  return grad_coord;
}

tensor_list CuaevDoubleAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  Tensor grad_force = grad_outputs[0];
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  Tensor grad_grad_aev = cuaev_computer->double_backward(grad_force, ctx->get_saved_variables());
  return {grad_grad_aev, torch::Tensor(), torch::Tensor()};
}

Tensor CuaevAutograd::forward(
    AutogradContext* ctx,
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  at::AutoNonVariableTypeMode g;
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  if (coordinates_t.requires_grad()) {
    ctx->saved_data["cuaev_computer"] = cuaev_computer;
    ctx->save_for_backward(result);
  }
  return result.aev_t;
}

tensor_list CuaevAutograd::backward(AutogradContext* ctx, tensor_list grad_outputs) {
  torch::intrusive_ptr<CuaevComputer> cuaev_computer = ctx->saved_data["cuaev_computer"].toCustomClass<CuaevComputer>();
  tensor_list result_tensors = ctx->get_saved_variables();
  Tensor grad_coord = CuaevDoubleAutograd::apply(grad_outputs[0], cuaev_computer, result_tensors);
  return {grad_coord, Tensor(), Tensor()};
}

Tensor run_only_forward(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  Result result = cuaev_computer->forward(coordinates_t, species_t);
  return result.aev_t;
}

Tensor run_autograd(
    const Tensor& coordinates_t,
    const Tensor& species_t,
    const torch::intrusive_ptr<CuaevComputer>& cuaev_computer) {
  return CuaevAutograd::apply(coordinates_t, species_t, cuaev_computer);
}

TORCH_LIBRARY(cuaev, m) {
  m.class_<CuaevComputer>("CuaevComputer")
      .def(torch::init<double, double, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int64_t>())
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<CuaevComputer>& self) -> std::vector<Tensor> {
            std::vector<Tensor> state;
            state.push_back(torch::tensor(self->aev_params.Rcr));
            state.push_back(torch::tensor(self->aev_params.Rca));
            state.push_back(self->aev_params.EtaR_t);
            state.push_back(self->aev_params.ShfR_t);
            state.push_back(self->aev_params.EtaA_t);
            state.push_back(self->aev_params.Zeta_t);
            state.push_back(self->aev_params.ShfA_t);
            state.push_back(self->aev_params.ShfZ_t);
            state.push_back(torch::tensor(self->aev_params.num_species));
            return state;
          },
          // __setstate__
          [](std::vector<Tensor> state) -> c10::intrusive_ptr<CuaevComputer> {
            return c10::make_intrusive<CuaevComputer>(
                state[0].item<double>(),
                state[1].item<double>(),
                state[2],
                state[3],
                state[4],
                state[5],
                state[6],
                state[7],
                state[8].item<int64_t>());
          });
  m.def("run", run_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, CUDA, m) {
  m.impl("run", run_only_forward);
}

TORCH_LIBRARY_IMPL(cuaev, Autograd, m) {
  m.impl("run", run_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
