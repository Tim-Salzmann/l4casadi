#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <torch/script.h>

#include "l4casadi.hpp"

class L4CasADi::L4CasADiImpl
{
    torch::jit::script::Module forward_model;
    torch::jit::script::Module jac_model;

public:
    L4CasADiImpl(std::string model_path, std::string model_prefix, std::string device) {
        std::filesystem::path dir (model_path);
        std::filesystem::path forward_model_file (model_prefix + "_forward.pt");
        this->forward_model = torch::jit::load(dir / forward_model_file);
        this->forward_model.eval();
        this->forward_model = torch::jit::optimize_for_inference(this->forward_model);

        std::filesystem::path jac_model_file (model_prefix + "_jacrev.pt");
        this->jac_model = torch::jit::load(dir / jac_model_file);
        this->jac_model.eval();
        this->jac_model = torch::jit::optimize_for_inference(this->jac_model);
    }

    torch::Tensor forward(torch::Tensor input) {
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        return this->forward_model.forward(inputs).toTensor();
    }

    torch::Tensor jac(torch::Tensor input) {
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        return this->jac_model.forward(inputs).toTensor();
    }
};

L4CasADi::L4CasADi(std::string model_path, std::string model_prefix, bool has_batch, std::string device):
    pImpl{std::make_unique<L4CasADiImpl>(model_path, model_prefix, device)},
    has_batch{has_batch} {}

void L4CasADi::forward(const double* in, int rows, int cols, double* out) {
    torch::Tensor in_tensor;
    if (this->has_batch) {
        in_tensor = torch::from_blob(( void * )in, {1, rows}, at::kDouble).to(torch::kFloat);
    } else {
        in_tensor = torch::from_blob(( void * )in, {rows, cols}, at::kDouble).to(torch::kFloat);
    }

    torch::Tensor out_tensor = this->pImpl->forward(in_tensor).to(torch::kDouble).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::jac(const double* in, int rows, int cols, double* out) {
    torch::Tensor in_tensor;
    if (this->has_batch) {
        in_tensor = torch::from_blob(( void * )in, {1, rows}, at::kDouble).to(torch::kFloat);
    } else {
        in_tensor = torch::from_blob(( void * )in, {rows, cols}, at::kDouble).to(torch::kFloat);
    }
    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->jac(in_tensor).to(torch::kDouble).transpose(-2, -1).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

L4CasADi::~L4CasADi() = default;
