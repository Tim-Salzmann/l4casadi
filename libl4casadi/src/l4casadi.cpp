#include <iostream>
#include <filesystem>

#include <torch/torch.h>
#include <torch/script.h>
//#include <torch/mps.h>

#include "l4casadi.hpp"

torch::Device cpu(torch::kCPU);

class L4CasADi::L4CasADiImpl
{
    torch::jit::script::Module forward_model;
    torch::jit::script::Module jac_model;
    torch::jit::script::Module hess_model;

    torch::Device device;

public:
    L4CasADiImpl(std::string model_path, std::string model_prefix, std::string device, bool has_jac, bool has_hess)
        : device{torch::kCPU} {
        if (torch::cuda::is_available() && device.compare("cpu")) {
            std::cout << "CUDA is available! Using GPU " << device << "." << std::endl;
            this->device = torch::Device(device);
        // } else if (torch::mps::is_available() && device.compare("cpu")) {
        //     std::cout << "MPS is available! Training on MPS " << device << "." << std::endl;
        //     this->device = torch::Device(device);
        } else if (!torch::cuda::is_available() && device.compare("cpu")) {
            std::wcout << "CUDA is not available! Using CPU." << std::endl;
            this->device = torch::Device(torch::kCPU);
        } else {
            this->device = torch::Device(device);
        }

        std::filesystem::path dir (model_path);
        std::filesystem::path forward_model_file (model_prefix + "_forward.pt");
        this->forward_model = torch::jit::load(dir / forward_model_file);
        this->forward_model.to(this->device);
        this->forward_model.eval();
        this->forward_model = torch::jit::optimize_for_inference(this->forward_model);

        if (has_jac) {
            std::filesystem::path jac_model_file (model_prefix + "_jacrev.pt");
            this->jac_model = torch::jit::load(dir / jac_model_file);
            this->jac_model.to(this->device);
            this->jac_model.eval();
            this->jac_model = torch::jit::optimize_for_inference(this->jac_model);
        }

        if (has_hess) {
            std::filesystem::path hess_model_file (model_prefix + "_hess.pt");
            this->hess_model = torch::jit::load(dir / hess_model_file);
            this->hess_model.to(this->device);
            this->hess_model.eval();
            this->hess_model = torch::jit::optimize_for_inference(this->hess_model);
        }
    }

    torch::Tensor forward(torch::Tensor input) {
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.to(this->device));
        return this->forward_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor jac(torch::Tensor input) {
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.to(this->device));
        return this->jac_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor hess(torch::Tensor input) {
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.to(this->device));
        return this->hess_model.forward(inputs).toTensor().to(cpu);
    }
};

L4CasADi::L4CasADi(std::string model_path, std::string model_prefix, bool model_expects_batch_dim, std::string device,
        bool has_jac, bool has_hess):
    pImpl{std::make_unique<L4CasADiImpl>(model_path, model_prefix, device, has_jac, has_hess)},
    model_expects_batch_dim{model_expects_batch_dim} {}

void L4CasADi::forward(const double* in, int rows, int cols, double* out) {
    torch::Tensor in_tensor;
    if (this->model_expects_batch_dim) {
        in_tensor = torch::from_blob(( void * )in, {1, rows}, at::kDouble).to(torch::kFloat);
    } else {
        in_tensor = torch::from_blob(( void * )in, {rows, cols}, at::kDouble).to(torch::kFloat);
    }

    torch::Tensor out_tensor = this->pImpl->forward(in_tensor).to(torch::kDouble).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::jac(const double* in, int rows, int cols, double* out) {
    torch::Tensor in_tensor;
    if (this->model_expects_batch_dim) {
        in_tensor = torch::from_blob(( void * )in, {1, rows}, at::kDouble).to(torch::kFloat);
    } else {
        in_tensor = torch::from_blob(( void * )in, {rows, cols}, at::kDouble).to(torch::kFloat);
    }
    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->jac(in_tensor).to(torch::kDouble).transpose(-2, -1).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::hess(const double* in, int rows, int cols, double* out) {
    torch::Tensor in_tensor;
    if (this->model_expects_batch_dim) {
        in_tensor = torch::from_blob(( void * )in, {1, rows}, at::kDouble).to(torch::kFloat);
    } else {
        in_tensor = torch::from_blob(( void * )in, {rows, cols}, at::kDouble).to(torch::kFloat);
    }

    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->hess(in_tensor).to(torch::kDouble).transpose(-2, -1).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

L4CasADi::~L4CasADi() = default;
