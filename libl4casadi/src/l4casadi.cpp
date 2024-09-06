#include <iostream>
#include <stdexcept>
#include <filesystem>

#include <torch/torch.h>
#include <torch/script.h>
//#include <torch/mps.h>

#if ENABLE_TORCH_COMPILE
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#if USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#endif

#include "l4casadi.hpp"

torch::Device cpu(torch::kCPU);

class L4CasADi::L4CasADiImpl
{
protected:
    std::string path;
    std::string function_name;

    bool has_jac;
    bool has_adj1;
    bool has_jac_adj1;
    bool has_hess;

    bool is_mutable;

    torch::Device device;

public:
    L4CasADiImpl(std::string path, std::string function_name, std::string device, bool has_jac, bool has_adj1,
            bool has_jac_adj1, bool has_hess, bool is_mutable): device{torch::kCPU}, path{path},
            function_name{function_name}, has_jac{has_jac}, has_adj1{has_adj1}, has_jac_adj1{has_jac_adj1},
            has_hess{has_hess}, is_mutable(is_mutable) {
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
    }
    virtual torch::Tensor forward(torch::Tensor) = 0;
    virtual torch::Tensor jac(torch::Tensor) = 0;
    virtual torch::Tensor adj1(torch::Tensor, torch::Tensor) = 0;
    virtual torch::Tensor jac_adj1(torch::Tensor, torch::Tensor) = 0;
    virtual torch::Tensor hess(torch::Tensor) = 0;

    virtual ~L4CasADiImpl() = default;
};

#if ENABLE_TORCH_COMPILE
class L4CasADi::L4CasADiCompiledImpl : public L4CasADi::L4CasADiImpl
{
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> forward_model;
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> jac_model;
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> adj1_model;
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> jac_adj1_model;
    std::unique_ptr<torch::inductor::AOTIModelContainerRunner> hess_model;

    std::mutex model_update_mutex;

public:
    L4CasADiCompiledImpl(std::string path, std::string function_name, std::string device, bool has_jac, bool has_adj1,
            bool has_jac_adj1, bool has_hess, bool is_mutable): L4CasADiImpl(path, function_name, device, has_jac,
            has_adj1, has_jac_adj1, has_hess, is_mutable) {

        this->load_model_from_disk();

        if (is_mutable) {
            throw std::invalid_argument("Mutable functions are not yet supported for compiled models.");
        }
    }

    ~L4CasADiCompiledImpl() = default;

    void load_model_from_disk() {
        std::filesystem::path dir (this->path);
        std::filesystem::path forward_model_file (this->function_name + ".so");
#if USE_CUDA
        if (this-> device == cpu) {
             this->forward_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / forward_model_file).generic_string());
        }
        else {
            this->forward_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>((dir / forward_model_file).generic_string());
        }
#else
        this->forward_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / forward_model_file).generic_string());
#endif
        if (this->has_adj1) {
            std::filesystem::path adj1_model_file ("adj1_" + this->function_name + ".so");
#if USE_CUDA
            if (this-> device == cpu) {
                 this->adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / adj1_model_file).generic_string());
            }
            else {
                this->adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>((dir / adj1_model_file).generic_string());
            }
#else
            this->adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / adj1_model_file).generic_string());
#endif
        }

        if (this->has_jac_adj1) {
            std::filesystem::path jac_adj1_model_file ("jac_adj1_" + this->function_name + ".so");
#if USE_CUDA
            if (this-> device == cpu) {
                 this->jac_adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / jac_adj1_model_file).generic_string());
            }
            else {
                this->jac_adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>((dir / jac_adj1_model_file).generic_string());
            }
#else
            this->jac_adj1_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / jac_adj1_model_file).generic_string());
#endif
        }

        if (this->has_jac) {
            std::filesystem::path jac_model_file ("jac_" + this->function_name + ".so");
#if USE_CUDA
            if (this-> device == cpu) {
                 this->jac_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / jac_model_file).generic_string());
            }
            else {
                this->jac_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>((dir / jac_model_file).generic_string());
            }
#else
            this->jac_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / jac_model_file).generic_string());
#endif
        }

        if (this->has_hess) {
            std::filesystem::path hess_model_file ("jac_jac_" + this->function_name + ".so");
#if USE_CUDA
            if (this-> device == cpu) {
                 this->hess_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / hess_model_file).generic_string());
            }
            else {
                this->hess_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>((dir / hess_model_file).generic_string());
            }
#else
            this->hess_model = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>((dir / hess_model_file).generic_string());
#endif
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::Tensor> inputs;
        inputs.push_back(x);
        auto out = this->forward_model->run(inputs)[0].to(cpu);
        return out;
    }

    torch::Tensor jac(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::Tensor> inputs;
        inputs.push_back(x.to(this->device));
        return this->jac_model->run(inputs)[0].to(cpu);
    }

    torch::Tensor adj1(torch::Tensor primal, torch::Tensor tangent) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::Tensor> inputs;
        inputs.push_back(primal.to(this->device));
        inputs.push_back(tangent.to(this->device));
        return this->adj1_model->run(inputs)[0].to(cpu);
    }

    torch::Tensor jac_adj1(torch::Tensor primal, torch::Tensor tangent){
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::Tensor> inputs;
        inputs.push_back(primal.to(this->device));
        inputs.push_back(tangent.to(this->device));
        return this->jac_adj1_model->run(inputs)[0].to(cpu);
    }

    torch::Tensor hess(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::Tensor> inputs;
        inputs.push_back(x.to(this->device));
        return this->hess_model->run(inputs)[0].to(cpu);
    }

};
#endif

class L4CasADi::L4CasADiScriptedImpl : public L4CasADi::L4CasADiImpl
{
    torch::jit::script::Module adj1_model;
    torch::jit::script::Module forward_model;
    torch::jit::script::Module jac_model;
    torch::jit::script::Module jac_adj1_model;
    torch::jit::script::Module hess_model;

    std::thread online_model_reloader_thread;
    std::mutex model_update_mutex;
    std::atomic<bool> reload_model_loop_running = false;

public:
    L4CasADiScriptedImpl(std::string path, std::string function_name, std::string device, bool has_jac, bool has_adj1,
            bool has_jac_adj1, bool has_hess, bool is_mutable): L4CasADiImpl(path, function_name, device, has_jac,
            has_adj1, has_jac_adj1, has_hess, is_mutable) {

        this->load_model_from_disk();

        if (is_mutable) {
            this->reload_model_loop_running = true;
            this->online_model_reloader_thread = std::thread(&L4CasADiScriptedImpl::reload_runner, this);
        }
    }

    ~ L4CasADiScriptedImpl() {
        if (this->reload_model_loop_running == true) {
            this->reload_model_loop_running = false;
            this->online_model_reloader_thread.join();
        }
    }

    void reload_runner() {
        std::filesystem::path dir (this->path);
        std::filesystem::path reload_file (this->function_name + ".reload");

        while(this->reload_model_loop_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (std::filesystem::exists(dir / reload_file)) {
                std::unique_lock<std::mutex> lock(this->model_update_mutex);
                this->load_model_from_disk();
                std::filesystem::remove(dir / reload_file);
            }
        }
    }

    void load_model_from_disk() {
        std::filesystem::path dir (this->path);
        std::filesystem::path forward_model_file (this->function_name + ".pt");
        this->forward_model = torch::jit::load((dir / forward_model_file).generic_string());
        this->forward_model.to(this->device);
        this->forward_model.eval();
        this->forward_model = torch::jit::optimize_for_inference(this->forward_model);

        if (this->has_adj1) {
            std::filesystem::path adj1_model_file ("adj1_" + this->function_name + ".pt");
            this->adj1_model = torch::jit::load((dir / adj1_model_file).generic_string());
            this->adj1_model.to(this->device);
            this->adj1_model.eval();
            this->adj1_model = torch::jit::optimize_for_inference(this->adj1_model);
        }

        if (this->has_jac_adj1) {
            std::filesystem::path jac_adj1_model_file ("jac_adj1_" + this->function_name + ".pt");
            this->jac_adj1_model = torch::jit::load((dir / jac_adj1_model_file).generic_string());
            this->jac_adj1_model.to(this->device);
            this->jac_adj1_model.eval();
            this->jac_adj1_model = torch::jit::optimize_for_inference(this->jac_adj1_model);
        }

        if (this->has_jac) {
            std::filesystem::path jac_model_file ("jac_" + this->function_name + ".pt");
            this->jac_model = torch::jit::load((dir / jac_model_file).generic_string());
            this->jac_model.to(this->device);
            this->jac_model.eval();
            this->jac_model = torch::jit::optimize_for_inference(this->jac_model);
        }

        if (this->has_hess) {
            std::filesystem::path hess_model_file ("jac_jac_" + this->function_name + ".pt");
            this->hess_model = torch::jit::load((dir / hess_model_file).generic_string());
            this->hess_model.to(this->device);
            this->hess_model.eval();
            this->hess_model = torch::jit::optimize_for_inference(this->hess_model);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x.to(this->device));
        return this->forward_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor jac(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x.to(this->device));
        return this->jac_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor adj1(torch::Tensor primal, torch::Tensor tangent) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(primal.to(this->device));
        inputs.push_back(tangent.to(this->device));

        return this->adj1_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor jac_adj1(torch::Tensor primal, torch::Tensor tangent){
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(primal.to(this->device));
        inputs.push_back(tangent.to(this->device));

        return this->jac_adj1_model.forward(inputs).toTensor().to(cpu);
    }

    torch::Tensor hess(torch::Tensor x) {
        std::unique_lock<std::mutex> lock(this->model_update_mutex);
        c10::InferenceMode guard;
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(x.to(this->device));
        return this->hess_model.forward(inputs).toTensor().to(cpu);
    }
};

L4CasADi::L4CasADi(std::string path, std::string function_name, int rows_in, int cols_in, int rows_out, int cols_out,
    std::string device, bool has_jac, bool has_adj1, bool has_jac_adj1, bool has_hess, bool scripting, bool is_mutable):
    rows_in{rows_in}, cols_in{cols_in}, rows_out{rows_out}, cols_out{cols_out} {
#if ENABLE_TORCH_COMPILE
    if (scripting == true) {
        this->pImpl = std::make_unique<L4CasADiScriptedImpl>(path, function_name, device, has_jac, has_adj1, has_jac_adj1, has_hess, is_mutable);
    } else {
        this->pImpl = std::make_unique<L4CasADiCompiledImpl>(path, function_name, device, has_jac, has_adj1, has_jac_adj1, has_hess, is_mutable);
   }
#else
     this->pImpl = std::make_unique<L4CasADiScriptedImpl>(path, function_name, device, has_jac, has_adj1, has_jac_adj1, has_hess, is_mutable);
#endif
}

void L4CasADi::forward(const double* x, double* out) {
    torch::Tensor x_tensor;
    x_tensor = torch::from_blob(( void * )x, {this->cols_in, this->rows_in}, at::kDouble).to(torch::kFloat).permute({1, 0});
    torch::Tensor out_tensor = this->pImpl->forward(x_tensor).to(torch::kDouble).permute({1, 0}).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::jac(const double* x, double* out) {
    torch::Tensor x_tensor;
    x_tensor = torch::from_blob(( void * )x, {this->cols_in, this->rows_in}, at::kDouble).to(torch::kFloat).permute({1, 0});
    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->jac(x_tensor).to(torch::kDouble).permute({3, 2, 1, 0}).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::adj1(const double* p, const double* t, double* out) {
    // adj1 [i0, out_o0, adj_o0] -> [out_adj_i0]
    torch::Tensor p_tensor, t_tensor;
    p_tensor = torch::from_blob(( void * )p, {this->cols_in, this->rows_in}, at::kDouble).to(torch::kFloat).permute({1, 0});
    t_tensor = torch::from_blob(( void * )t, {this->cols_out, this->rows_out}, at::kDouble).to(torch::kFloat).permute({1, 0});

    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->adj1(p_tensor, t_tensor).to(torch::kDouble).permute({1, 0}).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::jac_adj1(const double* p, const double* t, double* out) {
    // jac_adj1 [i0, out_o0, adj_o0, out_adj_i0] -> [jac_adj_i0_i0, jac_adj_i0_out_o0, jac_adj_i0_adj_o0]
    torch::Tensor p_tensor, t_tensor;
    p_tensor = torch::from_blob(( void * )p, {this->cols_in, this->rows_in}, at::kDouble).to(torch::kFloat).permute({1, 0});
    t_tensor = torch::from_blob(( void * )t, {this->cols_out, this->rows_out}, at::kDouble).to(torch::kFloat).permute({1, 0});

    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->jac_adj1(p_tensor, t_tensor).to(torch::kDouble).permute({3, 2, 1, 0}).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::jac_jac(const double* x, double* out) {
    torch::Tensor x_tensor;
    x_tensor = torch::from_blob(( void * )x, {this->cols_in, this->rows_in}, at::kDouble).to(torch::kFloat).permute({1, 0});

    // CasADi expects the return in Fortran order -> Transpose last two dimensions
    torch::Tensor out_tensor = this->pImpl->hess(x_tensor).to(torch::kDouble).permute({5, 4, 3, 2, 1, 0}).contiguous();
    std::memcpy(out, out_tensor.data_ptr<double>(), out_tensor.numel() * sizeof(double));
}

void L4CasADi::invalid_argument(std::string error_msg) {
    throw std::invalid_argument(error_msg);
}

L4CasADi::~L4CasADi() = default;
