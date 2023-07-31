![L4CasADi CI](https://github.com/Tim-Salzmann/l4casadi/actions/workflows/ci.yaml/badge.svg)

# Learning 4 CasADi Framework

L4CasADi enables using PyTorch models and functions in a CasADi graph while supporting CasADis code generation 
capabilities. The only requirement on the PyTorch model is to be traceable and differentiable.

If you use this framework please cite our paper
```
@article{salzmann2023neural,
  title={Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms},
  author={Salzmann, Tim and Kaufmann, Elia and Arrizabalaga, Jon and Pavone, Marco and Scaramuzza, Davide and Ryll, Markus},
  journal={IEEE Robotics and Automation Letters},
  doi={10.1109/LRA.2023.3246839},
  year={2023}
}
```

## Installation
Tested on Ubuntu 20.04 and MacOS.

### Prerequisites
Python 3.9 or higher.
gcc, make and cmake.

### CPU Installation
CPU only installation of `PyTorch==2.0.0`. CPU versions `>2.0.0` might work too.

Install L4CasADi via `pip install .`

### GPU (CUDA) Installation
CUDA installation of `PyTorch==2.0.0`.

Install L4CasADi via `USE_CUDA=TRUE CUDACXX=<PATH_TO_nvcc> pip install .`

### Mac M1 - ARM
On MacOS with M1 chip you will have to compile [tera_renderer](https://github.com/acados/tera_renderer) from source
and place the binary in `l4casadi/template_generation/bin`. For other platforms it will be downloaded automatically.

## Example
https://github.com/Tim-Salzmann/l4casadi/blob/23e07380e214f70b8932578317aa373d2216b57e/examples/readme.py#L28-L40

Please note that only `casadi.MX` symbolic variables are supported as input.

Multi-input multi-output functions can be realized by concatenating the symbolic inputs when passing to the model and
splitting them inside the PyTorch function.

To use GPU (CUDA) simply pass `device="cuda"` to the `L4CasADi` constructor.

An example of solving a simple NLP with torch system model can be found in
[examples/simple_nlp.py](/examples/simple_nlp.py).

## Integration with Acados
To use this framework with Acados:
- Follow the [installation instructions](https://docs.acados.org/installation/index.html).
- Install the [Python Interface](https://docs.acados.org/python_interface/index.html).
- Ensure that `LD_LIBRARY_PATH` is set correctly (`DYLD_LIBRARY_PATH`on MacOS).
- Ensure that `ACADOS_SOURCE_DIR` is set correctly.

An example of how a PyTorch model can be used as dynamics model in the Acados framework for Model Predictive Control 
can be found in [examples/acados.py](/examples/acados.py)

To use L4CasADi with Acados you will have to set `model_external_shared_lib_dir` and `model_external_shared_lib_name`
in the `AcadosOcp.solver_options` accordingly.

```
ocp.solver_options.model_external_shared_lib_dir = l4c_model.shared_lib_dir
ocp.solver_options.model_external_shared_lib_name = l4c_model.name
```

https://github.com/Tim-Salzmann/l4casadi/blob/421de6ef408267eed0fd2519248b2152b610d2cc/examples/acados.py#L156-L160

## Warm Up

Note that PyTorch builds the graph on first execution. Thus, the first call(s) to the CasADi function will be slow.
You can warm up to the execution graph by calling the generated CasADi function one or multiple times before using it.

## Roadmap
Further development of this framework will be prioritized by popular demand. If a feature is important to your work
please get in contact or create a pull request.

Possible upcoming features include:
```
- Explicit multi input, multi output functions.
```