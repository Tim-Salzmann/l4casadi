[![PyPI version](https://badge.fury.io/py/l4casadi.svg)](https://badge.fury.io/py/l4casadi)
![L4CasADi CI](https://github.com/Tim-Salzmann/l4casadi/actions/workflows/ci.yaml/badge.svg)
![Downloads](https://img.shields.io/pypi/dm/l4casadi.svg)

---
# Learning 4 CasADi Framework

L4CasADi enables the seamless integration of PyTorch-learned models with CasADi for efficient and potentially
hardware-accelerated numerical optimization.
The only requirement on the PyTorch model is to be traceable and differentiable.

<div align="center">
  <img src="./examples/nerf_trajectory_optimization/media/animation.gif" alt="Collision-free minimum snap optimized trajectory through a NeRF" width="200" height="200">
  <img src="./examples/fish_turbulent_flow/media/trajectory_generation_vorticity.gif" alt="Energy Efficient Fish Navigation in Turbulent Flow" width="400" height="200">
  <br><a target="_blank" href="https://colab.research.google.com/github/Tim-Salzmann/l4casadi/blob/main/examples/nerf_trajectory_optimization/NeRF_Trajectory_Optimization.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a target="_blank" href="https://colab.research.google.com/github/Tim-Salzmann/l4casadi/blob/main/examples/fish_turbulent_flow/Fish_Turbulent_Flow.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <p><i>Two L4CasADi examples: Collision-free trajectory through a NeRF and Navigation in Turbulent Flow</i></p>
</div>

arXiv: [Learning for CasADi: Data-driven Models in Numerical Optimization](https://arxiv.org/pdf/2312.05873.pdf)

Talk: [Youtube](https://youtu.be/UYdkRnGr8eM?si=KEPcFEL9b7Vk2juI&t=3348)

## Table of Content
- [Projects using L4CasADi](#projects-using-l4casadi)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Online Learning](#online-learning-and-updating)
- [Naive L4CasADi](#naive-l4casadi) - Use this for small Multi Layer Perceptron Models.
- [Real-time L4CasADi](#real-time-l4casadi) - Use this for fast MPC with Acados.
- [Examples](#examples)

If you use this framework please cite the following two paper
```
@article{salzmann2023neural,
  title={Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms},
  author={Salzmann, Tim and Kaufmann, Elia and Arrizabalaga, Jon and Pavone, Marco and Scaramuzza, Davide and Ryll, Markus},
  journal={IEEE Robotics and Automation Letters},
  doi={10.1109/LRA.2023.3246839},
  year={2023}
}
```

```
@inproceedings{{salzmann2024l4casadi,
  title={Learning for CasADi: Data-driven Models in Numerical Optimization},
  author={Salzmann, Tim and Arrizabalaga, Jon and Andersson, Joel and Pavone, Marco and Ryll, Markus},
  booktitle={Learning for Dynamics and Control Conference (L4DC)},
  year={2024}
}
```

## Projects using L4CasADi
- Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms <br/> [Paper](https://arxiv.org/pdf/2203.07747.pdf) | [Code](https://github.com/TUM-AAS/neural-mpc)
- Neural Potential Field for Obstacle-Aware Local Motion Planning <br/> [Paper](https://arxiv.org/pdf/2310.16362.pdf) | [Video](https://www.youtube.com/watch?v=KL3bfvUwGqs) | [Code](https://github.com/cog-isa/NPField)
- N-MPC for Deep Neural Network-Based Collision Avoidance exploiting Depth Images <br/> [Paper](https://arxiv.org/pdf/2402.13038.pdf) | [Code](https://github.com/ntnu-arl/colpred_nmpc)
- Reinforcement Learning based MPC with Neural Dynamical Models <br/> [Paper](https://folk.ntnu.no/skoge/publications/2024/adhau-ecc24/ECC24_0903_FI.pdf)

If your project is using L4CasADi and you would like to be featured here, please reach out.

---
## Installation
### Prerequisites
Independently if you install from source or via pip you will need to meet the following requirements:

- Working build system: CMake compatible C++ compiler (GCC version 10 or higher).
- PyTorch (`>=2.0`) installation in your python environment.\
`python -c "import torch; print(torch.__version__)"`

### Pip Install (CPU Only)
- Ensure Torch CPU-version is installed\
`pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cpu`
- Ensure all build dependencies are installed
```
setuptools>=68.1
scikit-build>=0.17
cmake>=3.27
ninja>=1.11
```

- Run\
`pip install l4casadi --no-build-isolation`

### From Source (CPU Only)
- Clone the repository\
`git clone https://github.com/Tim-Salzmann/l4casadi.git`

- All build dependencies installed via\
`pip install -r requirements_build.txt`

- Build from source\
`pip install . --no-build-isolation`

The `--no-build-isolation` flag is required for L4CasADi to find and link against the installed PyTorch.

### GPU (CUDA)
CUDA installation requires nvcc to be installed which is part of the CUDA toolkit and can be installed on Linux via
`sudo apt-get -y install cuda-toolkit-XX-X` (where `XX-X` is your installed Cuda version - e.g. `12-3`).
Once the CUDA toolkit is installed nvcc is commonly found at `/usr/local/cuda/bin/nvcc`.

Make sure `nvcc -V` can be executed and run `pip install l4casadi --no-build-isolation` or `CUDACXX=<PATH_TO_NVCC> pip install . --no-build-isolation` to build from source.

If `nvcc` is not automatically part of your path you can specify the `nvcc` path for L4CasADi.
E.g. `CUDACXX=<PATH_TO_NVCC> pip install l4casadi --no-build-isolation`.

---

## Quick Start

Defining an L4CasADi model in Python given a pre-defined PyTorch model is as easy as
```python
import l4casadi as l4c

l4c_model = l4c.L4CasADi(pyTorch_model, device='cpu')
```

where the architecture of the PyTorch model is unrestricted and large models can be accelerated with dedicated hardware.

---

## Online Learning and Updating
L4CasADi supports updating the PyTorch model online in the CasADi graph. To use this feature, pass `mutable=True` when
initializing a L4CasADi. To update the model, call the `update` function on the `L4CasADi` object.
You can optionally pass an updated model as parameter. If no model is passed, the reference passed at
initialization is assumed to be updated and will be used for the update.

---

## Naive L4CasADi

While L4CasADi was designed with efficiency in mind by internally leveraging torch's C++ interface, this can still
result in overhead, which can be disproportionate for small, simple models. Thus, L4CasADi additionally provides a
`NaiveL4CasADiModule` which directly recreates the PyTorch computational graph using CasADi operations and copies the
weights --- leading to a pure C computational graph without context switches to torch. However, this approach is
limited to a small predefined subset of PyTorch operations --- only `MultiLayerPerceptron`
models and CPU inference are supported.

The torch framework overhead dominates for networks smaller than three hidden layers, each with 64
neurons (or equivalent). For models smaller than this size we recommend using the NaiveL4CasADiModule.
For larger models, the overhead becomes negligible and L4CasADi should be used.

https://github.com/Tim-Salzmann/l4casadi/blob/59876b190941c8d43286b6eb3d710add6f14a1d2/examples/naive/readme.py#L5-L9

---
## Real-time L4CasADi
Real-time L4Casadi (former `Approximated` approach in [ML-CasADi](https://github.com/TUM-AAS/ml-casadi)) is the underlying framework powering
[Real-time Neural-MPC](https://arxiv.org/pdf/2203.07747). It replaces complex models with local Taylor approximations.
For certain optimization procedures (such as MPC with multiple shooting nodes) this can lead to improved optimization times.
However, `Real-time L4Casadi`, comes with many restrictions (only Python, no C(++) code generation, ...) and is therefore not
a one-to-one replacement for `L4Casadi`. Rather it is a complementary framework for certain special use cases.

More information [here](l4casadi/realtime).

https://github.com/Tim-Salzmann/l4casadi/blob/62219f61375af4c4133167f1d8b138d90f678c32/l4casadi/realtime/examples/readme.py#L32-L43

---

## Examples
https://github.com/Tim-Salzmann/l4casadi/blob/23e07380e214f70b8932578317aa373d2216b57e/examples/readme.py#L28-L40

Please note that only `casadi.MX` symbolic variables are supported as input.

Multi-input multi-output functions can be realized by concatenating the symbolic inputs when passing to the model and
splitting them inside the PyTorch function.

To use GPU (CUDA) simply pass `device="cuda"` to the `L4CasADi` constructor.

Further examples:
- Collision-free minimum snap optimized trajectory through a NeRF: [examples/nerf_trajectory_optimization](/examples/nerf_trajectory_optimization)
- Energy Efficient Fish Navigation in Turbulent Flow: [examples/fish_turbulent_flow](/examples/fish_turbulent_flow)
- Simple nonlinear programming with L4CasADi model as objective and constraints: [examples/simple_nlp.py](/examples/simple_nlp.py)
- L4CasADi in pure C(++) projects: [examples/cpp_executable](/examples/cpp_usage)
- Use PyTorch L4CasADi Model in Matlab: [examples/matlab](/examples/matlab)

---

## Acados Integration
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

---

## FYIs

### Batch Dimension

If your PyTorch model expects a batch dimension as first dimension (which most models do) you should pass
`model_expects_batch_dim=True` to the `L4CasADi` constructor. The `MX` input to the L4CasADi component is then expected
to be a vector of shape `[X, 1]`. L4CasADi will add a batch dimension of `1` automatically such that the input to the
underlying PyTorch model is of shape `[1, X]`.

---

### Warm Up

Note that PyTorch builds the graph on first execution. Thus, the first call(s) to the CasADi function will be slow.
You can warm up to the execution graph by calling the generated CasADi function one or multiple times before using it.
