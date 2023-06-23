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

Prerequisites: make and cmake. CPU only installation of `PyTorch==2.0.0`. CPU versions `>2.0.0` might work too.

Install L4CasADi via `pip install .`

On MacOS with M1 chip you will have to compile [tera_renderer](https://github.com/acados/tera_renderer) from source
and place the binary in `l4casadi/template_generation/bin`. For other platforms it will be downloaded automatically.

## Example
https://github.com/Tim-Salzmann/l4casadi/blob/5edbe4b31d915c6d897608f183d06c53eaf14f63/examples/readme.py#L28-L40

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

## Warm Up

Note that PyTorch builds the graph on first execution. Thus, the first call(s) to the CasADi function will be slow.
You can warm up to the execution graph by calling the generated CasADi function one or multiple times before using it.

## Roadmap
Further development of this framework will be prioritized by popular demand. If a feature is important to your work
please get in contact or create a pull request.

Possible upcoming features include:
```
- GPU support.
- Multi input, multi output functions.
```