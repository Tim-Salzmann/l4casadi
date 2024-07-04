# Real-time L4CasADi
This is the underlying framework enabling Real-time Neural-MPC in our paper
```
Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms
```
[Arxiv Link](https://arxiv.org/pdf/2203.07747)

Real-time L4CasADi provides an easy template to approximate a PyTorch Model in CasADi as first- or second-order
Taylor-Approximation. The approximation has to updated by the user from a Python interface depending on the use-case.

The necessary parameters to update the approximation can be inferenced in parallel in PyTorch potentially on a GPU.
This makes this approach efficient for large models where multiple approximations are used in parallel in an optimization (MPC with multiple shooting nodes).

## Real-time L4CasADi in Acados MPC
We provide a example of how to use Real-time L4CasADi in Acados MPC in this [example](../../examples/realtime/mpc_mlp_example.py).
Note that this is a dummy example of a single integrator with a learned residual dynamic set to zero. However, it
demonstrates the integration of RealTimeL4CasADi in Acados (RTI-)MPC.

## Examples
https://github.com/Tim-Salzmann/l4casadi/blob/dc15956b03a91549fc8e97eb88a89408f5c1d5a6/examples/realtime/readme.py#L32-L43

## Citing
If you use our work please cite our paper
```
@article{salzmann2023neural,
  title={Real-time Neural-MPC: Deep Learning Model Predictive Control for Quadrotors and Agile Robotic Platforms},
  author={Salzmann, Tim and Kaufmann, Elia and Arrizabalaga, Jon and Pavone, Marco and Scaramuzza, Davide and Ryll, Markus},
  journal={IEEE Robotics and Automation Letters},
  doi={10.1109/LRA.2023.3246839},
  year={2023}
}
```