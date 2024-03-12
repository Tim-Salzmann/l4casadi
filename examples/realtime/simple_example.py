import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import torch

import l4casadi as l4c


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(2, 512)

        hidden_layers = []
        for i in range(1):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


def example():
    # Create a randomly initialized non linear Multi Layer Perceptron
    pyTorch_model = MultiLayerPerceptron()

    casadi_sym_inp = cs.MX.sym('inp', 2)

    # First-order Taylor (Linear) approximation of the model as Casadi Function
    l4c_model_order1 = l4c.realtime.RealTimeL4CasADi(pyTorch_model, approximation_order=1)
    casadi_lin_approx_sym_out = l4c_model_order1(casadi_sym_inp)
    casadi_lin_approx_func = cs.Function('model2_lin',
                                         [casadi_sym_inp,
                                          l4c_model_order1.get_sym_params()],
                                         [casadi_lin_approx_sym_out])
    casadi_lin_approx_param = l4c_model_order1.get_params(np.zeros((2,)))

    # Second-order Taylor (Quadratic) approximation of the model as Casadi Function
    l4c_model_order2 = l4c.realtime.RealTimeL4CasADi(pyTorch_model, approximation_order=2)
    casadi_quad_approx_sym_out = l4c_model_order2(casadi_sym_inp)
    casadi_quad_approx_func = cs.Function('model2_lin',
                                         [casadi_sym_inp,
                                          l4c_model_order2.get_sym_params()],
                                         [casadi_quad_approx_sym_out])
    casadi_quad_approx_param = l4c_model_order2.get_params(np.zeros((2,)))

    # Evaluate the functions and compare to the torch MLP
    inputs = np.stack([np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)], axis=1)
    torch_out = pyTorch_model(torch.tensor(inputs).float()).detach().numpy()
    casadi_lin_out = []
    casadi_quad_out = []

    # Casadi can not handle batches
    for i in range(100):
        casadi_lin_out.append(casadi_lin_approx_func(inputs[i], casadi_lin_approx_param))
        casadi_quad_out.append(casadi_quad_approx_func(inputs[i], casadi_quad_approx_param))

    casadi_lin_out = np.array(casadi_lin_out).squeeze(axis=-1)
    casadi_quad_out = np.array(casadi_quad_out).squeeze(axis=-1)

    plt.plot(torch_out, label='Torch', linewidth=4)
    plt.plot(casadi_lin_out, label='RealTimeL4CasADi Linear')
    plt.plot(casadi_quad_out, label='RealTimeL4CasADi Quadratic')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
