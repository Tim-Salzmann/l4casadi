import l4casadi as l4c
import casadi as cs
import numpy as np
import torch


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(2, 512)

        hidden_layers = []
        for i in range(20):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


pyTorch_model = MultiLayerPerceptron()

size_in = 2
size_out = 1
l4c_model = l4c.realtime.RealTimeL4CasADi(pyTorch_model, approximation_order=1)  # approximation_order=2

x_sym = cs.MX.sym('x', 2, 1)
y_sym = l4c_model(x_sym)

casadi_func = cs.Function('model_rt_approx',
                          [x_sym, l4c_model.get_sym_params()],
                          [y_sym])

x = np.ones([1, size_in])  # torch needs batch dimension
casadi_param = l4c_model.get_params(x)
casadi_out = casadi_func(x.transpose((-2, -1)), casadi_param)  # transpose for vector rep. expected by casadi

t_out = pyTorch_model(torch.tensor(x, dtype=torch.float32))

print(casadi_out)
print(t_out)