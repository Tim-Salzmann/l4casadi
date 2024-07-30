import casadi as cs
import torch
import l4casadi as l4c


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
l4c_model = l4c.L4CasADi(pyTorch_model, device='cpu')  # device='cuda' for GPU

x_sym = cs.MX.sym('x', 1, 2)
y_sym = l4c_model(x_sym)
f = cs.Function('y', [x_sym], [y_sym])
df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
ddf = cs.Function('ddy', [x_sym], [cs.hessian(y_sym, x_sym)[0]])

x = cs.DM([[0.], [2.]])
print(l4c_model(x))
print(f(x))
print(df(x))
print(ddf(x))
