import numpy as np
import pytest
import torch
import casadi as cs
import l4casadi as l4c


class DeepModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.input_layer = torch.nn.Linear(dim_in, 512)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.ln = torch.nn.LayerNorm(512)

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, dim_out)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.ln(x)
        x = self.out_layer(x)
        return x


class TestRealTimeL4CasADi:
    @pytest.fixture(
        params=[(1, 3), (2, 3), (3, 1)]
    )
    def deep_model(self, request):
        in_dim, out_dim = request.param
        return DeepModel(in_dim, out_dim)

    def test_realtime_l4casadi_deep_model(self, deep_model):
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = deep_model(rand_inp)

        mx_inp = cs.MX.sym('x', deep_model.input_layer.in_features, 1)

        l4c_model = l4c.realtime.RealTimeL4CasADi(deep_model)

        mx_out = l4c_model(mx_inp)

        params = l4c_model.get_params(rand_inp)

        cs_fun = cs.Function('model', [mx_inp, l4c_model.get_sym_params()], [mx_out])

        l4c_out = cs_fun(rand_inp.transpose(-2, -1).detach().numpy(), params)

        assert np.allclose(l4c_out, torch_out.transpose(-2, -1).detach().numpy(), atol=1e-6)

    def test_realtime_l4casadi_deep_model_second_order(self, deep_model):
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = deep_model(rand_inp)

        mx_inp = cs.MX.sym('x', deep_model.input_layer.in_features, 1)

        l4c_model = l4c.realtime.RealTimeL4CasADi(deep_model, approximation_order=2)

        mx_out = l4c_model(mx_inp)

        params = l4c_model.get_params(rand_inp)

        cs_fun = cs.Function('model', [mx_inp, l4c_model.get_sym_params()], [mx_out])

        l4c_out = cs_fun(rand_inp.transpose(-2, -1).detach().numpy(), params)

        assert np.allclose(l4c_out, torch_out.transpose(-2, -1).detach().numpy(), atol=1e-6)

