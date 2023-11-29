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
        for i in range(20):
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


class TrigModel(torch.nn.Module):
    def forward(self, x):
        return torch.concat([torch.sin(x[:1]), torch.cos(x[1:2])], dim=0)


class TestL4CasADi:
    @pytest.fixture(
        params=[(1, 3), (2, 3), (3, 1)]
    )
    def deep_model(self, request):
        in_dim, out_dim = request.param
        return DeepModel(in_dim, out_dim)

    @pytest.fixture
    def triag_model(self):
        return TrigModel()

    def test_l4casadi_deep_model(self, deep_model):
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = deep_model(rand_inp)

        l4c_out = l4c.L4CasADi(deep_model, model_expects_batch_dim=True)(rand_inp.transpose(-2, -1).detach().numpy())

        assert np.allclose(l4c_out, torch_out.transpose(-2, -1).detach().numpy())

    def test_l4casadi_triag_model(self, triag_model):
        rand_inp = torch.rand((12, 12))
        torch_out = triag_model(rand_inp)

        l4c_out = l4c.L4CasADi(triag_model, model_expects_batch_dim=False)(rand_inp.detach().numpy())

        assert np.allclose(l4c_out, torch_out.detach().numpy())

    def test_l4casadi_triag_model_jac(self, triag_model):
        rand_inp = torch.rand((12, 12))
        torch_out = torch.func.jacrev(triag_model)(rand_inp)

        mx_inp = cs.MX.sym('x', 12, 12)

        jac_fun = cs.Function('f_jac',
                              [mx_inp],
                              [cs.jacobian(l4c.L4CasADi(triag_model, model_expects_batch_dim=False)(mx_inp), mx_inp)])

        l4c_out = jac_fun(rand_inp.detach().numpy())

        assert np.allclose(
            np.moveaxis(np.array(l4c_out).reshape((12, 2, 12, 12)), (0, 1, 2, 3), (1, 0, 3, 2)),  # Reshape in Fortran
            torch_out.detach().numpy())

    def test_l4casadi_triag_model_hess_double_jac(self, triag_model):
        rand_inp = torch.rand((12, 12))
        torch_out = torch.func.hessian(triag_model)(rand_inp)

        mx_inp = cs.MX.sym('x', 12, 12)

        hess_fun = cs.Function('f_hess_double_jac',
                               [mx_inp],
                               [cs.jacobian(
                                   cs.jacobian(
                                       l4c.L4CasADi(triag_model, model_expects_batch_dim=False)(mx_inp), mx_inp
                                   )[0, 0], mx_inp)])

        l4c_out = hess_fun(rand_inp.transpose(-2, -1).detach().numpy())

        assert np.allclose(np.reshape(l4c_out, (12, 12)).transpose((-2, -1)), torch_out[0, 0, 0, 0].detach().numpy())

    def test_l4casadi_deep_model_jac(self, deep_model):
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = torch.func.vmap(torch.func.jacrev(deep_model))(rand_inp)[0]

        mx_inp = cs.MX.sym('x', deep_model.input_layer.in_features, 1)

        jac_fun = cs.Function('f_jac',
                              [mx_inp],
                              [cs.jacobian(l4c.L4CasADi(deep_model, model_expects_batch_dim=True)(mx_inp), mx_inp)])

        l4c_out = jac_fun(rand_inp.transpose(-2, -1).detach().numpy())

        assert np.allclose(l4c_out, torch_out.detach().numpy())

    def test_l4casadi_deep_model_hess(self):
        deep_model = DeepModel(4, 1)
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = torch.func.vmap(torch.func.hessian(deep_model))(rand_inp)[0]

        mx_inp = cs.MX.sym('x', deep_model.input_layer.in_features, 1)

        hess_fun = cs.Function('f_hess',
                               [mx_inp],
                               [cs.hessian(l4c.L4CasADi(deep_model, model_expects_batch_dim=True)(mx_inp), mx_inp)[0]])

        l4c_out = hess_fun(rand_inp.transpose(-2, -1).detach().numpy())

        assert np.allclose(l4c_out, torch_out.detach().numpy())

    def test_l4casadi_deep_model_hess_double_jac(self):
        deep_model = DeepModel(4, 2)
        rand_inp = torch.rand((1, deep_model.input_layer.in_features))
        torch_out = torch.func.vmap(torch.func.hessian(deep_model))(rand_inp)[0]

        mx_inp = cs.MX.sym('x', deep_model.input_layer.in_features, 1)

        hess_fun = cs.Function('f_hess_double_jac',
                               [mx_inp],
                               [cs.jacobian(
                                   cs.jacobian(
                                       l4c.L4CasADi(deep_model, model_expects_batch_dim=True)(mx_inp), mx_inp
                                   )[0], mx_inp)])

        l4c_out = hess_fun(rand_inp.transpose(-2, -1).detach().numpy())

        assert np.allclose(l4c_out, torch_out[0, 0].detach().numpy())

