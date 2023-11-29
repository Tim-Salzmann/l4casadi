import numpy as np
import torch
import casadi as cs
import l4casadi as l4c


class TestNaiveL4CasADi:

    def test_naive_l4casadi_mlp(self):
        naive_mlp = l4c.naive.MultiLayerPerceptron(2, 128, 2, 2, 'Tanh')

        rand_inp = torch.rand((1, 2))
        torch_out = naive_mlp(rand_inp)

        cs_inp = cs.DM(rand_inp.transpose(-2, -1).detach().numpy())

        l4c_out = l4c.L4CasADi(naive_mlp, model_expects_batch_dim=True)(cs_inp)

        assert np.allclose(l4c_out, torch_out.transpose(-2, -1).detach().numpy())
