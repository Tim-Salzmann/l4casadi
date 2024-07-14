import pytest
import torch
import l4casadi as l4c
import casadi as cs
import numpy as np


class TestL4CasADiBatching:
    @pytest.mark.parametrize("batch_size,input_size,output_size,jac_ccs_target,hess_ccs_target", [
        (10, 3, 2, [20, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19, 0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17, 8, 18, 9, 19], [600, 30, 0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 0, 10, 200, 210, 400, 410, 21, 31, 221, 231, 421, 431, 42, 52, 242, 252, 442, 452, 63, 73, 263, 273, 463, 473, 84, 94, 284, 294, 484, 494, 105, 115, 305, 315, 505, 515, 126, 136, 326, 336, 526, 536, 147, 157, 347, 357, 547, 557, 168, 178, 368, 378, 568, 578, 189, 199, 389, 399, 589, 599, 0, 10, 200, 210, 400, 410, 21, 31, 221, 231, 421, 431, 42, 52, 242, 252, 442, 452, 63, 73, 263, 273, 463, 473, 84, 94, 284, 294, 484, 494, 105, 115, 305, 315, 505, 515, 126, 136, 326, 336, 526, 536, 147, 157, 347, 357, 547, 557, 168, 178, 368, 378, 568, 578, 189, 199, 389, 399, 589, 599, 0, 10, 200, 210, 400, 410, 21, 31, 221, 231, 421, 431, 42, 52, 242, 252, 442, 452, 63, 73, 263, 273, 463, 473, 84, 94, 284, 294, 484, 494, 105, 115, 305, 315, 505, 515, 126, 136, 326, 336, 526, 536, 147, 157, 347, 357, 547, 557, 168, 178, 368, 378, 568, 578, 189, 199, 389, 399, 589, 599]),
        (3, 4, 3, [9, 12, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 0, 3, 6, 1, 4, 7, 2, 5, 8, 0, 3, 6, 1, 4, 7, 2, 5, 8, 0, 3, 6, 1, 4, 7, 2, 5, 8, 0, 3, 6, 1, 4, 7, 2, 5, 8], [108, 12, 0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 0, 3, 6, 27, 30, 33, 54, 57, 60, 81, 84, 87, 10, 13, 16, 37, 40, 43, 64, 67, 70, 91, 94, 97, 20, 23, 26, 47, 50, 53, 74, 77, 80, 101, 104, 107, 0, 3, 6, 27, 30, 33, 54, 57, 60, 81, 84, 87, 10, 13, 16, 37, 40, 43, 64, 67, 70, 91, 94, 97, 20, 23, 26, 47, 50, 53, 74, 77, 80, 101, 104, 107, 0, 3, 6, 27, 30, 33, 54, 57, 60, 81, 84, 87, 10, 13, 16, 37, 40, 43, 64, 67, 70, 91, 94, 97, 20, 23, 26, 47, 50, 53, 74, 77, 80, 101, 104, 107, 0, 3, 6, 27, 30, 33, 54, 57, 60, 81, 84, 87, 10, 13, 16, 37, 40, 43, 64, 67, 70, 91, 94, 97, 20, 23, 26, 47, 50, 53, 74, 77, 80, 101, 104, 107])
    ])
    def test_ccs(self, batch_size, input_size, output_size, jac_ccs_target, hess_ccs_target):
        jac_ccs, hess_ccs = l4c.L4CasADi.generate_block_diagonal_ccs(batch_size, input_size, output_size)

        assert jac_ccs == jac_ccs_target
        assert hess_ccs == hess_ccs_target

    def test_l4casadi_sparse_out(self):
        def model(x):
            return torch.stack([(x[:, 0]**2 * x[:, 1]**2 * x[:, 2]**2), - (x[:, 0]**2 * x[:, 1]**2)], dim=-1)

        def model_cs(x):
            return cs.hcat([(x[:, 0]**2 * x[:, 1]**2 * x[:, 2]**2), - (x[:, 0]**2 * x[:, 1]**2)])

        inp = np.ones((5, 3))
        inp_sym = cs.MX.sym('x', 5, 3)

        jac_func_cs = cs.Function('f', [inp_sym], [cs.jacobian(model_cs(inp_sym), inp_sym)])
        jac_sparse_cs = jac_func_cs(inp)

        hess_func_cs = cs.Function('f', [inp_sym], [cs.jacobian(cs.jacobian(model_cs(inp_sym), inp_sym), inp_sym)])
        hess_sparse_cs = hess_func_cs(inp)

        l4c_model = l4c.L4CasADi(model, batched=True, generate_jac_jac=True)

        jac_func = cs.Function('f', [inp_sym], [cs.jacobian(l4c_model(inp_sym), inp_sym)])
        jac_sparse = jac_func(inp)

        hess_func = cs.Function('f', [inp_sym], [cs.jacobian(cs.jacobian(l4c_model(inp_sym), inp_sym), inp_sym)])
        hess_sparse = hess_func(inp)

        assert np.allclose(np.array(jac_sparse), np.array(jac_sparse_cs))
        assert np.allclose(np.array(hess_sparse), np.array(hess_sparse_cs))

    def test_l4casadi_sparse_out_adj1(self):
        def model(x):
            return torch.stack([(x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2), - (x[:, 0] ** 2 * x[:, 1] ** 2)], dim=-1)

        def model_cs(x):
            return cs.hcat([(x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2), -(x[:, 0] ** 2 * x[:, 1] ** 2)])

        inp = np.ones((5, 3))
        tangent = np.zeros((5, 2))
        tangent[:, 0] = 1.

        inp_sym = cs.MX.sym('x', 5, 3)
        tangent_sym = cs.MX.sym('x', 5, 2)

        func_cs = cs.Function('f', [inp_sym], [model_cs(inp_sym)])
        adj1_func_cs = func_cs.reverse(1)

        out_sym = func_cs(inp_sym)
        out_cs = func_cs(inp)
        adj1_out_cs = adj1_func_cs(inp, out_cs, tangent)


        l4c_model = l4c.L4CasADi(model, batched=True)
        y = l4c_model(inp_sym)

        func_t = l4c_model._cs_fun
        adj1_func_t = func_t.reverse(1)

        out_t = func_t(inp)
        adj1_out_t = adj1_func_t(inp, out_t, tangent)

        assert (np.array(adj1_out_cs) == np.array(adj1_out_t)).all()

        jac_adj1_func_cs = cs.Function('jac_adj1_f', [inp_sym, tangent_sym],
                                       [cs.jacobian(adj1_func_cs(inp_sym, out_sym, tangent_sym), inp_sym)])
        jac_adj1_cs = jac_adj1_func_cs(inp, tangent)

        jac_adj1_func_t = cs.Function('jac_adj1_ft', [inp_sym, tangent_sym],
                                      [cs.jacobian(adj1_func_t(inp_sym, func_t(inp_sym), tangent_sym), inp_sym)])
        jac_adj1_t = jac_adj1_func_t(inp, tangent)

        assert (np.array(jac_adj1_cs) == np.array(jac_adj1_t)).all()

