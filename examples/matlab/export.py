import os
import torch
import casadi as cs
import l4casadi as l4c


class TorchModel(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


def generate():
    l4casadi_model = l4c.L4CasADi(TorchModel(), name='sin_l4c')
    sym_in = cs.MX.sym('x', 1, 1)
    l4casadi_model.build(sym_in)
    return


if __name__ == '__main__':
    print(f'Export LIB_DIR: {os.path.dirname(os.path.abspath(l4c.__file__))}/lib:{os.path.dirname(os.path.abspath(torch.__file__))}/lib')
    generate()
