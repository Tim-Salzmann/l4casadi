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

    sym_out = l4casadi_model(sym_in)

    f = cs.Function('sin_torch', [sym_in], [sym_out])

    f.generate('sin.cpp', {'main': True, 'with_header': True})

    return l4casadi_model.shared_lib_dir


if __name__ == '__main__':
    print(f'L4CASADI_LIB_DIR: {os.path.dirname(os.path.abspath(l4c.__file__))}/lib')
    print(f'TORCH_LIB_DIR: {os.path.dirname(os.path.abspath(torch.__file__))}/lib')
    gen_lib_dir = generate()
    print(f'L4CASADI_GEN_LIB_DIR: {gen_lib_dir}')
