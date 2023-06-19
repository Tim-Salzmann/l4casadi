import json
import os
import pathlib
from typing import Tuple

import casadi as cs
import torch
from torch.func import vmap, jacrev
from torch.fx.experimental.proxy_tensor import make_fx

from l4casadi.template_generation import render_template


class L4CasADi:
    def __init__(self, model: torch.nn.Module, has_batch: bool = False, name: str = 'l4casadi_f'):
        self.model = model
        self.name = name
        self.has_batch = has_batch

        self.generation_path = pathlib.Path('./_l4c_generated')

        self._ext_cs_fun = None
        self._ready = False

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, inp: Tuple[cs.MX, cs.SX, cs.DM]):
        if self.has_batch:
            assert inp.shape[-1] == 1, "For batched PyTorch models only vector inputs are allowed."
            inp = cs.transpose(inp)

        if not self._ready:
            self.get_ready(inp)

        out = self._ext_cs_fun(inp)
        return out

    def maybe_make_generation_dir(self):
        if not os.path.exists(self.generation_path):
            os.makedirs(self.generation_path)

    def get_ready(self, inp: Tuple[cs.MX, cs.SX, cs.DM]):
        rows, cols = inp.shape

        self.maybe_make_generation_dir()
        self.export_torch_traces(torch.zeros((rows, cols)))
        self.generate_cpp_function_template(rows, cols)
        self.compile_cs_function()

        self._ext_cs_fun = cs.external(f'{self.name}', f'{self.generation_path / self.name}.so')

        self._ready = True

    def generate_cpp_function_template(self, rows, cols):
        rows_out, cols_out = self.model(torch.zeros(rows, cols)).shape[-2:]

        gen_params = {
            'model_path': self.generation_path.as_posix(),
            'name': self.name,
            'rows_in': rows,
            'cols_in': cols,
            'rows_out': rows_out,
            'cols_out': cols_out
        }
        with open(self.generation_path / f'{self.name}.json', 'w') as f:
            json.dump(gen_params, f)

        render_template(
            'casadi_function.in.cpp',
            f'{self.name}.cpp',
            self.generation_path.as_posix(),
            f'{self.name}.json'
        )

    def compile_cs_function(self):
        file_dir = pathlib.Path(__file__).parent.resolve()
        include_dir = file_dir / 'cpp' / 'include'

        # call gcc
        os_cmd = ("gcc"
                  " -fPIC -shared"
                  f" {self.generation_path / self.name}.cpp"
                  f" -o {self.generation_path / self.name}.so"
                  f" -I{include_dir} -L{file_dir}" 
                  f" -rpath '{file_dir}'"
                  " -ll4casadi -lstdc++ -std=c++17")

        status = os.system(os_cmd)
        if status != 0:
            raise Exception(f'Compilation failed!\n\nAttempted to execute OS command:\n{os_cmd}\n\n')

    def export_torch_traces(self, torch_dummy_inp):
        d_inp = torch_dummy_inp
        out_folder = self.generation_path
        torch.jit.trace(self.model, d_inp).save(out_folder / f'{self.name}_forward.pt')
        torch.jit.trace(make_fx(vmap(jacrev(self.model)))(d_inp), d_inp).save(out_folder / f'{self.name}_jacrev.pt')

    def cs_function(self):
        return self._ext_cs_fun
