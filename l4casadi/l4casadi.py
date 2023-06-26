import json
import os
import pathlib
import sys
from importlib.resources import files
from typing import Union, Optional, Callable, Text

import casadi as cs
import torch
from torch.func import vmap, jacrev, hessian
from torch.fx.experimental.proxy_tensor import make_fx

from l4casadi.template_generation import render_template


class L4CasADi:
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 has_batch: bool = False, device: Union[torch.device, Text] = "cpu", name: Text = "l4casadi_f"):
        self.model = model
        self.name = name
        self.has_batch = has_batch
        self.device = device if isinstance(device, str) else f'{device.type}:{device.index}'

        self.generation_path = pathlib.Path('./_l4c_generated')

        self._ext_cs_fun: Optional[cs.Function] = None
        self._ready = False

    def __call__(self, *args):
        return self.forward(*args)

    @property
    def shared_lib_dir(self):
        return self.generation_path.absolute().as_posix()

    def forward(self, inp: Union[cs.MX, cs.SX, cs.DM]):
        if self.has_batch:
            if not inp.shape[-1] == 1:   # type: ignore[attr-defined]
                raise ValueError("For batched PyTorch models only vector inputs are allowed.")

        if not self._ready:
            self.get_ready(inp)

        out = self._ext_cs_fun(inp)  # type: ignore[misc]

        return out

    def maybe_make_generation_dir(self):
        if not os.path.exists(self.generation_path):
            os.makedirs(self.generation_path)

    def get_ready(self, inp: Union[cs.MX, cs.SX, cs.DM]):
        rows, cols = inp.shape  # type: ignore[attr-defined]

        self.maybe_make_generation_dir()
        self.export_torch_traces(rows, cols)
        self.generate_cpp_function_template(rows, cols)
        self.compile_cs_function()

        self._ext_cs_fun = cs.external(f'{self.name}', f"{self.generation_path / f'lib{self.name}'}.so")
        self._ready = True

    def generate_cpp_function_template(self, rows: int, cols: int):
        if self.has_batch:
            rows_out = self.model(torch.zeros(1, rows)).shape[-1]
            cols_out = 1
        else:
            out_shape = self.model(torch.zeros(rows, cols)).shape
            if len(out_shape) == 1:
                rows_out = out_shape[0]
                cols_out = 1
            else:
                rows_out, cols_out = out_shape[-2:]

        gen_params = {
            'model_path': self.generation_path.as_posix(),
            'device': self.device,
            'name': self.name,
            'rows_in': rows,
            'cols_in': cols,
            'rows_out': rows_out,
            'cols_out': cols_out,
            'has_batch': self.has_batch
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
        file_dir = files('l4casadi')
        include_dir = files('l4casadi') / 'include'
        lib_dir = file_dir / 'lib'

        # call gcc
        os_cmd = ("gcc"
                  " -fPIC -shared"
                  f" {self.generation_path / self.name}.cpp"
                  f" -o {self.generation_path / f'lib{self.name}'}.so"
                  f" -I{include_dir} -L{lib_dir}"
                  " -ll4casadi -lstdc++ -std=c++17"
                  " -D_GLIBCXX_USE_CXX11_ABI=0")

        status = os.system(os_cmd)
        if status != 0:
            raise Exception(f'Compilation failed!\n\nAttempted to execute OS command:\n{os_cmd}\n\n')

    def export_torch_traces(self, rows: int, cols: int):
        if self.has_batch:
            d_inp = torch.zeros((1, rows))
        else:
            d_inp = torch.zeros((rows, cols))

        out_folder = self.generation_path
        torch.jit.trace(self.model, d_inp).save((out_folder / f'{self.name}_forward.pt').as_posix())

        if self.has_batch:
            torch.jit.trace(
                make_fx(vmap(jacrev(self.model)))(d_inp), d_inp).save(
                (out_folder / f'{self.name}_jacrev.pt').as_posix())

            torch.jit.trace(
                make_fx(vmap(hessian(self.model)))(d_inp), d_inp).save(
                (out_folder / f'{self.name}_hess.pt').as_posix())
        else:
            torch.jit.trace(
                make_fx(jacrev(self.model))(d_inp), d_inp).save(
                (out_folder / f'{self.name}_jacrev.pt').as_posix())

            torch.jit.trace(
                make_fx(hessian(self.model))(d_inp), d_inp).save(
                (out_folder / f'{self.name}_hess.pt').as_posix())

