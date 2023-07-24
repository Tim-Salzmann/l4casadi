import json
import os
import pathlib
import platform
from importlib.resources import files
from typing import Union, Optional, Callable, Text, Tuple

import casadi as cs
import torch
from torch.func import vmap, jacrev, hessian
from l4casadi.ts_compiler import ts_compile
from torch.fx.experimental.proxy_tensor import make_fx

from l4casadi.template_generation import render_template


def dynamic_lib_file_ending():
    return '.dylib' if platform.system() == 'Darwin' else '.so'


class L4CasADi(object):
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 has_batch: bool = False, device: Union[torch.device, Text] = "cpu", name: Text = "l4casadi_f"):
        self.model = model
        if isinstance(self.model, torch.nn.Module):
            self.model.eval().to(device)
            for parameters in self.model.parameters():
                parameters.requires_grad = False
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
        has_jac, has_hess = self.export_torch_traces(rows, cols)

        if not has_jac:
            print('Jacobian trace could not be generated. First-order sensitivities will not be available in CasADi.')
        if not has_hess:
            print('Hessian trace could not be generated. Second-order sensitivities will not be available in CasADi.')

        self.generate_cpp_function_template(rows, cols, has_jac, has_hess)
        self.compile_cs_function()

        self._ext_cs_fun = cs.external(
            f'{self.name}',
            f"{self.generation_path / f'lib{self.name}'}{dynamic_lib_file_ending()}"
        )
        self._ready = True

    def generate_cpp_function_template(self, rows: int, cols: int, has_jac: bool, has_hess: bool):
        if self.has_batch:
            rows_out = self.model(torch.zeros(1, rows).to(self.device)).shape[-1]
            cols_out = 1
        else:
            out_shape = self.model(torch.zeros(rows, cols).to(self.device)).shape
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
            'has_jac': has_jac,
            'has_hess': has_hess,
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
        soname = 'install_name' if platform.system() == 'Darwin' else 'soname'
        os_cmd = ("gcc"
                  " -fPIC -shared"
                  f" {self.generation_path / self.name}.cpp"
                  f" -o {self.generation_path / f'lib{self.name}'}{dynamic_lib_file_ending()}"
                  f" -I{include_dir} -L{lib_dir}"
                  f" -Wl,-{soname},lib{self.name}{dynamic_lib_file_ending()}"
                  " -ll4casadi -lstdc++ -std=c++17"
                  " -D_GLIBCXX_USE_CXX11_ABI=0")

        status = os.system(os_cmd)
        if status != 0:
            raise Exception(f'Compilation failed!\n\nAttempted to execute OS command:\n{os_cmd}\n\n')

    def export_torch_traces(self, rows: int, cols: int) -> Tuple[bool, bool]:
        if self.has_batch:
            d_inp = torch.zeros((1, rows))
        else:
            d_inp = torch.zeros((rows, cols))
        d_inp = d_inp.to(self.device)

        out_folder = self.generation_path

        torch.jit.trace(self.model, d_inp).save((out_folder / f'{self.name}_forward.pt').as_posix())

        if self.has_batch:
            jac_model = make_fx(vmap(jacrev(self.model)))(d_inp)
            hess_model = make_fx(vmap(hessian(self.model)))(d_inp)
        else:
            jac_model = make_fx(jacrev(self.model))(d_inp)
            hess_model = make_fx(hessian(self.model))(d_inp)

        exported_jacrev = self._jit_compile_and_save(
            jac_model,
            (out_folder / f'{self.name}_jacrev.pt').as_posix(),
            d_inp
        )
        exported_hess = self._jit_compile_and_save(
            hess_model,
            (out_folder / f'{self.name}_hess.pt').as_posix(),
            d_inp
        )

        return exported_jacrev, exported_hess

    @staticmethod
    def _jit_compile_and_save(model, file_path: str, dummy_inp: torch.Tensor):
        # Try tracing
        try:
            torch.jit.trace(model, dummy_inp).save(file_path)
        except:  # noqa
            # Try scripting
            try:
                ts_compile(model).save(file_path)
            except:  # noqa
                return False
        return True
