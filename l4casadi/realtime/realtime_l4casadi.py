from typing import Union, Callable, Text, List, Optional

import casadi as cs
import numpy as np
import torch

from l4casadi import L4CasADi
from .sensitivities import batched_jacobian, batched_hessian


class RealTimeL4CasADi(L4CasADi):
    def __init__(self,
                 model: Callable[[torch.Tensor], torch.Tensor],
                 approximation_order: int = 1,
                 device: Union[torch.device, Text] = 'cpu',
                 name: Text = 'rt_l4casadi_f'):
        """
        :param model: PyTorch model.
        :param approximation_order: Order of the Taylor approximation. 1 for linearization, 2 for quadratic
            approximation.
        :param device: Device on which the PyTorch model is executed.
        :param name: Unique name of the generated L4CasADi model. This name is used for autogenerated files.
            Creating two L4CasADi models with the same name will result in overwriting the files of the first model.
        """
        super().__init__(model, device=device, name=name)

        if approximation_order > 2 or approximation_order < 1:
            raise ValueError("Taylor approximation order must be 1 or 2.")

        self._approximation_order = approximation_order
        self._taylor_params = None

    @property
    def order(self):
        return self._approximation_order

    def _init_taylor_params(self, rows_in: int, rows_out: int):
        a = cs.MX.sym('a', rows_in, 1)
        f_a = cs.MX.sym('f_a', rows_out, 1)
        df_a = cs.MX.sym('df_a', rows_out, rows_in)

        if self.order == 2:
            ddf_as = []
            for i in range(rows_out):
                ddf_a_i = cs.MX.sym(f'ddf_a_{i}', rows_in, rows_in)
                ddf_as.append(ddf_a_i)
            return a, f_a, df_a, ddf_as
        else:
            return a, f_a, df_a

    @property
    def sym_params(self) -> List[cs.MX]:
        if self._taylor_params is None:
            raise RuntimeError("Model must be called before getting symbolic parameters.")
        return self._flatten_taylor_params(self._taylor_params)

    def get_sym_params(self):
        if len(self.sym_params) == 0:
            return cs.vertcat([])
        return cs.vcat([cs.reshape(mx, np.prod(mx.shape), 1) for mx in self.sym_params])

    def _get_params(self, a_t: torch.Tensor):
        if len(a_t.shape) == 1:
            a_t = a_t.unsqueeze(0)
        if self._approximation_order == 1:
            df_a, f_a = batched_jacobian(self.model, a_t, return_func_output=True)
            return [a_t.cpu().numpy(), f_a.cpu().numpy(), df_a.transpose(-2, -1).cpu().numpy()]
        elif self._approximation_order == 2:
            ddf_a, df_a, f_a = batched_hessian(self.model, a_t, return_func_output=True, return_jacobian=True)
            return ([a_t.cpu().numpy(), f_a.cpu().numpy(), df_a.transpose(-2, -1).cpu().numpy()]
                    + [ddf_a[:, i].transpose(-2, -1).cpu().numpy() for i in range(ddf_a.shape[1])])

    def get_params(self, a: Union[np.ndarray, torch.Tensor]):
        a_t = torch.tensor(a).float().to(self.device)
        params = self._get_params(a_t)

        if len(params) == 0:
            return np.array([])
        if len(a.shape) > 1:
            return np.hstack([p.reshape(p.shape[0], -1) for p in params])
        return np.hstack([p.flatten() for p in params])

    def taylor_approx(self, x: cs.MX, a: cs.MX, f_a: cs.MX, df_a: cs.MX,
                      ddf_a: Optional[List[cs.MX]] = None, parallel=False):
        """
        Approximation using first or second order Taylor Expansion
        """
        x_minus_a = x - a
        if ddf_a is None:
            return (f_a
                    + cs.mtimes(df_a, x_minus_a))
        else:
            if parallel:
                # Using OpenMP to parallel compute second order term of Taylor for all output dims

                def second_order_oi_term(x_minus_a, f_ddf_a):
                    return cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a))

                ddf_a_expl = ddf_a[3]
                x_minus_a_exp = cs.MX.sym('x_minus_a_exp', x_minus_a.shape[0], x_minus_a.shape[1])
                second_order_term_oi_fun = cs.Function('second_order_term_fun',
                                                       [x_minus_a_exp, ddf_a_expl],
                                                       [second_order_oi_term(x_minus_a_exp, ddf_a_expl)])

                n_o = f_a.shape[0]

                second_order_term_fun = second_order_term_oi_fun.map(n_o, 'openmp')

                x_minus_a_rep = cs.repmat(x_minus_a, 1, n_o)
                f_ddf_a_stack = cs.hcat(ddf_a)

                second_order_term = 0.5 * cs.transpose(second_order_term_fun(x_minus_a_rep, f_ddf_a_stack))
            else:
                f_ddf_as = ddf_a
                second_order_term = 0.5 * cs.vcat(
                        [cs.mtimes(cs.transpose(x_minus_a), cs.mtimes(f_ddf_a, x_minus_a))
                         for f_ddf_a in f_ddf_as])

            return (f_a
                    + cs.mtimes(df_a, x_minus_a)
                    + second_order_term)

    @staticmethod
    def _flatten_taylor_params(taylor_params):
        flat_params = list()
        for param in taylor_params:
            if isinstance(param, cs.MX):
                flat_params.append(param)
            else:
                flat_params.extend(param)
        return flat_params

    def build(self, inp: Union[cs.MX, cs.SX, cs.DM]) -> None:
        rows, cols = inp.shape  # type: ignore[attr-defined]
        rows_out = self.model(torch.zeros(1, rows).to(self.device)).shape[-1]

        self._taylor_params = self._init_taylor_params(rows, rows_out)
        self._cs_fun = cs.Function(
            f'taylor_approx_{self.name}',
            [inp] + self.sym_params,
            [self.taylor_approx(inp, *self._taylor_params)])  # type: ignore[misc]

        self._built = True

    def forward(self, inp: Union[cs.MX, cs.SX, cs.DM]):
        if not inp.shape[-1] == 1:  # type: ignore[attr-defined]
            raise ValueError("RealTimeL4CasADi only accepts vector inputs.")

        if not self._built:
            self.build(inp)

        out = self._cs_fun(inp, *self.sym_params)  # type: ignore[misc]

        return out
