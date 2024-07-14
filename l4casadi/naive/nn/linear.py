import torch
import casadi as cs

from l4casadi.naive import NaiveL4CasADiModule


class Linear(NaiveL4CasADiModule, torch.nn.Linear):
    def cs_forward(self, x):
        y = cs.mtimes(x, self.weight.transpose(1, 0).detach().numpy())
        if self.bias is not None:
            y = y + self.bias[None].repeat((x.shape[0], 1)).detach().numpy()
        return y
