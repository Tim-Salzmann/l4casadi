import torch
import casadi as cs

from l4casadi.naive import NaiveL4CasADiModule


class Linear(NaiveL4CasADiModule, torch.nn.Linear):
    def cs_forward(self, x):
        assert x.shape[1] == 1, 'Casadi can not handle batches.'
        y = cs.mtimes(self.weight.detach().numpy(), x)
        if self.bias is not None:
            y = y + self.bias.detach().numpy()
        return y
