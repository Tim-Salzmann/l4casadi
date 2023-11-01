import torch.nn

from l4casadi.naive.decorator import casadi


class NaiveL4CasADiModule(torch.nn.Module):
    @casadi
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    @__call__.explicit
    def _casadi_call_(self, *args, **kwargs):
        try:
            return self.cs_forward(*args, **kwargs)
        except NotImplementedError:
            return super().__call__(*args, **kwargs)

    def cs_forward(self, *args, **kwargs):
        raise NotImplementedError
