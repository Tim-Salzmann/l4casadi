import torch
import casadi as cs

from l4casadi.naive import NaiveL4CasADiModule


class Sigmoid(NaiveL4CasADiModule, torch.nn.Sigmoid):
    def cs_forward(self, x):
        y = 1 / (1 + cs.exp(-x))
        return y


class Tanh(NaiveL4CasADiModule, torch.nn.Tanh):
    def cs_forward(self, x):
        return cs.tanh(x)


class ReLU(NaiveL4CasADiModule, torch.nn.ReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., 0. * x, x)


class LeakyReLU(NaiveL4CasADiModule, torch.nn.LeakyReLU):
    def cs_forward(self, x):
        return cs.if_else(x < 0., self.negative_slope * x, x)
