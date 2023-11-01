import torch

from l4casadi.naive import NaiveL4CasADiModule
from l4casadi.naive.nn.linear import Linear
from l4casadi.naive.nn import activation as activations


class MultiLayerPerceptron(NaiveL4CasADiModule):
    def __init__(self, in_features, hidden_features, out_features, hidden_layers, activation=None):
        super().__init__()
        assert hidden_layers >= 1, 'There must be at least one hidden layer'
        self.input_size = in_features
        self.output_size = out_features
        self.input_layer = Linear(in_features, hidden_features)

        hidden = []
        for i in range(hidden_layers-1):
            hidden.append((Linear(hidden_features, hidden_features)))
        self.hidden_layers = torch.nn.ModuleList(hidden)

        self.output_layer = Linear(hidden_features, out_features)

        if activation is None:
            self.act = lambda x: x
        elif type(activation) is str:
            self.act = getattr(activations, activation)()
        else:
            self.act = activation

    def forward(self, x):
        x = self.input_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        y = self.output_layer(x)
        return y
