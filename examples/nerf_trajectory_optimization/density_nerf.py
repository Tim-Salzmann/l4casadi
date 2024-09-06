import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        multires = 10

        embed_fns = []
        d = 3
        out_dim = 0
        embed_fns.append(lambda x: x)
        out_dim += d

        max_freq = multires - 1
        N_freqs = multires

        freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DensityNeRF(nn.Module):
    """NeRF implementation just returning densities."""
    def __init__(self):
        super().__init__()

        self.embedder = Embedder()
        self.input_ch = self.embedder.out_dim
        self.skips = [4]

        D = 8
        W = 256
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [
                nn.Linear(W, W)
                if i != 4
                else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.alpha_linear = nn.Linear(W, 1)

    def forward(self, x):
        input_pts = self.embedder(x)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i == 4:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        alpha = torch.relu(alpha)

        return alpha
