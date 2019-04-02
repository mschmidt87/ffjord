import torch
import torch.nn as nn
from torch.nn import functional as F
from pyro.nn import MaskedLinear, AutoRegressiveNN


class AutoRegressiveNNSquash(AutoRegressiveNN):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 param_dims=[1, 1],
                 permutation=None,
                 skip_connections=False,
                 nonlinearity=nn.ReLU()):
        super(AutoRegressiveNNSquash, self).__init__(input_dim,
                                                     hidden_dims,
                                                     param_dims=[1, 1],
                                                     permutation=None,
                                                     skip_connections=False,
                                                     nonlinearity=nn.ReLU())

        hyper_masks = [nn.Linear(1, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            hyper_masks.append(nn.Linear(1, hidden_dims[i]))
        hyper_masks.append(nn.Linear(1, input_dim * self.output_multiplier))
        self.hyper_masks = nn.ModuleList(hyper_masks)

    def forward(self, t, x):
        """
        The forward method
        """
        h = x
        for layer, hyper_mask in zip(self.layers[:-1], self.hyper_masks[:-1]):
            h = self.f(layer(h)) * torch.sigmoid(hyper_mask(t.view(1, 1)))
        h = self.layers[-1](h) * torch.sigmoid(self.hyper_masks[-1](t.view(1, 1)))

        if self.skip_layer is not None:
            h = h + self.skip_layer(x)
        # Shape the output, squeezing the parameter dimension if all ones
        if self.output_multiplier == 1:
            return h
        else:
            h = h.reshape(list(x.size()[:-1]) + [self.output_multiplier, self.input_dim])

            # Squeeze dimension if all parameters are one dimensional
            if self.count_params == 1:
                return h

            elif self.all_ones:
                return torch.unbind(h, dim=-2)

            # If not all ones, then probably don't want to squeeze a single dimension parameter
            else:
                return tuple([h[..., s, :] for s in self.param_slices])

