import torch

from pyro.nn import AutoRegressiveNN
from .odefunc import NONLINEARITIES


class ODEMADEnet(AutoRegressiveNN):
    """
    ODE Wrapper around the Pyro implementation of MADE network.
    """

    def __init__(
            self, hidden_dims, input_shape, strides, conv,
            layer_type="concat", nonlinearity="softplus",
            num_squeeze=0
    ):
        super(ODEMADEnet, self).__init__(input_shape[0],
                                         hidden_dims,
                                         param_dims=[1, 1],
                                         permutation=torch.arange(input_shape[0]),
                                         nonlinearity=NONLINEARITIES[nonlinearity])

    def forward(self, t, y):
        mu, log_sigma = super(ODEMADEnet, self).forward(y)
        sigma = torch.exp(log_sigma)
        dx = sigma * y + mu
        return dx
