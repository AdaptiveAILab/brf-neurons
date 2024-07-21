import torch
from .. import functional
from .linear_layer import LinearMask

################################################################
# Neuron update functional
################################################################

# default values for time constants
DEFAULT_ALIF_TAU_M = 20.
DEFAULT_ALIF_TAU_ADP = 20.

# base threshold
DEFAULT_ALIF_THETA = 0.01

DEFAULT_ALIF_BETA = 1.8


def alif_update(
        x: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor,
        a: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
        beta: float = DEFAULT_ALIF_BETA,
        theta: float = DEFAULT_ALIF_THETA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # adapt spike accumulator.
    a = a.mul(rho) + z.mul(1.0 - rho)

    # determine dynamic threshold.
    theta_t = theta + a.mul(beta)

    # membrane potential update
    u = u.mul(alpha) + x.mul(1.0 - alpha)

    # generate spike.
    z = functional.StepDoubleGaussianGrad.apply(u - theta_t)

    # reset membrane potential.
    # soft reset (keeps remaining membrane potential)
    u = u - z.mul(theta_t)

    return z, u, a


################################################################
# Layer classes
################################################################

class ALIFCell(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            layer_size: int,
            adaptive_tau_mem_mean: float,
            adaptive_tau_mem_std: float,
            adaptive_tau_adp_mean: float,
            adaptive_tau_adp_std: float,
            tau_mem: float = DEFAULT_ALIF_TAU_M,  # time constant for alpha
            adaptive_tau_mem: bool = True,
            tau_adp: float = DEFAULT_ALIF_TAU_ADP,  # time constant for rho
            adaptive_tau_adp: bool = True,
            bias: bool = False,
            mask_prob: float = 0.,
            pruning: bool = False,
    ) -> None:
        super(ALIFCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        # LinearMask: pruning of hidden recurrent weights in forward pass
        if pruning:

            self.mask_prob = mask_prob

            self.linear = LinearMask(
                in_features=input_size,
                out_features=layer_size,
                bias=bias,
                mask_prob=mask_prob,
                lbd=input_size - layer_size,
                ubd=input_size,
            )

        else:

            self.linear = torch.nn.Linear(
                in_features=input_size,
                out_features=layer_size,
                bias=bias
            )

            torch.nn.init.xavier_uniform_(self.linear.weight)

        self.adaptive_tau_mem = adaptive_tau_mem
        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std

        tau_mem = tau_mem * torch.ones(layer_size)

        if adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=adaptive_tau_mem_mean, std=adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tau_mem)

        self.adaptive_tau_adp = adaptive_tau_adp
        self.adaptive_tau_adp_mean = adaptive_tau_adp_mean
        self.adaptive_tau_adp_mean = adaptive_tau_adp_std

        tau_adp = tau_adp * torch.ones(layer_size)

        if adaptive_tau_adp:
            self.tau_adp = torch.nn.Parameter(tau_adp)
            torch.nn.init.normal_(self.tau_adp, mean=adaptive_tau_adp_mean, std=adaptive_tau_adp_std)
        else:
            self.register_buffer("tau_adp", tau_adp)

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, u, a = state

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)

        alpha = torch.exp(-1. * 1 / tau_mem)

        tau_adp = torch.abs(self.tau_adp)

        rho = torch.exp(-1. * 1 / tau_adp)

        z, u, a = alif_update(
            x=in_sum,
            z=z,
            u=u,
            a=a,
            alpha=alpha,
            rho=rho,
        )

        return z, u, a


class ALIFCellBP(ALIFCell):
    def __init__(self,
                 *args,
                 bit_precision: int = 32,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.bit_precision = bit_precision

    def forward(
            self, x: torch.Tensor,
            state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, u, a = state

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)

        alpha = torch.exp(-1. * 1 / tau_mem)
        alpha = functional.quantize_tensor(alpha, self.bit_precision)

        tau_adp = torch.abs(self.tau_adp)

        rho = torch.exp(-1. * 1 / tau_adp)
        rho = functional.quantize_tensor(rho, self.bit_precision)

        z, u, a = alif_update(
            x=in_sum,
            z=z,
            u=u,
            a=a,
            alpha=alpha,
            rho=rho,
        )

        return z, u, a


