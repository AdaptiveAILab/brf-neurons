import torch
import math

################################################################
# Simple base functional
################################################################

@torch.jit.script
def step(x: torch.Tensor) -> torch.Tensor:
    #
    # x.gt(0.0).float()
    # is slightly faster (but less readable) than
    # torch.where(x > 0.0, 1.0, 0.0)
    #
    return x.gt(0.0).float()

@torch.jit.script
def exp_decay(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.abs(x))

@torch.jit.script
def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return (1 / (sigma * torch.sqrt(2 * torch.tensor(math.pi)))) * torch.exp(
        -((x - mu) ** 2) / (2.0 * (sigma ** 2))
    )


def gaussian_non_normalized(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    return torch.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))


def std_gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -0.5 * (x ** 2)
    )


def linear_peak(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - torch.abs(x))


def linear_peak_antiderivative(x: torch.Tensor) -> torch.Tensor:

    xa = torch.relu(1.0 - torch.abs(x))
    xa_sq = xa ** 2

    return 0.5 * torch.where(
        x < 0,
        xa_sq,
        2.0 - xa_sq
    )

@torch.jit.script
def DoubleGaussian(x: torch.Tensor) -> torch.Tensor:
    p = 0.15
    scale = 6.
    len = 0.5

    gamma = 0.5

    sigma1 = len
    sigma2 = scale * len
    return gamma * (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
    - p * gaussian(x, mu=len, sigma=sigma2) - p * gaussian(x, mu=-len, sigma=sigma2)


def quantize_tensor(tensor: torch.Tensor, f: int) -> torch.Tensor:
    # Quantization formula: tensor_q = round(2^f * tensor) * 2^(-f)
    return torch.round(2**f * tensor) * 0.5**f


def spike_deletion(hidden_z: torch.Tensor, spike_del_p: float) -> torch.Tensor:
    return hidden_z.mul(spike_del_p < torch.rand_like(hidden_z))
