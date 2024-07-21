import torch
from .base import *


################################################################
# Autograd function classes
################################################################

class StepGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = gaussian(x)
        return grad_output * dfdx


class StepLinearGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = torch.relu(1.0 - torch.abs(x))
        return grad_output * dfdx


class StepExpGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        dfdx = torch.exp(-torch.abs(x))
        return grad_output * dfdx


class StepDoubleGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

        return grad_output * dfd * gamma


class StepMultiGaussianGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return step(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors

        p = 0.15
        scale = 6.
        len = 0.5

        sigma1 = len
        sigma2 = scale * len

        gamma = 0.5
        dfd = (1. + p) * gaussian(x, mu=0., sigma=sigma1) \
              - p * gaussian(x, mu=len, sigma=sigma2) - p * gaussian(x, mu=-len, sigma=sigma2)

        return grad_output * dfd * gamma


def FGI_DGaussian(x: torch.Tensor) -> torch.Tensor:

    x_detached = step(x).detach()

    p = 0.15
    scale = 6.
    len = 0.5

    sigma1 = len
    sigma2 = scale * len

    gamma = 0.5

    df = (1. + p) * gaussian(x, mu=0., sigma=sigma1) - 2. * p * gaussian(x, mu=0., sigma=sigma2)

    df_detached = df.detach()

    # detach of df prevents the gradients to flow through x of the gaussian function.
    dfd = gamma * df_detached * x

    dfd_detached = dfd.detach()

    return dfd - dfd_detached + x_detached
