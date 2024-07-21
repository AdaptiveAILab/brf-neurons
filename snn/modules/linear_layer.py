import torch
from .. import functional


class LinearMask(torch.nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 lbd: int,
                 ubd: int,
                 mask_prob: float = 0.,
                 ):
        super(LinearMask, self).__init__(in_features, out_features, bias)

        # weight dim = (out_features, in_features) = (hidden_size, input_size + hidden_size)
        # RF init Xavier uniform
        torch.nn.init.xavier_uniform_(self.weight)

        if bias:
            torch.nn.init.constant_(self.bias, 0)

        # create mask
        mask = torch.ones_like(self.weight)

        # if masking required
        if mask_prob > 0:
            masked_region = torch.rand((out_features, ubd - lbd)) > mask_prob
            mask[:, lbd:ubd] *= masked_region
            print(masked_region.shape)
            print(masked_region)

        # save in buffer
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        self.weight.data *= self.mask

        return torch.nn.functional.linear(x, self.weight, self.bias)






