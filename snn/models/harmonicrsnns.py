import torch
from .. import modules


class SimpleHarmonicRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            adaptive_omega_a: float,
            adaptive_omega_b: float,
            adaptive_b_offset_a: float,
            adaptive_b_offset_b: float,
            out_adaptive_tau_mem_mean: float,
            out_adaptive_tau_mem_std: float,
            label_last: bool,
            hidden_bias: bool = False,
            output_bias: bool = False
    ) -> None:
        super(SimpleHarmonicRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_last = label_last

        self.hidden = modules.HRFCell(
            input_size=input_size + hidden_size,  # recurrency. For no recurrency just input (and change forward pass)
            layer_size=hidden_size,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            bias=hidden_bias
        )

        self.out = modules.LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], float]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs_u = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)  # always send to device

        num_spikes = 0

        for t in range(sequence_length):
            input_t = x[t]
            hidden = hidden_z, hidden_u, hidden_v, hidden_a

            hidden_z, hidden_u, hidden_v, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # input_t
                hidden
            )

            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)
            outputs_u.append(out_u)  # indent for no LL

        outputs = torch.stack(outputs_u)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u,), out_u), num_spikes
