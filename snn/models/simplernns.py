import torch

from .. import modules, functional

# alpha parameter initialization
DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_ALIF_ADAPTIVE_TAU_M_STD = 5.


# rho parameter initialization
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN = 20.
DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD = 5.


# Input + ALIF + LI
class SimpleALIFRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            pruning: bool = False,
            adaptive_tau_mem: bool = True,  # adaptive time constant for alpha
            adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            adaptive_tau_adp: bool = True,  # adaptive time constant for rho
            adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN,
            adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD,
            out_adaptive_tau: bool = True,  # adaptive time constant for LI output
            out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            hidden_bias: bool = False,
            output_bias: bool = False,
            sub_seq_length: int = 0,
            mask_prob: float = 0.,
            label_last: bool = False,
            n_last: int = 1,
    ) -> None:
        super(SimpleALIFRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length
        self.label_last = label_last
        self.n_last = n_last

        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std

        self.adaptive_tau_adp_mean = adaptive_tau_adp_mean
        self.adaptive_tau_adp_std = adaptive_tau_adp_std

        self.out_adaptive_tau_mem_mean = out_adaptive_tau_mem_mean
        self.out_adaptive_tau_mem_std = out_adaptive_tau_mem_std

        self.hidden_bias = hidden_bias
        self.output_bias = output_bias

        self.hidden = modules.ALIFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            adaptive_tau_mem=adaptive_tau_mem,
            adaptive_tau_mem_mean=adaptive_tau_mem_mean,
            adaptive_tau_mem_std=adaptive_tau_mem_std,
            adaptive_tau_adp=adaptive_tau_adp,
            adaptive_tau_adp_mean=adaptive_tau_adp_mean,
            adaptive_tau_adp_std=adaptive_tau_adp_std,
            bias=hidden_bias,
            mask_prob=mask_prob,
            pruning=pruning
        )

        self.out = modules.LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=out_adaptive_tau,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

    def forward(
            self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], int]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        num_spikes = 0

        for t in range(sequence_length):
            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_a

            hidden_z, hidden_u, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            # sop
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-self.n_last:, :, :]

        return outputs, ((hidden_z, hidden_u, hidden_a), out_u), num_spikes


class SimpleALIFRNNTbptt(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            criterion: torch.nn.Module,
            pruning: bool = False,
            adaptive_tau_mem: bool = True,  # adaptive time constant for alpha
            adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            adaptive_tau_adp: bool = True,  # adaptive time constant for rho
            adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN,
            adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD,
            out_adaptive_tau: bool = True,  # adaptive time constant for LI output
            out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            hidden_bias: bool = False,
            output_bias: bool = False,
            tbptt_steps: int = 50,
            sub_seq_length: int = 0,
            mask_prob: float = 0.,
            label_last: bool = False,
            n_last: int = 1,
    ) -> None:
        super(SimpleALIFRNNTbptt, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length
        self.label_last = label_last
        self.n_last = n_last

        self.tbptt_steps = tbptt_steps

        self.criterion = criterion

        self.hidden = modules.ALIFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            adaptive_tau_mem=adaptive_tau_mem,
            adaptive_tau_mem_mean=adaptive_tau_mem_mean,
            adaptive_tau_mem_std=adaptive_tau_mem_std,
            adaptive_tau_adp=adaptive_tau_adp,
            adaptive_tau_adp_mean=adaptive_tau_adp_mean,
            adaptive_tau_adp_std=adaptive_tau_adp_std,
            bias=hidden_bias,
            mask_prob=mask_prob,
            pruning=pruning
        )

        self.out = modules.LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=out_adaptive_tau,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias
        )

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            optimizer=None,
    ) -> tuple[torch.Tensor, int, int]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        # y = y.clone().argmax(dim=2)

        outputs = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        loss_val = torch.zeros(1).to(x.device)
        total_loss = 0
        total_loss_ll = 0
        num_spikes = 0

        for t in range(sequence_length):

            input_t = x[t]
            target_t = y[t]

            hidden = hidden_z, hidden_u, hidden_a

            hidden_z, hidden_u, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # input_t
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            if t >= self.sub_seq_length:

                outputs.append(out_u)

                if isinstance(self.criterion, torch.nn.NLLLoss):
                    out = torch.nn.functional.log_softmax(out_u, dim=1)
                else:
                    out = out_u

                # update running loss
                loss = self.criterion(out, target_t)
                loss_val += loss

                # for printing
                total_loss += loss.item()

                # update the weights at each truncation step
                if t % self.tbptt_steps == 0 and self.training and loss_val.item() != 0:

                    if self.label_last:
                        loss_val = loss
                        total_loss_ll += loss.item()

                    loss_val.backward()

                    optimizer.step()

                    # reset loss
                    loss_val = torch.zeros_like(loss)

                    # detach hidden and output states
                    hidden_z.detach_()
                    hidden_u.detach_()
                    hidden_a.detach_()

                    out_u.detach_()

                    # reset optimizer
                    optimizer.zero_grad()

                # compute last truncation chunk
                if t == (sequence_length - 1) and loss_val.item() != 0 and self.training:

                    if self.label_last:
                        loss_val = loss
                        total_loss_ll += loss.item()

                    loss_val.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-self.n_last:, :, :]
            total_loss = total_loss_ll

        return outputs, total_loss, num_spikes


class DoubleALIFRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden1_size: int,
            hidden2_size: int,
            output_size: int,
            adaptive_tau_mem: bool = True,  # adaptive time constant for alpha
            adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            adaptive_tau_adp: bool = True,  # adaptive time constant for rho
            adaptive_tau_adp_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_MEAN,
            adaptive_tau_adp_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_ADP_STD,
            out_adaptive_tau: bool = True,  # adaptive time constant for LI output
            out_adaptive_tau_mem_mean: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_MEAN,
            out_adaptive_tau_mem_std: float = DEFAULT_ALIF_ADAPTIVE_TAU_M_STD,
            hidden1_bias: bool = False,
            hidden2_bias: bool = False,
            output_bias: bool = False,
            sub_seq_length: int = 0,
            mask_prob: float = 0.,
            label_last: bool = False
    ) -> None:
        super(DoubleALIFRNN, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length
        self.label_last = label_last

        self.hidden1 = modules.ALIFCell(
            input_size=input_size + hidden1_size,
            layer_size=hidden1_size,
            adaptive_tau_mem=adaptive_tau_mem,
            adaptive_tau_mem_mean=adaptive_tau_mem_mean,
            adaptive_tau_mem_std=adaptive_tau_mem_std,
            adaptive_tau_adp=adaptive_tau_adp,
            adaptive_tau_adp_mean=adaptive_tau_adp_mean,
            adaptive_tau_adp_std=adaptive_tau_adp_std,
            bias=hidden1_bias,
            mask_prob=mask_prob
        )

        self.hidden2 = modules.ALIFCell(
            input_size=hidden1_size + hidden2_size,
            layer_size=hidden2_size,
            adaptive_tau_mem=adaptive_tau_mem,
            adaptive_tau_mem_mean=adaptive_tau_mem_mean,
            adaptive_tau_mem_std=adaptive_tau_mem_std,
            adaptive_tau_adp=adaptive_tau_adp,
            adaptive_tau_adp_mean=adaptive_tau_adp_mean,
            adaptive_tau_adp_std=adaptive_tau_adp_std,
            bias=hidden2_bias,
            mask_prob=mask_prob
        )

        self.out = modules.LICell(
            input_size=hidden2_size,
            layer_size=output_size,
            adaptive_tau_mem=out_adaptive_tau,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

    def forward(
            self, x: torch.Tensor,
    ) -> tuple[torch.Tensor,
    tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        hidden1_z = torch.zeros((batch_size, self.hidden1_size)).to(x.device)
        hidden1_u = torch.zeros_like(hidden1_z)
        hidden1_a = torch.zeros_like(hidden1_z)

        hidden2_z = torch.zeros((batch_size, self.hidden2_size)).to(x.device)
        hidden2_u = torch.zeros_like(hidden2_z)
        hidden2_a = torch.zeros_like(hidden2_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        outputs = list()

        for t in range(sequence_length):
            input_t = x[t]

            hidden1 = hidden1_z, hidden1_u, hidden1_a

            hidden1_z, hidden1_u, hidden1_a = self.hidden1(
                torch.cat((input_t, hidden1_z), dim=1),
                hidden1
            )

            hidden2 = hidden2_z, hidden2_u, hidden2_a

            hidden2_z, hidden2_u, hidden2_a = self.hidden2(
                torch.cat((hidden1_z, hidden2_z), dim=1),
                hidden2
            )

            out_u = self.out(hidden2_z, out_u)

            # accumulate outputs with sub_seq_length delay
            if t > self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1, :, :].unsqueeze(dim=0)

        return outputs, ((hidden1_z, hidden1_u, hidden1_a, hidden2_z, hidden2_u, hidden2_a), out_u)


# specifically for spike deletion
class ALIFRSNN_SD(SimpleALIFRNN):
    def __init__(
            self,
            *args,
            spike_del_p: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.spike_del_p = spike_del_p

    def forward(
            self, x: torch.Tensor,
            state: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], int]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        num_spikes = 0

        for t in range(sequence_length):
            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_a

            hidden_z, hidden_u, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            hidden_z = functional.spike_deletion(hidden_z, self.spike_del_p)

            # sop
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u, hidden_a), out_u), num_spikes


class ALIFRSNN_BP(SimpleALIFRNN):
    def __init__(
            self,
            bit_precision: int = 52,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.bit_precision = bit_precision

        self.hidden = modules.ALIFCellBP(
            input_size=self.input_size + self.hidden_size,
            layer_size=self.hidden_size,
            adaptive_tau_mem_mean=self.adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.adaptive_tau_mem_std,
            adaptive_tau_adp_mean=self.adaptive_tau_adp_mean,
            adaptive_tau_adp_std=self.adaptive_tau_adp_std,
            bias=self.hidden_bias,
            bit_precision=bit_precision
        )

        self.out = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std,
            bit_precision=bit_precision,
            bias=self.output_bias
        )

    def forward(
            self, x: torch.Tensor,
            state: tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor] = None
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor], int]:
        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        x = functional.quantize_tensor(x, self.bit_precision)
        hidden_z = functional.quantize_tensor(torch.zeros((batch_size, self.hidden_size)).to(x.device), self.bit_precision)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = functional.quantize_tensor(torch.zeros((batch_size, self.output_size)).to(x.device), self.bit_precision)

        num_spikes = 0

        for t in range(sequence_length):
            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_a

            hidden_z, hidden_u, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            # sop
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u, hidden_a), out_u), num_spikes
