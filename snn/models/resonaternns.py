import torch
from .. import modules, functional


class SimpleResRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            pruning: bool = False,
            adaptive_omega_a: float = 5.,
            adaptive_omega_b: float = 10.,
            adaptive_b_offset_a: float = 0.,
            adaptive_b_offset_b: float = 1.,
            out_adaptive_tau_mem_mean: float = 20.,
            out_adaptive_tau_mem_std: float = 5.,
            n_last: int = 1,
            mask_prob: float = 0.,
            sub_seq_length: int = 0,
            hidden_bias: bool = False,
            output_bias: bool = False,
            label_last: bool = False,
            dt: float = 0.01,
    ) -> None:
        super(SimpleResRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length

        self.label_last = label_last
        self.n_last = n_last

        self.mask_prob = mask_prob

        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        self.adaptive_b_offset_a = adaptive_b_offset_a
        self.adaptive_b_offset_b = adaptive_b_offset_b

        self.out_adaptive_tau_mem_mean = out_adaptive_tau_mem_mean
        self.out_adaptive_tau_mem_std = out_adaptive_tau_mem_std

        self.hidden = modules.BRFCell(
            input_size=input_size + hidden_size,  # only input_size for non-recurrency
            layer_size=hidden_size,
            bias=hidden_bias,
            mask_prob=mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
            pruning=pruning
        )

        self.out = modules.LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = torch.tensor(0.).to(x.device)

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_q = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v, hidden_q

            hidden_z, hidden_u, hidden_v, hidden_q = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # input_t for non-recurrency
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-self.n_last:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes


# BASIC RF MODEL IMPLEMENTED
class SimpleVanillaRFRNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            pruning: bool = False,
            adaptive_omega_a: float = 5.,
            adaptive_omega_b: float = 10.,
            adaptive_b_offset_a: float = 0.,
            adaptive_b_offset_b: float = 1.,
            out_adaptive_tau_mem_mean: float = 20.,
            out_adaptive_tau_mem_std: float = 5.,
            n_last: int = 1,
            mask_prob: float = 0.,
            sub_seq_length: int = 0,
            hidden_bias: bool = False,
            output_bias: bool = False,
            label_last: bool = False,
            dt: float = 0.01,
    ) -> None:
        super(SimpleVanillaRFRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.sub_seq_length = sub_seq_length

        self.label_last = label_last
        self.n_last = n_last

        self.mask_prob = mask_prob

        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        self.adaptive_b_offset_a = adaptive_b_offset_a
        self.adaptive_b_offset_b = adaptive_b_offset_b

        self.out_adaptive_tau_mem_mean = out_adaptive_tau_mem_mean
        self.out_adaptive_tau_mem_std = out_adaptive_tau_mem_std

        self.hidden = modules.RFCell(
            input_size=input_size + hidden_size,  # only input_size for non-recurrency
            layer_size=hidden_size,
            bias=hidden_bias,
            mask_prob=mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            dt=dt,
            pruning=pruning
        )

        self.out = modules.LICell(
            input_size=hidden_size,
            layer_size=output_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias,
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[
        tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = torch.tensor(0.).to(x.device)

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v

            hidden_z, hidden_u, hidden_v = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # only input_t for non-recurrency
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-self.n_last:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes


class SimpleResRNNTbptt(torch.nn.Module):
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
            criterion: torch.nn.Module,
            mask_prob: float = 0,
            tbptt_steps: int = 50,
            sub_seq_length: int = 0,
            label_last: bool = False,
            hidden_bias: bool = False,
            output_bias: bool = False,
    ) -> None:
        super(SimpleResRNNTbptt, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.criterion = criterion
        self.tbptt_steps = tbptt_steps

        self.sub_seq_length = sub_seq_length

        self.label_last = label_last

        self.mask_prob = mask_prob

        self.hidden = modules.BRFCell(
            input_size=input_size + hidden_size,
            layer_size=hidden_size,
            bias=hidden_bias,
            mask_prob=mask_prob,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b
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
            y: torch.Tensor,
            optimizer: torch.nn.Module = None,
            gradient_clip_value: float = 0.,
    ) -> tuple[torch.Tensor, float, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_a = torch.zeros_like(hidden_z)
        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        loss_val = torch.zeros(1).to(x.device)
        total_loss = 0
        total_loss_ll = 0

        for t in range(sequence_length):

            input_t = x[t]
            target_t = y[t]

            hidden = hidden_z, hidden_u, hidden_v, hidden_a

            hidden_z, hidden_u, hidden_v, hidden_a = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),  # input_t
                hidden
            )

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

                    if gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_value)

                    optimizer.step()

                    # reset loss
                    loss_val = torch.zeros_like(loss)

                    # detach hidden and output states
                    hidden_z.detach_()
                    hidden_u.detach_()
                    hidden_v.detach_()
                    hidden_a.detach_()

                    out_u.detach_()

                    # reset optimizer
                    optimizer.zero_grad()

                # compute last truncation chunk
                if t == (sequence_length - 1) and self.training and loss_val.item() != 0:

                    if self.label_last:
                        loss_val = loss
                        # for printing
                        total_loss_ll += loss.item()

                    loss_val.backward()
                    if gradient_clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_value)
                    optimizer.step()
                    optimizer.zero_grad()

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1, :, :].unsqueeze(dim=0)
            total_loss = total_loss_ll

        return outputs, total_loss, ((hidden_z, hidden_u), out_u)


# specifically for spike deletion
class BRFRSNN_SD(SimpleResRNN):
    def __init__(
            self,
            *args,
            spike_del_p: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spike_del_p = spike_del_p

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], int]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = 0

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_q = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v, hidden_q

            hidden_z, hidden_u, hidden_v, hidden_q = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            hidden_z = functional.spike_deletion(hidden_z, self.spike_del_p)

            # SOP
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)


        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes


class BRFRSNN_BP(SimpleResRNN):
    def __init__(
            self,
            *args,
            bit_precision: int = 52,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.bit_precision = bit_precision

        self.out = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], int]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = 0

        x = functional.quantize_tensor(x, self.bit_precision)

        hidden_z = functional.quantize_tensor(
            torch.zeros((batch_size, self.hidden_size)).to(x.device),
            f=self.bit_precision
        )

        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)
        hidden_q = torch.zeros_like(hidden_z)

        out_u = functional.quantize_tensor(
            torch.zeros((batch_size, self.output_size)).to(x.device),
            f=self.bit_precision
        )

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v, hidden_q

            hidden_z, hidden_u, hidden_v, hidden_q = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes


# specifically for spike deletion
class RFRSNN_SD(SimpleVanillaRFRNN):
    def __init__(
            self,
            *args,
            spike_del_p: float = 0.,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.spike_del_p = spike_del_p

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], int]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = 0

        hidden_z = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)

        out_u = torch.zeros((batch_size, self.output_size)).to(x.device)

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v

            hidden_z, hidden_u, hidden_v = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            hidden_z = functional.spike_deletion(hidden_z, self.spike_del_p)

            # SOP
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes


# Flexible bit precision for Vanilla RF
class RFRSNN_BP(SimpleVanillaRFRNN):

    def __init__(
            self,
            *args,
            bit_precision: int = 32,
            **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.bit_precision = bit_precision

        self.out = modules.LICellBP(
            input_size=self.hidden_size,
            layer_size=self.output_size,
            bit_precision=bit_precision,
            adaptive_tau_mem_mean=self.out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=self.out_adaptive_tau_mem_std
        )

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], int]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs = list()

        num_spikes = 0

        x = functional.quantize_tensor(x, self.bit_precision)
        hidden_z = functional.quantize_tensor(torch.zeros((batch_size, self.hidden_size)).to(x.device), f=self.bit_precision)
        hidden_u = torch.zeros_like(hidden_z)
        hidden_v = torch.zeros_like(hidden_z)

        out_u = functional.quantize_tensor(
            torch.zeros((batch_size, self.output_size)).to(x.device),
            f=self.bit_precision
        )

        for t in range(sequence_length):

            input_t = x[t]

            hidden = hidden_z, hidden_u, hidden_v

            hidden_z, hidden_u, hidden_v = self.hidden(
                torch.cat((input_t, hidden_z), dim=1),
                hidden
            )

            # SOP
            num_spikes += hidden_z.sum().item()

            out_u = self.out(hidden_z, out_u)

            # Records outputs with sub_seq_length delay
            if t >= self.sub_seq_length:
                outputs.append(out_u)

        outputs = torch.stack(outputs)

        if self.label_last:
            outputs = outputs[-1:, :, :]

        return outputs, ((hidden_z, hidden_u), out_u), num_spikes
