import typing
import torch
import time
from torch.utils.data import DataLoader

class NoisyDataLoader(DataLoader):
    def __init__(self, *args,
                 noise_percent=0.1,
                 noise_mean=0,
                 noise_std=1,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.noise_percent = noise_percent
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __iter__(self):
        for batch in super().__iter__():
            batch_data, batch_targets = batch
            batch_size = batch_data.shape[0]
            dim_1 = batch_data.shape[1]
            dim_2 = batch_data.shape[2]

            overlay = torch.rand(batch_size)
            # noise for whole images within percentage of each batch
            # e.g. bs 256 with 25% noise -> 64 images with noise
            noise = (overlay < self.noise_percent).\
                        view(batch_size, 1, 1, 1).\
                        expand(batch_size, dim_1, dim_2, dim_2) * \
                    self.noise_std * torch.randn_like(batch_data)

            batch_data_noisy = batch_data + noise

            yield batch_data_noisy, batch_targets

def apply_seq_loss(
        criterion: torch.nn.Module,
        outputs: torch.Tensor,
        target: torch.Tensor,
        scale_func: typing.Callable[[int], float] = None
) -> torch.Tensor:

    sequence_length = outputs.shape[0]
    # batch_size = outputs.shape[1]
    # num_layers = outputs.shape[2]

    loss = 0

    if isinstance(criterion, torch.nn.NLLLoss):
        log_softmax = torch.nn.LogSoftmax(dim=2)
        out = log_softmax(outputs)
    else:
        out = outputs

    if scale_func is None:
        for t in range(sequence_length):
            loss = loss + criterion(out[t], target)
    else:
        for t in range(sequence_length):
            loss = loss + scale_func(t) + criterion(out[t], target)

    return loss


def count_correct_predictions(predictions: torch.Tensor, labels: torch.Tensor) -> int:
    return predictions.argmax(dim=1).eq(labels).sum().item()


def custom_print(message_to_print, log_file: str):
    print(message_to_print)
    with open(log_file, 'a') as of:
        of.write(message_to_print + '\n')


class PerformanceCounter:

    def __init__(self) -> None:
        self.start_time = 0

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def time(self) -> float:
        current_time = time.perf_counter()
        return current_time - self.start_time


# input and hidden weights combined in this network, if not combined, input=0
def filterwise_norm(input: int, param: torch.Tensor) -> torch.Tensor:
    d = torch.randn_like(param)

    d_in = d[:, :input]
    d_in = (d_in / torch.norm(d_in, dim=1, keepdim=True)) * torch.norm(param[:, :input], dim=1, keepdim=True)

    d_hidden = d[:, input:]
    d_hidden = (d_hidden / torch.norm(d_hidden, dim=1, keepdim=True)) * torch.norm(param[:, input:], dim=1,
                                                                                   keepdim=True)

    d = torch.concat((d_in, d_hidden), dim=1)

    return d