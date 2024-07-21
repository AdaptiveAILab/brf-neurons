import torch
from torch.utils.data import TensorDataset, DataLoader
import typing
import time


def convert_data_format(data: dict) -> TensorDataset:

    # input train: (618, 1301, 4); input test: (141, 1301, 4)
    inputs = data['x']
    inputs = torch.from_numpy(inputs).to(torch.float)

    # remove -1 from last sequence step -> (618, 1300, 4) & (141, 1300, 4)
    inputs_last_step_removed = inputs[:, :-1, :]

    # target train: (618, 1301, 6); target test: (141, 1301, 6)
    # one hot vector for each time step
    targets = data['y']
    targets = torch.from_numpy(targets).to(torch.float)

    # remove -1 from last sequence step -> (618, 1300, 6) & (141, 1300, 6)
    targets_last_step_removed = targets[:, :-1, :]

    dataset = TensorDataset(*[inputs_last_step_removed, targets_last_step_removed])

    return dataset


def apply_seq_loss(
        criterion: torch.nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        scale_func: typing.Callable[[int], float] = None
) -> torch.Tensor:

    # shape: (sequence_length, batch_size, num_classes)
    sequence_length = outputs.shape[0]

    # get argmax from one-hot vector: target value 0 to 5
    targets_argmax = targets.argmax(dim=2)

    # criterion=NLLLoss: so outputs must be soft-maxed
    log_softmax = torch.nn.LogSoftmax(dim=2)
    out = log_softmax(outputs)

    loss = 0

    if scale_func is None:

        for t in range(sequence_length):

            loss = loss + criterion(out[t], targets_argmax[t])

    else:

        for t in range(sequence_length):

            loss = loss + scale_func(t) + criterion(out[t], targets_argmax[t])

    return loss


def count_correct_prediction(
        predictions: torch.Tensor,
        targets: torch.Tensor
) -> int:
    # elementwise comparison of num_classes; correct predictions summed up for each sequence length and batch
    return predictions.argmax(dim=2).eq(targets.argmax(dim=2)).sum().item()



class PerformanceCounter:

    def __init__(self) -> None:
        self.start_time = 0

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def time(self) -> float:
        current_time = time.perf_counter()
        return current_time - self.start_time
