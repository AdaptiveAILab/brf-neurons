import typing
import torch
from numpy import load
from torch.utils.data import TensorDataset, DataLoader
import time

def shd_to_dataset(
        input_file_path: str,
        label_file_path: str,
) -> TensorDataset:

    inputs = load(input_file_path)
    inputs = torch.Tensor(inputs)

    targets = load(label_file_path).astype(float)
    targets = torch.Tensor(targets).long()

    return TensorDataset(inputs, targets)

def apply_seq_loss(
        criterion: torch.nn.Module,
        outputs: torch.Tensor,
        target: torch.Tensor,
        scale_func: typing.Callable[[int], float] = None
) -> torch.Tensor:

    sequence_length = outputs.shape[0]

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


class PerformanceCounter:

    def __init__(self) -> None:
        self.start_time = 0

    def reset(self) -> None:
        self.start_time = time.perf_counter()

    def time(self) -> float:
        current_time = time.perf_counter()
        return current_time - self.start_time
