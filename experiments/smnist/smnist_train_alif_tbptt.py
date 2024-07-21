import torch.nn
import torchvision
from torch.utils.data import DataLoader, random_split
import tools
from datetime import datetime
import math

import sys
sys.path.append("../..")
import snn
import random
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

################################################################
# General settings
################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if device == "cuda":
    pin_memory = True
    num_workers = 1
else:
    pin_memory = False
    num_workers = 0

print(device)

################################################################
# Data loading and preparation, logging
################################################################

PERMUTED = False
label_last = True

sequence_length = 28 * 28
input_size = 1
num_classes = 10
batch_size = 256  # (256 from Yin et al. 2021)

# validation and test batch size can be chosen higher
# (depending on VRAM capacity)
val_batch_size = 256
test_batch_size = 256

train_dataset = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

total_dataset_size = len(train_dataset)

# we use 5% - 10% of the training data for validation
val_dataset_size = int(total_dataset_size * 0.1)
train_dataset_size = total_dataset_size - val_dataset_size

train_dataset, val_dataset = random_split(
    train_dataset, [train_dataset_size, val_dataset_size]
)

test_dataset = torchvision.datasets.MNIST(
    root="data",
    train=False,
    transform=torchvision.transforms.ToTensor()
)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False
)

test_dataset_size = len(test_dataset)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False
)


def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor
):
    tensor = tensor.to(device=device).view(batch_size_, sequence_length_, input_size_)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor[permuted_idx_, :, :]
    return tensor

################################################################
# Model helpers and model setup
################################################################

hidden_size = 256

# recorded into comment
# fraction of the elements in the hidden.linear.weight to be zero
mask_prob = 0.0

# ALIF alpha tau_mem init normal dist.
adaptive_tau_mem_mean = 20.
adaptive_tau_mem_std = 5.

# ALIF rho tau_adp init normal dist.
adaptive_tau_adp_mean = 200.
adaptive_tau_adp_std = 50.

# LI alpha tau_mem init normal distribution
out_adaptive_tau_mem_mean = 20.
out_adaptive_tau_mem_std = 5.

hidden_bias = True
output_bias = True

tbptt_steps = 50

criterion = torch.nn.NLLLoss()

model = snn.models.SimpleALIFRNNTbptt(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    mask_prob=mask_prob,
    criterion=criterion,
    adaptive_tau_mem_mean=adaptive_tau_mem_mean,
    adaptive_tau_mem_std=adaptive_tau_mem_std,
    adaptive_tau_adp_mean=adaptive_tau_adp_mean,
    adaptive_tau_adp_std=adaptive_tau_adp_std,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    label_last=label_last,
    hidden_bias=hidden_bias,
    output_bias=output_bias,
    tbptt_steps=tbptt_steps
).to(device)

################################################################
# Setup experiment (optimizer etc.)
################################################################

optimizer_lr = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

# Number of iterations per epoch
total_steps = len(train_loader)
epochs_num = 300
padding = 0

# learning rate scheduling
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs_num)
learning_rates = []

rand_num = random.randint(1, 10000)

# [logging] Only thing manually changed in the string: Optimizer, criterion and scheduler!
opt_str = "{}_Adam({}),PERMUTED({}),LinearLR,NLL,LL({}),TBPTT({})".format(rand_num, optimizer_lr, PERMUTED, label_last, tbptt_steps)
net_str = "RSNN(1,256,10,bs_{},ep_{},h_o_bias)"\
    .format(batch_size, epochs_num)
unit_str = "ALIF(tau_m({},{}),tau_a({},{}),linearMask({}))LI(tau_m({},{}))"\
    .format(adaptive_tau_mem_mean, adaptive_tau_mem_std, adaptive_tau_adp_mean, adaptive_tau_adp_std, mask_prob,
            out_adaptive_tau_mem_mean, out_adaptive_tau_mem_std)

comment = opt_str + "," + net_str + "," + unit_str


writer = SummaryWriter(comment=comment)
start_time = datetime.now().strftime("%m-%d_%H-%M-%S")


# PSMNIST fixed random permutation
if PERMUTED:
    permuted_idx = torch.randperm(sequence_length)
    torch.save(permuted_idx, './models/{}'.format(start_time) + comment + '_permuted_idx.pt')
    print(permuted_idx)
else:
    permuted_idx = torch.arange(sequence_length)

print(start_time, comment)

save_path = "models/{}_".format(start_time) + opt_str + "," + net_str + "," + unit_str + ".pt"
save_init_path = "models/{}_init_".format(start_time) + opt_str + "," + net_str + "," + unit_str + ".pt"

# save initial parameters for analysis
torch.save({'model_state_dict': model.state_dict()}, save_init_path)

# print(model.state_dict())

print_every = 150

################################################################
# Training loop
################################################################

iteration = 0
min_val_loss = float("inf")
loss_value = 1.
end_training = False

run_time = tools.PerformanceCounter()
tools.PerformanceCounter.reset(run_time)

for epoch in range(epochs_num + 1):

    # Go into eval mode
    model.eval()

    with torch.no_grad():

        val_loss = 0
        val_correct = 0

        # Perform validation
        for i, (inputs, targets) in enumerate(val_loader):

            current_batch_size = len(inputs)

            # Reshape inputs in [sequence_length, batch_size, data_size].
            inputs = smnist_transform_input_batch(
                tensor=inputs.to(device=device),
                sequence_length_=sequence_length,
                batch_size_=current_batch_size,
                input_size_=input_size,
                permuted_idx_=permuted_idx
            )

            # Reshape targets (for MNIST it's a single pattern).
            targets = targets.to(device=device)

            outputs, loss, _ = model(inputs, targets.repeat((sequence_length, 1)), optimizer=None)

            # for Label Last
            if label_last:
                loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=targets)
                val_loss_value = loss.item()
            else:
                val_loss_value = loss / sequence_length

            val_loss += val_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), targets)
            val_correct += batch_correct

        val_loss /= len(val_loader)  # val_dataset_size
        val_accuracy = (val_correct / val_dataset_size) * 100.0

        # Log current val loss and accuracy
        writer.add_scalar(
            "Loss/val",
            val_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/val",
            val_accuracy,
            epoch
        )

        # Persist current best model.
        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            best_model_state_dict = model.state_dict()
            # TODO save checkpoint of the training including model.state_dict() and optimizer.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_value,
            }, save_path)

        test_loss = 0
        test_correct = 0

        # Perform Inference
        for i, (inputs, targets) in enumerate(test_loader):
            # Reshape inputs in [sequence_length, batch_size, data_size].
            current_batch_size = len(inputs)

            inputs = smnist_transform_input_batch(
                tensor=inputs.to(device=device),
                sequence_length_=sequence_length,
                batch_size_=current_batch_size,
                input_size_=input_size,
                permuted_idx_=permuted_idx
            )

            # Reshape targets (for MNIST it's a single pattern).
            targets = targets.to(device=device)

            outputs, loss, _ = model(inputs, targets.repeat((sequence_length, 1)), optimizer=None)

            # for Label Last
            if label_last:
                loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=targets)
                test_loss_value = loss.item()
            else:
                test_loss_value = loss / sequence_length

            test_loss += test_loss_value

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), targets)
            test_correct += batch_correct

        test_loss /= len(test_loader)  # test_dataset_size
        test_accuracy = (test_correct / test_dataset_size) * 100.0

        # Log current test loss and accuracy
        writer.add_scalar(
            "Loss/test",
            test_loss,
            epoch
        )
        writer.add_scalar(
            "Accuracy/test",
            test_accuracy,
            epoch
        )

        # Update logging outputs
        writer.flush()

        print(
            "Epoch [{:4d}/{:4d}]  |  Summary  |  Loss/val: {:.6f}, Accuracy/val: {:.4f}%  |  Loss/test: {:.6f}, "
            "Accuracy/test: {:.4f}".format(
                epoch, epochs_num, val_loss, val_accuracy, test_loss, test_accuracy), flush=True
        )


    if epoch < epochs_num:
        # Go into train mode.
        model.train()

        train_correct = 0
        print_train_loss = 0
        print_correct = 0
        print_total = 0

        # Perform training epoch (iterate over all mini batches in training set).
        for i, (inputs, targets) in enumerate(train_loader):

            current_batch_size = len(inputs)

            # Reshape inputs in [sequence_length, batch_size, data_size].
            inputs = smnist_transform_input_batch(
                tensor=inputs.to(device=device),
                sequence_length_=sequence_length,
                batch_size_=current_batch_size,
                input_size_=input_size,
                permuted_idx_=permuted_idx
            )

            # Reshape targets (for MNIST it's a single pattern).
            targets = targets.to(device=device)

            # Clear previous gradients
            optimizer.zero_grad()

            outputs, loss, _ = model(inputs, targets.repeat((sequence_length, 1)), optimizer)

            # for Label Last
            if label_last:
                loss_value = loss / math.ceil(sequence_length / tbptt_steps)
            else:
                loss_value = loss / sequence_length

            if math.isnan(loss_value):
                end_training = True
                break

            # Calculate batch accuracy
            batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), targets)

            # Log current loss and accuracy
            writer.add_scalar(
                "Loss/train",
                loss_value,
                iteration
            )
            writer.add_scalar(
                "Accuracy/train",
                (batch_correct / current_batch_size) * 100.0,
                iteration
            )

            print_train_loss += loss_value
            print_total += current_batch_size
            print_correct += batch_correct

            # Print current training loss/acc at every 50th iteration
            if i % print_every == (print_every - 1):

                print_acc = (print_correct / print_total) * 100.0

                print("Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f}, Accuracy/train: {:8.4f}".format(
                    epoch + 1, epochs_num, i + 1, total_steps, print_train_loss / print_every, print_acc), flush=True
                )

                print_correct = 0
                print_total = 0
                print_train_loss = 0

            iteration += 1

        scheduler.step()

    if end_training:
        break

print("Minimum val loss: {:.6f} at epoch: {}".format(min_val_loss, min_val_epoch))
print(tools.PerformanceCounter.time(run_time) / 3600, "hr")
