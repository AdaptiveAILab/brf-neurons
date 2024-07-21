import math
from experiments.smnist.tools import *
import torchvision
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, StepLR, ConstantLR, SequentialLR

import sys
sys.path.append('../../..')
import snn

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
test_batch_size = 10000
hidden_size = 256

root = './../models/'
criterion = torch.nn.NLLLoss()
test_dataset = torchvision.datasets.MNIST(
    root="./../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_dataset_size = len(test_dataset)

test_loader = torch.utils.data.DataLoader(
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


# omega init uniform distribution CHECK!
omega_a = 15.
omega_b = 50.

# b_offset init uniform distribution CHECK!
b_offset_a = 0.1
b_offset_b = 1.

# LI alpha init normal distribution
out_adaptive_tau_mem_mean = 20.0
out_adaptive_tau_mem_std = 5.0

neuron ='rf'  #'alif_tbptt' # brf
points = 50
bound = 1

model = snn.models.SimpleResRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=num_classes,
    adaptive_omega_a=omega_a,
    adaptive_omega_b=omega_b,
    adaptive_b_offset_a=b_offset_a,
    adaptive_b_offset_b=b_offset_b,
    out_adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
    out_adaptive_tau_mem_std=out_adaptive_tau_mem_std,
    label_last=label_last,
    mask_prob=0.,
    output_bias=False,
    pruning=True
).to(device)

# model = snn.models.SimpleALIFRNNTbptt(
#     input_size=input_size,
#     hidden_size=hidden_size,
#     output_size=num_classes,
#     mask_prob=0,
#     criterion=criterion,
#     adaptive_tau_mem_mean=20.,
#     adaptive_tau_mem_std=5.,
#     adaptive_tau_adp_mean=200.,
#     adaptive_tau_adp_std=50.,
#     out_adaptive_tau_mem_mean=20.,
#     out_adaptive_tau_mem_std=50.,
#     label_last=label_last,
#     hidden_bias=False,
#     output_bias=False,
#     tbptt_steps=50,
#     pruning=True,
# ).to(device)


permuted_idx = torch.arange(sequence_length)

#############################################################################################
# Optimization
#############################################################################################

# Trained until val acc of 50% reached.
train_model = False

def opt_model(optimizer_lr: float):

    batch_size = 256
    val_batch_size = 256

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

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_dataset_size, val_dataset_size]
    )


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )


    gradient_clip_value = 1.

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)

    # Number of iterations per epoch
    total_steps = len(train_loader)
    epochs_num = 300

    # learning rate scheduling
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / (epochs_num))

    comment = 'smnist_best_{}_model'.format(neuron)

    writer = SummaryWriter(comment=comment)
    start_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    save_path = root + "{}_".format(start_time) + comment + ".pt"

    iteration = 0
    loss_value = 1.
    end_training = False
    print_every = 150

    for epoch in range(epochs_num + 1):

        # Go into train mode.
        model.train()

        print_train_loss = 0

        # Perform training epoch (iterate over all mini batches in training set).
        for i, (inputs, targets) in enumerate(train_loader):

            current_batch_size = len(inputs)

            # Reshape inputs in [sequence_length, batch_size, data_size].
            input = smnist_transform_input_batch(
                tensor=inputs.to(device=device),
                sequence_length_=sequence_length,
                batch_size_=current_batch_size,
                input_size_=input_size,
                permuted_idx_=permuted_idx
            )

            # Reshape targets (for MNIST it's a single pattern).
            target = targets.to(device=device)

            # Clear previous gradients
            optimizer.zero_grad()

            outputs, _, _ = model(input)

            # Apply loss sequentially against single pattern.
            loss = apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            #for Label Last
            if label_last:
                loss_value = loss.item()
            else:
                loss_value = loss.item() / sequence_length

            # calculate gradient
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

            # Perform learning step
            optimizer.step()

            if math.isnan(loss_value):
                end_training = True
                break

            # Log current loss and accuracy
            writer.add_scalar(
                "Loss/train",
                loss_value,
                iteration
            )

            print_train_loss += loss_value

            # Print current training loss/acc at every 50th iteration
            if i % print_every == (print_every - 1):

                print("Epoch [{:4d}/{:4d}]  |  Step [{:4d}/{:4d}]  |  Loss/train: {:.6f}".format(
                    epoch + 1, epochs_num, i + 1, total_steps, print_train_loss / print_every), flush=True
                )

                print_train_loss = 0

            iteration += 1

        scheduler.step()

        # Go into eval mode
        model.eval()

        with torch.no_grad():

            val_loss = 0
            val_correct = 0

            # Perform validation
            for i, (inputs, targets) in enumerate(val_loader):

                current_batch_size = len(inputs)

                # Reshape inputs in [sequence_length, batch_size, data_size].
                input = smnist_transform_input_batch(
                    tensor=inputs.to(device=device),
                    sequence_length_=sequence_length,
                    batch_size_=current_batch_size,
                    input_size_=input_size,
                    permuted_idx_=permuted_idx
                )

                # Reshape targets (for MNIST it's a single pattern).
                target = targets.to(device=device)

                outputs, _, _ = model(input)
                # outputs, _ = model(input)
                # outputs, _, _ = model(input, target.repeat((sequence_length, 1)), optimizer=None)

                # # Apply loss sequentially against single pattern.
                loss = apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

                # for Label Last
                if label_last:
                    val_loss_value = loss.item()
                else:
                    val_loss_value = loss.item() / sequence_length

                val_loss += val_loss_value

                # Calculate batch accuracy
                batch_correct = count_correct_predictions(outputs.mean(dim=0), target)
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
            if 50.0 <= val_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                }, save_path)
                end_training = True
                break

        # Update logging outputs
        writer.flush()

        if end_training:
            break

########################################################################
# train model
########################################################################

if train_model:
    opt_model(optimizer_lr=0.1)

########################################################################
# loss landscape calculation
########################################################################

alphas = np.linspace(-bound, bound, num=points)
betas = np.linspace(-bound, bound, num=points)

weight = model.hidden.linear.weight
init_weight = model.hidden.linear.weight.clone()
d_delta = filterwise_norm(input_size, weight)
d_eta = filterwise_norm(input_size, weight)

# save loss for each alpha and beta value
losses = np.ones((points, points))

# Single batch with whole test dataset
inputs, targets = next(iter(test_loader))
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
target = targets.to(device=device)

for a, alpha in enumerate(alphas):

    for b, beta in enumerate(betas):

        model.hidden.linear.weight.data = init_weight + alpha * d_delta + beta * d_eta

        model.eval()

        with torch.no_grad():

            test_loss = 0

            outputs, _, _ = model(inputs)

            # Apply loss sequentially against single pattern.
            loss = apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

            # for Label Last
            if label_last:
                test_loss_value = loss.item()
            else:
                test_loss_value = loss.item() / sequence_length

            losses[a][b] = test_loss_value

# np.save('./stats/best_{}_model_{}_{}_{}'.format(neuron, points, bound, test_batch_size), losses)
