import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from experiments.smnist.tools import *
import sys
import os

sys.path.append('../../../..')
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
neuron = 'vanilla_rf'  #'alif' # rf

sequence_length = 28 * 28
input_size = 1
num_classes = 10
test_batch_size = 10000
hidden_size = 256

root = './../../models/'
criterion = torch.nn.NLLLoss()
test_dataset = torchvision.datasets.MNIST(
    root="./../../data",
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
        permuted_idx_: torch.Tensor,
        std_dev: float,
):
    tensor = tensor.to(device=device).view(batch_size_, sequence_length_, input_size_)
    # add gaussian noise into input
    tensor = tensor + std_dev*torch.randn_like(tensor)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor[permuted_idx_, :, :]
    return tensor


if 'rf' in neuron:

    label_last = True
    # brf smnist -> switch to SimpleResRNN
    # comment = 'Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300),' \
    #               'RF(abs(omega_uni(15.0,50.0)),sust_osc,abs(b_offset(uni(0.1,1.0))-q,theta(1),linearMask(0.0))' \
    #               'LI(norm_20.0,5.0)'
    # vrf
    comment = 'Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300),' \
              'RF(abs(omega_uni(15.0,50.0)),no_sust_osc,-abs(b(uni(0.1,1.0)),linearMask(0.0))LI(norm_20.0,5.0)'

    model = snn.models.SimpleVanillaRFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        adaptive_omega_a=15,
        adaptive_omega_b=50,
        adaptive_b_offset_a=0.1,
        adaptive_b_offset_b=1,
        out_adaptive_tau_mem_mean=20,
        out_adaptive_tau_mem_std=5,
        label_last=label_last,
        # mask_prob=0.,
        output_bias=False,
    ).to(device)

else:

    label_last = True
    # alif smnist
    comment = 'Adam(0.001),PERMUTED(False),LinearLR,NLL,LabelLast(True),TBPTT(50),RSNN(1,256,10,bs_256,ep_300,' \
              'no_bias),ALIF(tau_m(20.0,5.0),tau_a(200.0,50.0),linearMask(0.0))LI(tau_m(20.0,5.0))'

    model = snn.models.SimpleALIFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        adaptive_tau_mem_mean=20.,
        adaptive_tau_mem_std=5.,
        adaptive_tau_adp_mean=200.,
        adaptive_tau_adp_std=50.,
        out_adaptive_tau_mem_mean=20.,
        out_adaptive_tau_mem_std=50.,
        label_last=label_last,
        hidden_bias=False,
        output_bias=False,
    ).to(device)

models_str = [f for f in os.listdir(root) if comment in f]

# take out initial model and permuted idx
models_str = [f for f in models_str if '_init_' not in f]
models_str = [f for f in models_str if 'permuted_' not in f]
print(len(models_str))

std_dev = np.arange(0, 0.4, 0.05)
repeats = 5
losses = np.zeros((len(std_dev), len(models_str), repeats))
accs = np.zeros((len(std_dev), len(models_str), repeats))
sop = np.zeros((len(std_dev), len(models_str), repeats))

inputs, targets = next(iter(test_loader))

current_batch_size = len(inputs)

# Reshape targets (for MNIST it's a single pattern).
target = targets.to(device=device)

for j in range(len(models_str)):
    print(j)

    # changed to each model
    post_training_dict = torch.load(root + models_str[j], map_location=device)
    model.load_state_dict(post_training_dict['model_state_dict'])

    permuted_idx = torch.arange(sequence_length)

    for i in range(len(std_dev)):

        for h in range(repeats):

            input = smnist_transform_input_batch(
                tensor=inputs.to(device=device),
                sequence_length_=sequence_length,
                batch_size_=current_batch_size,
                input_size_=input_size,
                permuted_idx_=permuted_idx,
                std_dev=std_dev[i]
            )

            model.eval()

            with torch.no_grad():

                outputs, _, num_spikes = model(input)

                # Apply loss sequentially against single pattern.
                loss = apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

                # save loss
                if label_last:
                    test_loss_value = loss.item()
                else:
                    test_loss_value = loss.item() / sequence_length

                # Calculate accuracy
                test_correct = count_correct_predictions(outputs.mean(dim=0), target)
                test_accuracy = (test_correct / test_dataset_size) * 100.0

                losses[i][j][h] = test_loss_value
                accs[i][j][h] = test_accuracy
                sop[i][j][h] = num_spikes / test_dataset_size

print('losses', losses)
print('accs', accs)
print('sop', sop)

np.save('./smnist_noise_input_5reps_{}'.format(neuron), (losses, accs, sop))
