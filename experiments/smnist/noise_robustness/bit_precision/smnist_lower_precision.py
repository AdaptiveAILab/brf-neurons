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
label_last = True
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

###################################################################################
# Helper functions
###################################################################################


def smnist_transform_input_batch(
        tensor: torch.Tensor,
        sequence_length_: int,
        batch_size_: int,
        input_size_: int,
        permuted_idx_: torch.Tensor,
):
    tensor = tensor.to(device=device).view(batch_size_, sequence_length_, input_size_)
    tensor = tensor.permute(1, 0, 2)
    tensor = tensor[permuted_idx_, :, :]
    return tensor


def saved_models(neuron_: str,
                 root_=root
                 ) -> tuple[list[str], str]:

    if 'rf' in neuron_:

        # brf smnist
        # comment_ = 'Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300),' \
        #           'RF(abs(omega_uni(15.0,50.0)),sust_osc,abs(b_offset(uni(0.1,1.0))-q,theta(1),linearMask(0.0))' \
        #           'LI(norm_20.0,5.0)'

        # vrf
        comment_ = 'Adam(0.1),NLL,LinearLR,LabelLast(True),PERMUTED(False),RFSNN(1,256,10,bs=256,ep=300),' \
                   'RF(abs(omega_uni(15.0,50.0)),no_sust_osc,-abs(b(uni(0.1,1.0)),linearMask(0.0))LI(norm_20.0,5.0)'
    else:

        # alif smnist
        comment_ = 'Adam(0.001),PERMUTED(False),LinearLR,NLL,LabelLast(True),TBPTT(50),RSNN(1,256,10,bs_256,ep_300,' \
                  'no_bias),ALIF(tau_m(20.0,5.0),tau_a(200.0,50.0),linearMask(0.0))LI(tau_m(20.0,5.0))'

    models_str_ = [f for f in os.listdir(root_) if comment_ in f]

    # take out initial model and permuted idx
    models_str_ = [f for f in models_str_ if '_init_' not in f]
    models_str_ = [f for f in models_str_ if 'permuted_' not in f]
    print(len(models_str_))

    return models_str_, comment_


def model_init(neuron_: str,
               label_last_: bool,
               bit: int,
               device_=device):

    if 'rf' in neuron_:

        model = snn.models.RFRSNN_BP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            adaptive_omega_a=15,
            adaptive_omega_b=50,
            adaptive_b_offset_a=0.1,
            adaptive_b_offset_b=1,
            out_adaptive_tau_mem_mean=20,
            out_adaptive_tau_mem_std=5,
            label_last=label_last_,
            output_bias=False,
            bit_precision=bit
        ).to(device_)

    else:

        model = snn.models.ALIFRSNN_BP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            adaptive_tau_mem_mean=20.,
            adaptive_tau_mem_std=5.,
            adaptive_tau_adp_mean=200.,
            adaptive_tau_adp_std=50.,
            out_adaptive_tau_mem_mean=20.,
            out_adaptive_tau_mem_std=50.,
            label_last=label_last_,
            hidden_bias=False,
            output_bias=False,
            bit_precision=bit
        ).to(device_)

    return model


def quantize_model(model_, f):
    # Apply quantization to all parameters and buffers of the model
    for param in model_.parameters():
        param.data = snn.functional.quantize_tensor(param.data, f)

    for name, buffer in model_.named_buffers():
        if buffer is not None:
            buffer.data = snn.functional.quantize_tensor(buffer.data, f)

    return model_

########################################################################################
# Test bit precision
########################################################################################


bits = [1, 2, 4, 8, 16, 32]
print(bits)

models_str, comment = saved_models(neuron_=neuron)

losses = np.zeros((len(bits), len(models_str)))
accs = np.zeros((len(bits), len(models_str)))
sop = np.zeros((len(bits), len(models_str)))

inputs, targets = next(iter(test_loader))

current_batch_size = len(inputs)

# Reshape targets (for MNIST it's a single pattern).
target = targets.to(device=device)

for j in range(len(models_str)):

    permuted_idx = torch.arange(sequence_length)

    for i in range(len(bits)):

        model = model_init(neuron_=neuron,
                           label_last_=label_last,
                           bit=bits[i])

        # changed to each model
        post_training_dict = torch.load(root + models_str[j], map_location=device)
        model.load_state_dict(post_training_dict['model_state_dict'])

        # quantize model
        model_lp = quantize_model(model, bits[i])

        input = smnist_transform_input_batch(
            tensor=inputs.to(device=device),
            sequence_length_=sequence_length,
            batch_size_=current_batch_size,
            input_size_=input_size,
            permuted_idx_=permuted_idx,
        )

        model_lp.eval()

        with torch.no_grad():

            outputs, _, num_spikes = model_lp(input)

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

            losses[i][j] = test_loss_value
            accs[i][j] = test_accuracy
            sop[i][j] = num_spikes / test_dataset_size

print('losses', losses)
print('accs', accs)
print('sop', sop)

# np.save('./smnist_bit_precision_strict_' + neuron, (losses, accs, sop))
