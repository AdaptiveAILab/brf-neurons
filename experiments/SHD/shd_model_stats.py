import torch.nn
import tools
from torch.utils.data import DataLoader
import os
import sys
sys.path.append("../..")
import snn

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
# Dataset
################################################################

# TEST DATASET #
test_dataset = tools.shd_to_dataset('./data/testX_4ms.npy', './data/testY_4ms.npy')

# 2264 sequences in test dataset
test_dataset_size = len(test_dataset)

####################################################################
# DataLoader
####################################################################

sequence_length = 250
input_size = 700
hidden_size = 128
num_classes = 20
label_last = False

# Change manually for different models
neuron = "vrf"  # "vrf", "brf", "alif"

# validation and test batch size can be chosen higher
# (depending on VRAM capacity)
test_batch_size = 2264

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=test_batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False
)

################################################################
# Model setup
################################################################

if "vrf" in neuron:

    sub_seq_length = 0
    # vrf (without reset) SHD: test acc. of saved model 90.15 %
    comment = "Adam(0.075),NLL,LinLR,LL(False,no_gc),700,128,20,bs_32,ep_20,VRF(omega5.0,10.0b2.0,3.0,Nr)LI(20.0,5.0)"

    model = snn.models.SimpleVanillaRFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        sub_seq_length=sub_seq_length,
        pruning=True,  # setting of the model, whether pruning tensor saved in the model (even if no pruning)
    ).to(device)

elif "brf" in neuron:

    sub_seq_length = 0

    # brf SHD: test acc. of saved model 92.27 %
    comment = 'Adam(0.075),NLL,LinearLR,LabelLast(False,no_gc),RFSNN(700,128,20,sub_seq_length_0,bs_32,ep_20),' \
              'RF(abs_omega_uni_5.0_10.0,abs_b_offset_uni_2.0,3.0-q,linearMask(0.0))LI(norm(20.0,5.0))'

    model = snn.models.SimpleResRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        sub_seq_length=sub_seq_length,
        pruning=True
    ).to(device)

else:

    sub_seq_length = 10

    # alif SHD: test acc. of saved model 73.63 %
    comment = 'Adam(0.075),NLL,LinearLR,LabelLast(False),no_gc,RSNN(700,128,20,sub_seq(10),bs_32,ep_20_no_bias),' \
              'ALIF(tau_m(20.0,5.0),tau_a(150.0,10.0),linearMask_0.0)LI(tau_m(20.0,5.0))'

    model = snn.models.SimpleALIFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        sub_seq_length=sub_seq_length,
        pruning=True
    ).to(device)


criterion = torch.nn.NLLLoss()

path = './models/'

models_str = [f for f in os.listdir(path) if comment in f]

# take out initial model and permuted idx
models_str = [f for f in models_str if '_init_' not in f]
models_str = [f for f in models_str if 'permuted_' not in f]

print(models_str)
PATH = "./models/" + models_str[0]
checkpoint = torch.load(PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Go into eval mode
model.eval()

with torch.no_grad():

    test_loss = 0
    test_correct = 0
    total_spikes = 0

    # Perform Inference
    inputs, targets = next(iter(test_loader))

    # Reshape inputs in [sequence_length, batch_size, data_size].
    inputs = inputs.permute(1, 0, 2).to(device)

    target = targets.to(device=device)

    outputs, _, num_spikes = model(inputs)

# accumulate total spikes
total_spikes += num_spikes

# Apply loss sequentially against single pattern.
loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, target=target)

# for Label Last
if label_last:
    test_loss_value = loss.item()
else:
    test_loss_value = loss.item() / (sequence_length - sub_seq_length)

test_loss += test_loss_value

# Calculate batch accuracy
batch_correct = tools.count_correct_predictions(outputs.mean(dim=0), target)
test_correct += batch_correct

test_loss /= len(test_loader)
test_accuracy = (test_correct / test_dataset_size) * 100.0

# total average SOP (spike operations per sample)
SOP = total_spikes / test_dataset_size

# total average SOP per sequence step
SOP_per_step = SOP / sequence_length

# firing rate per neuron
firing_rate = total_spikes / (test_dataset_size * sequence_length * hidden_size)

print(
    'Test loss: {:.6f}, Test acc: {:.4f}, SOP: {:.2f}, SOP per step: {:.2f}, mean firing rate per neuron: {:.2f}'
    .format(test_loss, test_accuracy, SOP, SOP_per_step, firing_rate))

