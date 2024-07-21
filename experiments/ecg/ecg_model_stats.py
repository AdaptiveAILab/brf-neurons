import torch.nn
import tools
from torch.utils.data import DataLoader
import os
import scipy
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

####################################################################
# Process Dataset
####################################################################

# TEST DATASET #
preprocessed_test_dataset = scipy.io.loadmat('./data/QTDB_test.mat')
test_dataset = tools.convert_data_format(preprocessed_test_dataset)

# 141 sequences in test dataset
test_dataset_size = len(test_dataset)

####################################################################
# DataLoader
####################################################################

sequence_length = 1300
input_size = 4
hidden_size = 36
num_classes = 6
test_batch_size = 141
label_last = False

# Change manually for different models
neuron = "brf"  # "vrf", "brf", "alif"

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

    # vrf (without reset) ecg: test acc. of saved model 85.1 %
    comment = "Adam(0.1),NLL,LinearLR,no_gc,4,36,6,bs=16,ep=400,VRF(omega3.0,5.0,b0.1,1.0,NoR)LI(20.0,1.0)"

    model = snn.models.SimpleVanillaRFRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=num_classes,
        label_last=label_last,
        sub_seq_length=sub_seq_length,
        pruning=False,  # setting of the model, whether pruning tensor saved in the model (even if no pruning)
    ).to(device)

elif "brf" in neuron:

    sub_seq_length = 0

    # brf ecg: test acc. of saved model 86.2 %
    comment = 'Adam(0.1),NLL,LinearLR,no_gc,RFSNN(4,36,6,sub_seq_length=0,bs=16,ep=400),RF(thres(0),' \
              'abs(omega_uni(3.0,5.0)),sust_osc,abs(b_offset(uni(0.1,1.0))-q,theta(1),LinearMask(0.0))LI(20.0,1.0)'

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

    # alif ecg: test acc. of saved model 70.75 %
    comment = 'Adam(0.05),NLL,LinearLR,no_gc,RSNN(4,36,6,sub_seq_10,bs_64,ep_400,h_o_bias(False)),' \
              'ALIF(tau_m(20.0,0.5),tau_a(7.0,0.2),linearMask_0.0)LI(tau_m(20.0,0.5))'

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

    # Perform Inference
    inputs, targets = next(iter(test_loader))

    inputs = inputs.permute(1, 0, 2).to(device)
    target = targets.permute(1, 0, 2).to(device)

    outputs, _, num_spikes = model(inputs)

# compute loss
loss = tools.apply_seq_loss(criterion=criterion, outputs=outputs, targets=target[sub_seq_length:, :, :])
test_loss = loss.item() / (sequence_length - sub_seq_length)
test_loss /= len(test_loader)

# compute accuracy
test_correct = tools.count_correct_prediction(predictions=outputs, targets=target[sub_seq_length:, :, :])
test_acc = (test_correct / (test_dataset_size * (sequence_length - sub_seq_length))) * 100.0

# accumulate total spikes
total_spikes = num_spikes

# total average SOP (spike operations per sample)
SOP = total_spikes / test_dataset_size

# total average SOP per sequence step
SOP_per_step = SOP / sequence_length

# firing rate per neuron
firing_rate = total_spikes / (test_dataset_size * sequence_length * hidden_size)

print(
    'Test loss: {:.6f}, Test acc: {:.4f}, SOP: {:.2f}, SOP per step: {:.2f}, mean firing rate per neuron: {:.2f}'
    .format(test_loss, test_acc, SOP, SOP_per_step, firing_rate))

