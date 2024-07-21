# Balanced Resonate-and-Fire neurons
This is the source repository for the paper [Balanced Resonate-and-Fire Neurons](https://openreview.net/forum?id=dkdilv4XD4) [[Higuchi et al., 2024a]](/)

# Abstract 
The resonate-and-fire (RF) neuron, introduced over two decades ago, is a simple, efficient, yet biologically plausible spiking neuron model, which can extract frequency patterns within the time domain due to its resonating membrane dynamics. However, previous RF formulations suffer from intrinsic shortcomings that limit effective learning and prevent exploiting the principled advantage of RF neurons. Here, we introduce the balanced RF (BRF) neuron, which alleviates some of the intrinsic limitations of vanilla RF neurons and demonstrates its effectiveness within recurrent spiking neural networks (RSNNs) on various sequence learning tasks. We show that networks of BRF neurons achieve overall higher task performance, produce only a fraction of the spikes, and require significantly fewer parameters as compared to modern RSNNs. Moreover, BRF-RSNN consistently provide much faster and more stable training convergence, even when bridging many hundreds of time steps during backpropagation through time (BPTT). These results underscore that our BRF-RSNN is a strong candidate for future large-scale RSNN architectures, further lines of research in SNN methodology, and more efficient hardware implementations.

## Resonate-and-Fire neurons
First introduced by Izhikevich (2001), the dampened oscillatory behavior of biological neurons are simplified as the resonate-and-fire neurons. Each RF neuron has it's own oscillatory eigen-frequency (angular frequncy, omega) and resonates with similar input frequencies. Here is a simulation of how an RF neuron behaves when injected with input frequency similar to its eigen-frequency. 

https://github.com/AdaptiveAILab/brf-neurons/assets/64919377/26ea47a0-997b-4934-b971-5db3483b7eeb

## Balanced RF neurons
Extension of the RF neurons to combat the intrisic shortcoming of RF neurons in SNN training.
- Refractory period `q`: Temporal increase in threshold by firing of the neuron to induce spiking sparsity. Simplified version of the code:
  ````
  # z[t]: output spikes, u[t]: real part of the membrane potential, t: time step
  z[t] = (u[t] > theta + q[t-1])
  q[t] = 0.9 * q[t-1] + z[t]
  ````
- Divergence boundary `p(omega)`: ensure the neurons converge. Constant value over sequence.
- Smooth reset: dampening factor decreased when neuron spikes; amplitude of oscillation decays faster temporarily. This is done by implementing the refractory period into the dampening equation.
  ````
  b[t] = p(omega) - b_offset - q[t-1]
  ````

  
## Classification tasks
The following classification tasks were considered: Sequential MNIST (S-MNIST), permuted S-MNIST (PS-MNIST), ECG-QTDB, and Spiking Heidelberg Dataset (SHD)

![tnpt_dataset](https://github.com/AdaptiveAILab/brf-neurons/assets/64919377/64f003b5-7dfd-44c0-b622-844e5b2067a3)

## Results

The performance (accuracy), spiking sparsity and the convergence exceeded that of the modern ALIF-RSNN with less trainable parameters and smaller network structure.

The BRF-RSNN consistently yielded better performance and spiking sparsity compared to vanilla RF network (without reset mechanism). Convergence between the BRF and RF network were similar for SHD and ECG, as the resonator neurons only required the learning of smaller angular frequencies. Thus, no severe divergence of the system were seen for these datasets (in comparison to e.g. PS-MNIST, for which no learning took place with RF-RSNN)

Dots on the figure show at which epoch the model learned 95% of the final saved accuracy.

![conv_rf_brf_alif](https://github.com/AdaptiveAILab/brf-neurons/assets/64919377/5cb769c9-cac9-4c95-9b6d-2c4bd4380b38)


### Understanding the convergence

In ongoing research [[Higuchi, Bohté, and Otte, 2024]]() we studied the training convergence of BRF model found that a reason for its stable and fast convergence


## Publications and BibTeX 

- Saya Higuchi, Sebastian Kairat, Sander M. Bohté, and Sebastian Otte (2024). **Balanced Resonate-and-Fire Neurons**. *International Conference on Machine Learning (ICML)*. Accepted for publication. arXiv preprint [arXiv:2402.14603](https://arxiv.org/abs/2402.14603).

```
@misc{higuchi2024balanced,
      title={Balanced Resonate-and-Fire Neurons}, 
      author={Saya Higuchi and Sebastian Kairat and Sander M. Bohte and Sebastian Otte},
      year={2024},
      eprint={2402.14603},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```


- Saya Higuchi, Sander M. Bohté, and Sebastian Otte (2024). **Understanding the Convergence in Balanced Resonate-and-Fire Neurons**. *First Austrian Symposium on AI, Robotics, and Vision*. Accepted for publication. arXiv preprint [arXiv:2406.00389](https://arxiv.org/abs/2406.00389).


```
@misc{higuchi2024understanding,
      title={Understanding the Convergence in Balanced Resonate-and-Fire Neurons}, 
      author={Saya Higuchi and Sander M. Bohte and Sebastian Otte},
      year={2024},
      eprint={2406.00389},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

## Guide 

### Dependencies
Provided as a reference.
````
python 3.10.4
pytorch 2.0.1
pytorch-cuda 11.7
torchvision 0.15.2
tensorboard 2.11.0
scipy 1.9.3
````

Additional packages necessary for downloading the SHD
````
tensorflow 2.11.0 (or any version compabitle with the python version)
tables 3.9.2
````

### Getting started
#### Structure 
`experiments` consist of code and data used for simulations. It also includes saved models, csv files of training convergence, and extended study on error landscape and noise robustness for S-MNIST.

`snn` is a custom python package including custom torch.nn.Module classes for network and neuron types. The surrogate gradient function and forward gradient injection method are defined in `function/autograd.py`

````
├── experiments
|   ├── ecg: python code for training model on ECG-QTDB.
|   ├── SHD: python code for training model on Spiking Heidelberg Dataset.
|   └── smnist: python code for training model on sequential MNIST and permuted S-MNIST.
└── snn
    ├── modules: ALIF, BRF and RF Cells.
    ├── models: network structure.
    └── functional: surrogate gradient functions and base functions.
````

#### Loading the dataset
- S-MNIST and PS-MNIST: MNIST dataset loaded from torchvision when running `smnist_train.py`.
- ECG QTDB: Preprocessed train and test dataset from Yin et al., (2021) available in `ecg/data`.
- SHD: run `generate_data_shd.py` to first download the dataset and precrocess to same sequence length.

It may be necessary to create `data` directory manually.

#### Running inference
For each dataset, `<dataset>_model_stats.py` lets you run the saved trained models, which loads the model and computes the performance stats from the inference phase.
To change the spiking neuron models, and with it the model, simply change the string 
````
neuron = "brf" # "vrf" or "alif"
````
in the script.

#### Training the models
`<dataset>_train.py` and `<dataset>_alif_train.py` were used to train the BRF, RF and ALIF models. To train, simply run the script
````
python smnist_train.py
````
 Tensorboard is used to track the convergence during training and require `runs` directory.
 
The script runs the training (90% of training dataset), validation (10% of traning dataset) and inference phase for the set amount of epochs in batches. Intermediate training, validation and test loss as well as accuracy are printed to the terminal. The model with the lowest validation loss is saved.
some hyperparameters of the model are

>`input_size: int` dimension of the input
>
>`hidden_size: int` number of neurons in the hidden layer
>
>`num_classes: int` number of output neurons
>
>`sequence_length: int` number of time steps per data sample (e.g. one image in MNIST 28-by-28 pixels -> sequence of 784)
>
>`label_last: bool` `True` if the network only considers the last time step for computing loss. If `False`, the loss is computed over the whole seqeunce.
>
>`sub_seq_length: int` number of time steps (at the beginning of the sequence) disregarded in the loss computation, only used if `label_last=False`.

`tools.py` in each dataset directory stows helper functions required for simulations. 

#### `snn` package

To apply snn, add path to sys:
````
import sys
sys.path.append('../..') # where 'snn' can be found in relation to your code
import snn
````

