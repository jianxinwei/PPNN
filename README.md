# Privacy Preserving Horizontal Federated Learning Framework for Financial Data
Autumn 2020. A decentralized horizontal federated learning framework with differential privacy and threshold partially homomorphic encryption, the course project of CS6203 Advanced Topics in Database Systems, National University of Singapore.

## Requirement
python 3.8.3+

matplotlib 3.3.3+

torch 1.7.0+

pandas 1.1.4+

numpy 1.19.4+

phe 1.4.0+

numba 0.51.2+

sympy 1.6.2+

multipledispatch 0.6.0+

memory_profiler 0.58.0+

opacus

ipdb

## Installation
We suggest to set up the environments by anaconda3 or miniconda3.
To install the required libraries, just run
```pip install -r requirements.txt```.

## Dataset
We provide four datasets in this repo: Bank, Bank\_Rnd, Bidding, and Credit. You can easily specify the dataset by setting up the
```--dataset```
argument when running the program.
Besides, we also provide foyr optimizers: SGD, Adam, Novograd, and Adafactor, where Adam, Novograd and Adafactor are adaptive optimizers, and Adafactor suuports parameter matrix factorization, which can save a lot memory during training.
One can easily use any of them by specifying 
```--optim``` 
argument.

## Usage
We have set up some default parameters for training, all default parameters are listed in 
```/utils/options.py```
, one is suugested to modify this file to configure his own default training setting if it is needed.

### Local Model
To run local model, type the following command in the terminal:

```
cd local

python local.py python local_client.py --dataset=bidding --epochs=100 --optim=sgd --dim_hidden=8 --dp --gpu=1
```

```--dp``` 
argument and 
```--gpu``` 
determines whether or not to enable differential privacy and gpu running respectively, while 
```--epochs```
, ```dim_hidden```, and 
```--optim``` 
arguments refer to the running epochs, hidden layer dimension, and optimizer of the model.

### Client-Server Model
Our default setting up is for the framework is 1 central server node and 5 client nodes.
Each node is assigned a unique rank id, which the rank id 0 is kept for the server node.
For client nodes, the assigned rank id should start from 1 increasingly.
To simulate the federated learning framework, we provide a json file 
```ip_port_client_server.json```
 to simulate the multi-machine running encironments.
One can easily modify the ip and port in this file to specify the network configuration.

For running,

```
cd client-server

python server.py --rank=0 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python client.py --rank=1 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python client.py --rank=2 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python client.py --rank=3 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python client.py --rank=4 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python client.py --rank=5 --dataset=bidding --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

```

For the simplicity of using, we provie a scripts to run them in a single machine, one can use the following command to run the scripts.


```
cd client-server

chmod +x run.sh

./run.sh
```

### Decentralized Model
In decentralized framework, we support both differential pri
vacy and threshold partially homomorphic encryption. The default set up is also 5 nodes in the network. The rank id for each node is assigned from 0.
The network configuration is set up in the json file 
```ip_port.json```
. To run the decentralized framework, one can run the following command:

```
cd decentralized 

python node.py --dataset=bidding --dp --tphe --rank=0 --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python node.py --dataset=bidding --dp --tphe --rank=1 --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python node.py --dataset=bidding --dp --tphe --rank=2 --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python node.py --dataset=bidding --dp --tphe --rank=3 --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

python node.py --dataset=bidding --dp --tphe --rank=4 --epochs=100 --gpu=1 --dim_hidden=8 --optim=sgd

```

Where, the 
```--dp```
 argument and the 
 ```---tphe``` 
 argument refers to whether enable differential privacy and thershold partially homomorphic encryption or not.
Also for the simplicity of using, we provide a simple shell scripts for users to run for a simulation in a single machine.

```
cd decentralized

chmod +x run.sh

./run.sh
```

### Running Output

For each of the above commands, the running will generate a best model and the final model for the respectively training framework, both are stored in the 
```save```
 directory. An addtional figure about the variation of training loss and valid loss is also provided in the directory for users' further references.
If a user runs the default scripts we have provided, some addtional logs will be generated in
 ```log```
  directory. We suggest users to refer to these logs to for an overview of  the overall training trend. 

### Further Support
If you have any enquiries, please don't hesitate to contact huangwentao@u.nus.edu (Huang Wentao) or jianxinwei@u.nus.edu (Wei Jianxin) for the further support, thank you.




