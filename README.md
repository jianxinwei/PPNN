# Privacy Preserving Horizontal Federated Learning Framework for Financial Data
Autumn 2020. A decentralized horizontal federated learning framework with differential privacy and threshold Paillier homomorphic encryption, the course project of CS6203 Advanced Topics in Database Systems, National University of Singapore.


## Comparison w.r.t. Optimizer on Farm Dataset

### CPU Setting

cmp|SGD|Adam|Adafactor|Novograd
---|---|----|---------|--------
memory|1474.9921875 MB|2207.2031250 MB |744.4531250 MB |1471.5000000 MB
time|398.2238694 s | 532.8475882 s |691.1723568 s | 97.9963092 s
test loss | 0.69 | 0.80 | 0.29 | 0.42
test acc | 52.83 | 79.98 | 88.78 | 81.42

### GPU Setting

cmp|SGD|Adam|Adafactor|Novograd
---|---|----|---------|--------
memory| 5551MiB MB | 6991 MB | 6251MB | 5551 MB
time| 11.7862406 s | 20.5685672s | 25.7762453s | 17.8444077s
test loss | 0.69 | 0.43 | 0.29 | 0.70
test acc | 52.83 | 87.94 | 90.23 | 52.83