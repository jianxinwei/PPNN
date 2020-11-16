# PrivacyPreservingNN

## Baselines

1) Single model (done)

2) Single model with DP (done)

3) FL model (done)

4) FL model with DP (done)

5) Decentralized model (done)

6) Decentralized model with DP (done)

7) Decentralized model with TPHE (done)

8) Decentralized model with DP and TPHE (done)


## Comparison w.r.t. Optimizer on Farm Dataset

cmp|SGD|Adam|Adafactor|Novograd
---|---|----|---------|--------
memory|1474.9921875 MB|2207.2031250 MB |744.4531250 MB |1471.5000000 MB
time|398.2238694 s | 532.8475882 s |691.1723568 s | 97.9963092 s
test loss | 0.69 | 0.80 | 0.29 | 0.42
test acc | 52.83 | 79.98 | 88.78 | 81.42