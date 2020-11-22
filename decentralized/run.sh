nohup python node.py --dataset=bidding --dp --tphe --rank=0 --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/decentralized_bidding_dp_tphe_sgd_0.log 2>&1 &
nohup python node.py --dataset=bidding --dp --tphe --rank=1 --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/decentralized_bidding_dp_tphe_sgd_1.log 2>&1 &
nohup python node.py --dataset=bidding --dp --tphe --rank=2 --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/decentralized_bidding_dp_tphe_sgd_2.log 2>&1 &
nohup python node.py --dataset=bidding --dp --tphe --rank=3 --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/decentralized_bidding_dp_tphe_sgd_3.log 2>&1 &
nohup python node.py --dataset=bidding --dp --tphe --rank=4 --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/decentralized_bidding_dp_tphe_sgd_4.log 2>&1 &