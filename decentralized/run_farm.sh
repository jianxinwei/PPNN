nohup python node.py --dataset=farm --dp --tphe --gpu=1 --rank=0 > ../logs/decentralized_farm_dp_tphe_sgd_0.log 2>&1 &
nohup python node.py --dataset=farm --dp --tphe --gpu=1 --rank=1 > ../logs/decentralized_farm_dp_tphe_sgd_1.log 2>&1 &
nohup python node.py --dataset=farm --dp --tphe --gpu=1 --rank=2 > ../logs/decentralized_farm_dp_tphe_sgd_2.log 2>&1 &
nohup python node.py --dataset=farm --dp --tphe --gpu=1 --rank=3 > ../logs/decentralized_farm_dp_tphe_sgd_3.log 2>&1 &
nohup python node.py --dataset=farm --dp --tphe --gpu=1 --rank=4 > ../logs/decentralized_farm_dp_tphe_sgd_4.log 2>&1 &