nohup python main_one_node.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/one_node_bidding.log 2>&1 &
nohup python main_one_node_dp.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/one_noded_dp_bidding.log 2>&1 &
# nohup python main_single.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/single_bidding.log 2>&1 &
# nohup python main_single_dp.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/single_dp_bidding.log 2>&1 &
nohup python main_fl.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/fl_bidding.log 2>&1 &
# nohup python main_fl_dp.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/fl_dp_bidding.log 2>&1 &
nohup python main_decentralized.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/decentralized_bidding.log 2>&1 &
# nohup python main_decentralized_dp.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/decentralized_dp_bidding.log 2>&1 &
# nohup python main_decentralized_tphe.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/decentralized_tphe_bidding.log 2>&1 &
# nohup python main_decentralized_tphe_dp.py --epochs=100 --dataset=bidding --lr=0.001 --dim_hidden=4 > logs/decentralized_tphe_dp_bidding.log 2>&1 &