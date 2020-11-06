python main_single.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/single_best_bidding.pkl
# rm save/single_final_bidding.pkl
python main_single_dp.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/single_dp_best_bidding.pkl
# rm save/single_dp_final_bidding.pkl
python main_fl.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/fl_best_bidding.pkl
# rm save/fl_final_bidding.pkl
python main_fl_dp.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/fl_dp_best_bidding.pkl
# rm save/fl_dp_final_bidding.pkl
python main_decentralized.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/decentralized_best_bidding.pkl
# rm save/decentralized_final_bidding.pkl
python main_decentralized_dp.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/decentralized_dp_best_bidding.pkl
# rm save/decentralized_dp_final_bidding.pkl
python main_decentralized_tphe.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/decentralized_tphe_best_bidding.pkl
# rm save/decentralized_tphe_final_bidding.pkl
python main_decentralized_tphe_dp.py --epochs=3 --local_ep=1 --dim_hidden=4 --dataset=bidding
# rm save/decentralized_tphe_dp_best_bidding.pkl
# rm save/decentralized_tphe_dp_final_bidding.pkl