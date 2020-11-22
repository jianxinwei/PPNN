nohup python server.py --rank=0 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_0.log 2>&1 &
nohup python client.py --rank=1 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_1.log 2>&1 &
nohup python client.py --rank=2 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_2.log 2>&1 &
nohup python client.py --rank=3 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_3.log 2>&1 &
nohup python client.py --rank=4 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_4.log 2>&1 &
nohup python client.py --rank=5 --dataset=bidding --epochs=100 --dim_hidden=8 --optim=sgd > ../logs/client-server_bidding_sgd_100_rank_5.log 2>&1 &