numactl --physcpubind=6 nohup python local_client.py --dataset=bidding --epochs=100 > ../logs/local_client_bidding_sgd.log 2>&1 &
numactl --physcpubind=7 nohup python local_client.py --dataset=bank --epochs=100 > ../logs/local_client_bank_sgd.log 2>&1 &
numactl --physcpubind=8 nohup python local_client.py --dataset=bank_random --epochs=100 > ../logs/local_client_bank_random_sgd.log 2>&1 &