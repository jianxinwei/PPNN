nohup python local.py --dataset=farm --epochs=100 --gpu=1 > ../logs/local_farm_sgd.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --gpu=1 --optim=adam > ../logs/local_farm_adam.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --gpu=1 --optim=adafactor > ../logs/local_farm_adafactor.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --gpu=1 --optim=novograd > ../logs/local_farm_novograd.log 2>&1 &

nohup python local.py --dataset=farm --epochs=100 --dp --gpu=1 > ../logs/local_farm_sgd_dp.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --dp --gpu=1 --optim=adam > ../logs/local_farm_adam_dp.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --dp --gpu=1 --optim=adafactor > ../logs/local_farm_adafactor_dp.log 2>&1 &
nohup python local.py --dataset=farm --epochs=100 --dp --gpu=1 --optim=novograd > ../logs/local_farm_novograd_dp.log 2>&1 &