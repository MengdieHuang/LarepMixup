source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#----------------------alexnet---------------------------------
#alexnet cifar fgsm 20220111  epsilon = 0.02
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode fgsm --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/cifar10-attack/cifar10-fgsm-attack-20220112.log 2>&1

#alexnet cifar fgsm 20220111  epsilon = 0.05
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode fgsm --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/cifar10-attack/cifar10-fgsm-attack-20220112.log 2>&1

#alexnet cifar fgsm 20220111  epsilon = 0.1
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode fgsm --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/cifar10-attack/cifar10-fgsm-attack-20220112.log 2>&1

#alexnet cifar fgsm 20220111  epsilon = 0.2
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode fgsm --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/cifar10-attack/cifar10-fgsm-attack-20220112.log 2>&1

#alexnet cifar fgsm 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode fgsm --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/cifar10-attack/cifar10-fgsm-attack-20220112.log 2>&1