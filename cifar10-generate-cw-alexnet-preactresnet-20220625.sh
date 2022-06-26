source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar cw generate 20220625
#----------------------alexnet---------------------------------
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --confidence 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/alexnet-cw-generate-20220625.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --confidence 0.5 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/alexnet-cw-generate-20220625.log 2>&1

CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 --confidence 1.0 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/alexnet-cw-generate-20220625.log 2>&1
