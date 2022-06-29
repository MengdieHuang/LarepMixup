source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#----------------------preactresnet18---------------------------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/preactresnet18-cw-generate-20220625.log 2>&1

#----------------------preactresnet34---------------------------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name preactresnet34-cifar10 --cla_model preactresnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet34-cifar10/20210927/00002-testacc-83.57/train-cifar10-dataset/standard-trained-classifier-preactresnet34-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/preactresnet34-cw-generate-20220625.log 2>&1

#----------------------preactresnet50---------------------------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220625.py run --mode attack --attack_mode cw --whitebox --exp_name preactresnet50-cifar10 --cla_model preactresnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet50-cifar10/20210925/00003-testacc-84.74/train-cifar10-dataset/standard-trained-classifier-preactresnet50-on-clean-cifar10-epoch-0021.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cw/preactresnet50-cw-generate-20220625.log 2>&1