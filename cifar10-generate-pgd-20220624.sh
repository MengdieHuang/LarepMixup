source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#----------------------preactresnet18---------------------------------
#preactresnet18 cifar autoattack 20220218 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/preactresnet18-pgd-generate.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/preactresnet18-pgd-generate.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/preactresnet18-pgd-generate.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/preactresnet18-pgd-generate.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/preactresnet18-pgd-generate.log 2>&1
