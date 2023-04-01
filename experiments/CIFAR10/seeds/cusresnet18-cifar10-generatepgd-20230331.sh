source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# ----------------------cusresnet18-acc96.41---------------------------------

# cifar10 pgd cusresnet18-acc96.41 eps=0.05 step size=2/255=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode attack --attack_mode pgd --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-step0.0078-iter20-20230331.log 2>&1

# cifar10 pgd cusresnet18-acc96.41 eps=0.1 step size=2/255=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode attack --attack_mode pgd --attack_eps 0.1 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-step0.0078-iter20-20230331.log 2>&1

# cifar10 pgd cusresnet18-acc96.41 eps=0.2 step size=2/255=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode attack --attack_mode pgd --attack_eps 0.2 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-step0.0078-iter20-20230331.log 2>&1
