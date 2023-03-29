source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# ----------------------cusresnet18-acc96.41---------------------------------

# cifar10 pgd cusresnet18-acc96.41 eps=8/255=0.031 step size=2/255=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-eps0.031-step0.0078-iter20-20230327.log 2>&1

# cifar10 pgd cusresnet18-acc96.41 eps=8/255=0.031 step size=2/255=0.0078  max_iter 7 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-eps0.031-step0.0078-iter7-20230327.log 2>&1

# cifar10 pgd cusresnet18-acc96.41 eps=8/255=0.031 step size=2/255=0.0078  max_iter 10 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-eps0.031-step0.0078-iter10-20230327.log 2>&1

# cifar10 pgd cusresnet18-acc96.41 eps=8/255=0.031 step size=2/255=0.0078  max_iter 50 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 50 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-eps0.031-step0.0078-iter50-20230327.log 2>&1


# cifar10 pgd cusresnet18-acc96.41 eps=8/255=0.031 step size=2/255=0.0078  max_iter 7 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --saveadvtrain --whitebox --exp_name cusresnet18-cifar10 --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/cusresnet18-acc96.41-generate-pgd-eps0.031-step0.0078-iter7-20230327.log 2>&1