source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat


# ----------------------wideresnet28_10-acc96.70---------------------------------

#1 cifar10 fgsm wideresnet28_10-acc96.70 eps=8/255=0.031
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230326.py run --mode attack --attack_mode fgsm --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/comparepgdfgsm/wideresnet28_10-acc96.70-generate-fgsm-eps0.031-20230326.log 2>&1

# cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 1 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.031 --attack_max_iter 1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/comparepgdfgsm/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.031-max_iter1-20230326.log 2>&1

# cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 1 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/comparepgdfgsm/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter1-20230326.log 2>&1