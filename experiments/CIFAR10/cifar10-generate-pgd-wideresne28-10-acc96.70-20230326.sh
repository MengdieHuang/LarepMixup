source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate wideresnet28_10 20230326

# ----------------------wideresnet28_10-acc96.70---------------------------------
# Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 10
# ----------------------wideresnet28_10-acc96.70---------------------------------
# cifar10 pgd wideresnet28_10-acc96.70 eps=0.031 eps_step=0.0078 max_iteration=10 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 50
# ----------------------wideresnet28_10-acc96.70---------------------------------
# cifar10 pgd wideresnet28_10-acc96.70 eps=0.031 eps_step=0.0078 max_iteration=50 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 50 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter50-20230326.log 2>&1

# Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 200
# ----------------------wideresnet28_10-acc96.70---------------------------------
# cifar10 pgd wideresnet28_10-acc96.70 eps=0.031 eps_step=0.0078 max_iteration=200 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 200 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter200-20230326.log 2>&1


# # cifar10 pgd wideresnet28_10-acc96.70 eps=0.05 eps_step=0.0125 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=0.1 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=0.2 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=0.3 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/wideresnet28_10-acc96.70-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1


