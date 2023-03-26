source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate resnet18 20230326

# ----------------------resnet18-acc96.41---------------------------------
# # Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 10
# # ----------------------resnet18-acc96.41---------------------------------
# # cifar10 pgd resnet18-acc96.41 eps=0.031 eps_step=0.0078 max_iteration=10 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 50
# # ----------------------resnet18-acc96.41---------------------------------
# # cifar10 pgd resnet18-acc96.41 eps=0.031 eps_step=0.0078 max_iteration=50 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 50 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter50-20230326.log 2>&1

# # Perturbation = 8/255=0.031 with step size 2/255=0.0078 max_iter 200
# # ----------------------resnet18-acc96.41---------------------------------
# # cifar10 pgd resnet18-acc96.41 eps=0.031 eps_step=0.0078 max_iteration=200 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --optimizer sgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 200 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter200-20230326.log 2>&1


# # cifar10 pgd resnet18-acc96.41 eps=0.05 eps_step=0.0125 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd resnet18-acc96.41 eps=0.1 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd resnet18-acc96.41 eps=0.2 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1

# # cifar10 pgd resnet18-acc96.41 eps=0.3 eps_step=0.0078 max_iteration=20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc96.41-pgd-generate-eps_step0.0078-max_iter10-20230326.log 2>&1


