source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate resnet18 20230325

# ----------------------resnet18-acc77.79---------------------------------
# # cifar10 pgd resnet18-acc77.79 eps=0.03 eps_step=0.005 max_iteration=100 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.03 --attack_eps_step 0.005 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc77.79-pgd-generate-eps_step0.005-max_iter100-20230325.log 2>&1

# # cifar10 pgd resnet18-acc77.79 eps=0.05 eps_step=0.005 max_iteration=100 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc77.79-pgd-generate-eps_step0.005-max_iter100-20230325.log 2>&1

# # cifar10 pgd resnet18-acc77.79 eps=0.1 eps_step=0.005 max_iteration=100 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 --attack_eps_step 0.005 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc77.79-pgd-generate-eps_step0.005-max_iter100-20230325.log 2>&1

# # cifar10 pgd resnet18-acc77.79 eps=0.2 eps_step=0.005 max_iteration=100 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 --attack_eps_step 0.005 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc77.79-pgd-generate-eps_step0.005-max_iter100-20230325.log 2>&1

# # cifar10 pgd resnet18-acc77.79 eps=0.3 eps_step=0.005 max_iteration=100 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 --attack_eps_step 0.005 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc77.79-pgd-generate-eps_step0.005-max_iter100-20230325.log 2>&1



# ----------------------resnet18-acc89.96---------------------------------
# cifar10 pgd resnet18-acc89.96 eps=0.03 eps_step=0.005 max_iteration=100 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030-testacc-0.8996/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.03 --attack_eps_step 0.005 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc89.96-pgd-generate-eps_step0.005-max_iter20-20230325.log 2>&1

# cifar10 pgd resnet18-acc89.96 eps=0.05 eps_step=0.0125 max_iteration=20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030-testacc-0.8996/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc89.96-pgd-generate-eps_step0.005-max_iter20-20230325.log 2>&1

# cifar10 pgd resnet18-acc89.96 eps=0.1 eps_step=0.005 max_iteration=20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030-testacc-0.8996/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.1 --attack_eps_step 0.005 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc89.96-pgd-generate-eps_step0.005-max_iter20-20230325.log 2>&1

# cifar10 pgd resnet18-acc89.96 eps=0.2 eps_step=0.005 max_iteration=20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030-testacc-0.8996/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.2 --attack_eps_step 0.005 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc89.96-pgd-generate-eps_step0.005-max_iter20-20230325.log 2>&1

# cifar10 pgd resnet18-acc89.96 eps=0.3 eps_step=0.005 max_iteration=20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030-testacc-0.8996/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.3 --attack_eps_step 0.005 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-acc89.96-pgd-generate-eps_step0.005-max_iter20-20230325.log 2>&1
