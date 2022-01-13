source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# #----------------------pgd 可视化---------------------------------
# #alexnet svhn pgd 20220111  epsilon = 0.02
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.02 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220113.log 2>&1

# #alexnet svhn pgd 20220111  epsilon = 0.05
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.05 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220113.log 2>&1

# #alexnet svhn pgd 20220111  epsilon = 0.1
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.1 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220113.log 2>&1

# #alexnet svhn pgd 20220111  epsilon = 0.2
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.2 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220113.log 2>&1

# #alexnet svhn pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220113.log 2>&1

#-------------------------om pgd 可视化------------------------------------
# alexnet svhn om-pgd 20220111  epsilon = 0.02
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.02 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220113.log 2>&1

# alexnet svhn om-pgd 20220111  epsilon = 0.05
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.05 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220113.log 2>&1


# alexnet svhn om-pgd 20220111  epsilon = 0.1
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.1 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220113.log 2>&1


# alexnet svhn om-pgd 20220111  epsilon = 0.2
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.2 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220113.log 2>&1


# alexnet svhn om-pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220113.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220113.log 2>&1
