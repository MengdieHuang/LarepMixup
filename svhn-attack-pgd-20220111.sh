source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#----------------------alexnet---------------------------------
# alexnet svhn pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # alexnet svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------resnet18------------------------------------
# resnet18 svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # resnet18 svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------resnet34------------------------------------
# resnet34 svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # resnet34 svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------resnet50------------------------------------
# resnet50 svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # resnet50 svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------vgg19------------------------------------
# vgg19 svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # vgg19 svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------densenet169------------------------------------
# densenet169 svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # densenet169 svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

#---------------------------googlenet------------------------------------
# googlenet svhn pgd 20220111  epsilon = 0.3
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --attack_mode pgd --whitebox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-pgd-attack-20220111.log 2>&1

# # googlenet svhn om-pgd 20220111  epsilon = 0.3
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --dataset svhn --attack_eps 0.3 >> /home/maggie/mmat/log/svhn-attack/svhn-ompgd-attack-20220111.log 2>&1

