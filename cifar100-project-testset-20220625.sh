source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# project cifar100 testset
#--------------------
CUDA_VISIBLE_DEVICES=1 python tasklauncher-20220624.py run --mode project --exp_name stylegan2ada-cifar100 --gen_model stylegan2ada --dataset cifar100 --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar100/20210723/00000/cifar100-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl >> /home/maggie/mmat/log/CIFAR100/dataset-project/tesetset/stylegan2ada-cifar100-tesetset-project-20220625.log 2>&1

