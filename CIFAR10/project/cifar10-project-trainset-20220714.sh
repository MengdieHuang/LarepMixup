source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# project cifar10 testset
#--------------------
CUDA_VISIBLE_DEVICES=1 python tasklauncher-20220713.py run --mode project --exp_name stylegan2ada-cifar10 --gen_model stylegan2ada --dataset cifar10 --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl >> /home/maggie/mmat/log/CIFAR10/dataset-project/trainset/stylegan2ada-cifar10-trainset-project-20220714.log 2>&1