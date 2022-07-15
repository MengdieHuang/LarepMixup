source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# project svhn testset
#--------------------
CUDA_VISIBLE_DEVICES=1 python tasklauncher-20220713.py run --mode project --exp_name stylegan2ada-svhn --gen_model stylegan2ada --dataset svhn --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl >> /home/maggie/mmat/log/SVHN/dataset-project/trainset/stylegan2ada-svhn-trainset-project-20220714.log 2>&1

