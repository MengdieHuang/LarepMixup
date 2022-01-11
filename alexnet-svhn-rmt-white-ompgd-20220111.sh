source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
#---------------------OM FGSM 第1遍---------------------------------
# alexnet svhn dual rmt cle+mix lr =0.001 beta(0.5, 0.5)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 0.5 --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset XXXXXX --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/alexnet-svhn-rmt/alexnet-svhn-rmt-ompgd-20220111.log 2>&1
