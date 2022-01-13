source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
#---------------------OM PGD eps=0.3---------------------------------
# resnet50 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220112.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode om-pgd --attack_eps 0.3 --blackbox --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet50-svhn/20220112/00001-ompgd-eps-0.3-acc-0.7300/attack-svhn-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet50-svhn-rmt/resnet50-svhn-rmt-ompgd-20220111.log 2>&1

# #---------------------OM PGD eps=0.02---------------------------------
# # resnet50 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220112.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode om-pgd --attack_eps 0.02 --blackbox --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet50-svhn/20220113/00002-ompgd-eps-0.02-acc-61.2100/attack-svhn-dataset/latent-attack-samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet50-svhn-rmt/resnet50-svhn-rmt-ompgd-20220111.log 2>&1

#可运行
