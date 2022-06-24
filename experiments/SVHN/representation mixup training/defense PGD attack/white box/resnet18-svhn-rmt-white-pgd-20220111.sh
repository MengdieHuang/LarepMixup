source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
# #---------------------PGD eps=0.3---------------------------------
# # resnet18 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220112.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.3 --blackbox --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet18-svhn/20220111/00000-eps-0.3-acc-9.9992/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-svhn-rmt/resnet18-svhn-rmt-pgd-20220111.log 2>&1

#---------------------PGD eps=0.02---------------------------------
# resnet18 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220112.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet18-svhn/20220113/00001-pgd-eps-0.02-acc-74.3892/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet18-svhn-rmt/resnet18-svhn-rmt-pgd-20220111.log 2>&1
#可运行
