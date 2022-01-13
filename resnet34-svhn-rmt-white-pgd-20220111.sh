source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
# #---------------------PGD eps=0.3---------------------------------
# # resnet34 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.3 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet34-svhn/20220111/00000-eps-0.3-acc-10.4833/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet34-svhn-rmt/resnet34-svhn-rmt-pgd-20220111.log 2>&1

# #---------------------PGD eps=0.02---------------------------------
# resnet34 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/resnet34-svhn/20220113/00001-pgd-eps-0.02-acc-76.3022/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/resnet34-svhn-rmt/resnet34-svhn-rmt-pgd-20220111.log 2>&1
#可运行

