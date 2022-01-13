source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
# # ---------------------PGD eps=0.3---------------------------------
# # densenet169 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.3 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/densenet169-svhn/20220112/00000-eps-0.3-acc-8.5433/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/densenet169-svhn-rmt/densenet169-svhn-rmt-pgd-20220111.log 2>&1

# ---------------------PGD eps=0.02---------------------------------
# densenet169 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/densenet169-svhn/20220113/00002-pgd-eps-0.02-acc-65.8805/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/densenet169-svhn-rmt/densenet169-svhn-rmt-pgd-20220111.log 2>&1

#可运行