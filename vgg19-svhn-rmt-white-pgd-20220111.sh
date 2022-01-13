source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
# #---------------------PGD eps=0.3---------------------------------
# # vgg19 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220111.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.3 --blackbox --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/vgg19-svhn/20220112/00000-eps-0.3-acc-11.9046/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-svhn-rmt/vgg19-svhn-rmt-pgd-20220111.log 2>&1

#---------------------PGD eps=0.02---------------------------------
# vgg19 svhn dual rmt cle+mix lr =0.001 beta(2.0, 2.0)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220112.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode pgd --attack_eps 0.02 --blackbox --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/pgd/vgg19-svhn/20220112/00000-eps-0.3-acc-11.9046/attack-svhn-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/vgg19-svhn-rmt/vgg19-svhn-rmt-pgd-20220111.log 2>&1

