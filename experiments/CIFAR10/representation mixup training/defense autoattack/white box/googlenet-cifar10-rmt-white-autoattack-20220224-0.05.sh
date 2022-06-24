source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# Dual RepMixup
# googlenet cifar10 autoattack dual rmt cle+mix lr =0.001 beta(2.0, 2.0) eps=0.05
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220218.py run --mode defense --defense_mode rmt --beta_alpha 2.0 --attack_mode autoattack --attack_eps 0.05 --blackbox --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --adv_dataset /home/maggie/mmat/result/attack/autoattack/googlenet-cifar10/20220223/00001-eps0.05-acc19.90/attack-cifar10-dataset/samples --batch_size 256 --epochs 40 --lr 0.001 >> /home/maggie/mmat/log/googlenet-cifar10-rmt/googlenet-cifar10-rmt-autoattack-20220224-0.05.log 2>&1
