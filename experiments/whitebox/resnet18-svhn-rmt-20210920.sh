source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# resnet18 svhn rmt 20210920000 only mix
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210920.py run --mode defense --defense_mode rmt --attack_mode fgsm --attack_eps 0.2 --whitebox True --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --gen_model stylegan2ada --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-svhn/20210718/00000/svhn-auto1-batch64-ada-bgc-noresume/network-snapshot-025000.pkl --dataset svhn --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-svhn/20210914/00000/project-svhn-trainset --adv_dataset /home/maggie/mmat/result/attack/fgsm/resnet18-svhn/20210914/00000-attackacc-0.29894/attack-svhn-dataset/samples --batch_size 1024 --epochs 50 --lr 0.1 >> /home/maggie/mmat/log/resnet18-svhn-rmt/resnet18-svhn-rmt-20210920.log 2>&1