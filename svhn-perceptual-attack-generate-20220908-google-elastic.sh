source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#---------------googlenet------------------
# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-googlenet-perceptual-attack-20220908.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-densenet169-perceptual-attack-20220908.log 2>&1