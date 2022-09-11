source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#---------------resnet34------------------
# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet34-perceptual-attack-20220908.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet18-perceptual-attack-20220908.log 2>&1