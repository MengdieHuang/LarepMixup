source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#---------------resnet50------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

#---------------vgg19------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1
