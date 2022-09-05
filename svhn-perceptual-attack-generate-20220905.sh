source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# ---------------alexnet------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-alexnet-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-alexnet-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-alexnet-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-alexnet-perceptual-attack-20220905.log 2>&1

# ---------------resnet18------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet18-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet18-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet18-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet18-perceptual-attack-20220905.log 2>&1

#---------------resnet34------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet34-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet34-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet34-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet34-perceptual-attack-20220905.log 2>&1

#---------------resnet50------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-resnet50-perceptual-attack-20220905.log 2>&1

#---------------vgg19------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-vgg19-perceptual-attack-20220905.log 2>&1

#---------------densenet169------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-densenet169-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-densenet169-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-densenet169-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-densenet169-perceptual-attack-20220905.log 2>&1

#---------------googlenet------------------
# fog
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode fog --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-googlenet-perceptual-attack-20220905.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode snow --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-googlenet-perceptual-attack-20220905.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode elastic --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-googlenet-perceptual-attack-20220905.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220905.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/perceptual/svhn-googlenet-perceptual-attack-20220905.log 2>&1