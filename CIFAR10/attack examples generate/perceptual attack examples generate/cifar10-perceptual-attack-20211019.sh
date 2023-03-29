source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#---------------alexnet------------------
# # fog
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-attack-20211019.log 2>&1

# # snow
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-attack-20211019.log 2>&1

# # elastic
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-attack-20211019.log 2>&1

# # jpeg
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name alexnet-cifar10 --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-cifar10/20210908/00015-testacc-0.8355/train-cifar10-dataset/standard-trained-classifier-alexnet-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-alexnet-attack-20211019.log 2>&1

#---------------resnet18------------------
# # fog
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-attack-20211019.log 2>&1

# # snow
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-attack-20211019.log 2>&1

# # elastic
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-attack-20211019.log 2>&1

# # jpeg
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet18-attack-20211019.log 2>&1

# #---------------resnet34------------------
# # fog
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-attack-20211019.log 2>&1

# # snow
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-attack-20211019.log 2>&1

# # elastic
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-attack-20211019.log 2>&1

# # jpeg
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet34-cifar10 --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-cifar10/20210907/00030-testacc-0.7739/train-cifar10-dataset/standard-trained-classifier-resnet34-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet34-attack-20211019.log 2>&1

#---------------resnet50------------------
# fog
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-attack-20211019.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-attack-20211019.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-attack-20211019.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name resnet50-cifar10 --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-cifar10/20210908/00002-attackacc-0.7624/train-cifar10-dataset/standard-trained-classifier-resnet50-on-clean-cifar10-finished.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-resnet50-attack-20211019.log 2>&1

#---------------vgg19------------------
# fog
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-attack-20211019.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-attack-20211019.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-attack-20211019.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name vgg19-cifar10 --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-cifar10/20210908/00002-testacc-0.8705/train-cifar10-dataset/standard-trained-classifier-vgg19-on-clean-cifar10-epoch-0013.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-vgg19-attack-20211019.log 2>&1

#---------------densenet169------------------
# fog
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-attack-20211019.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-attack-20211019.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-attack-20211019.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name densenet169-cifar10 --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-cifar10/20210909/00003-testacc-0.7929/train-cifar10-dataset/standard-trained-classifier-densenet169-on-clean-cifar10-epoch-0012.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-densenet169-attack-20211019.log 2>&1

#---------------googlenet------------------
# fog
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode fog --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-attack-20211019.log 2>&1

# snow
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode snow --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-attack-20211019.log 2>&1

# elastic
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode elastic --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-attack-20211019.log 2>&1

# jpeg
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20211019.py run --mode attack --perceptualattack --attack_mode jpeg --exp_name googlenet-cifar10 --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-cifar10/20210909/00001-testacc-0.8016/train-cifar10-dataset/standard-trained-classifier-googlenet-on-clean-cifar10-epoch-0011.pkl --dataset cifar10 >> /home/maggie/mmat/log/cifar10-attack/cifar10-googlenet-attack-20211019.log 2>&1