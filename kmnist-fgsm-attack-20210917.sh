source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # alexnet
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name alexnet-kmnist --cla_model alexnet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/alexnet-kmnist/20210912/00000-testacc-0.9789/train-kmnist-dataset/standard-trained-classifier-alexnet-on-clean-kmnist-finished.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # # resnet-18
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name resnet18-kmnist --cla_model resnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet18-kmnist/20210912/00000-testacc-0.9732/train-kmnist-dataset/standard-trained-classifier-resnet18-on-clean-kmnist-epoch-0018.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # # resnet-34
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name resnet34-kmnist --cla_model resnet34 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet34-kmnist/20210912/00000-testacc-0.9744/train-kmnist-dataset/standard-trained-classifier-resnet34-on-clean-kmnist-epoch-0014.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # resnet-50
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name resnet50-kmnist --cla_model resnet50 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/resnet50-kmnist/20210913/00000-testacc-0.9615/train-kmnist-dataset/standard-trained-classifier-resnet50-on-clean-kmnist-epoch-0019.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # # vgg-19
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name vgg19-kmnist --cla_model vgg19 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/vgg19-kmnist/20210913/00003-testacc-0.9809/train-kmnist-dataset/standard-trained-classifier-vgg19-on-clean-kmnist-epoch-0014.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # # densenet-169
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name densenet169-kmnist --cla_model densenet169 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/densenet169-kmnist/20210913/00000-testacc-0.9752/train-kmnist-dataset/standard-trained-classifier-densenet169-on-clean-kmnist-epoch-0018.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1

# # googlenet
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20210917.py run --mode attack --attack_mode fgsm --exp_name googlenet-kmnist --cla_model googlenet --cla_network_pkl /home/maggie/mmat/result/train/cla-train/googlenet-kmnist/20210913/00003-testacc-0.9804/train-kmnist-dataset/standard-trained-classifier-googlenet-on-clean-kmnist-epoch-0015.pkl --dataset kmnist --attack_eps 0.3 >> /home/maggie/mmat/log/kmnist-attack/kmnist-fgsm-attack-20210917.log 2>&1