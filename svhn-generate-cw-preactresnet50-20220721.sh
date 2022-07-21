source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # generate svhn cw adversarial examples
# #----------------------preactresnet18---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name preactresnet18-svhn --cla_model preactresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet18-svhn/20220627/00003-testacc-95.97/train-svhn-dataset/standard-trained-classifier-preactresnet18-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-preactresnet18-cw-generate-20220630.log 2>&1

# #----------------------preactresnet34---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name preactresnet34-svhn --cla_model preactresnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet34-svhn/20220627/00003-testacc-95.75/train-svhn-dataset/standard-trained-classifier-preactresnet34-on-clean-svhn-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-preactresnet34-cw-generate-20220630.log 2>&1

#----------------------preactresnet50---------------------------------
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220720.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name preactresnet50-svhn --cla_model preactresnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/preactresnet50-svhn/20220627/00001-testacc-95.76/train-svhn-dataset/standard-trained-classifier-preactresnet50-on-clean-svhn-epoch-0013.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-preactresnet50-cw-generate-20220720.log 2>&1

# #----------------------alexnet---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name alexnet-svhn --cla_model alexnet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/alexnet-svhn/20210909/00003-testacc-0.9475/train-svhn-dataset/standard-trained-classifier-alexnet-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-alexnet-cw-generate-20220630.log 2>&1

# #----------------------resnet18---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name resnet18-svhn --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-svhn/20210909/00000-testacc-0.9319/train-svhn-dataset/standard-trained-classifier-resnet18-on-clean-svhn-epoch-0010.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-resnet18-cw-generate-20220630.log 2>&1

# #----------------------resnet34---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name resnet34-svhn --cla_model resnet34 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet34-svhn/20210909/00004-testacc-0.9322/train-svhn-dataset/standard-trained-classifier-resnet34-on-clean-svhn-epoch-0009.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-resnet34-cw-generate-20220630.log 2>&1

# #----------------------resnet50---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name resnet50-svhn --cla_model resnet50 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet50-svhn/20210910/00002-testacc-0.9272/train-svhn-dataset/standard-trained-classifier-resnet50-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-resnet50-cw-generate-20220630.log 2>&1

# #----------------------vgg19---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name vgg19-svhn --cla_model vgg19 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/vgg19-svhn/20210910/00003-testacc-0.9573/train-svhn-dataset/standard-trained-classifier-vgg19-on-clean-svhn-epoch-0012.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-vgg19-cw-generate-20220630.log 2>&1

# #----------------------densenet169---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name densenet169-svhn --cla_model densenet169 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/densenet169-svhn/20210910/00000-testacc-0.9293/train-svhn-dataset/standard-trained-classifier-densenet169-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-densenet169-cw-generate-20220630.log 2>&1

# #----------------------googlenet---------------------------------
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20220630.py run --save_path /home/data/maggie/result-newhome --mode attack --attack_mode cw --whitebox --exp_name googlenet-svhn --cla_model googlenet --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/googlenet-svhn/20210910/00001-testacc-0.9382/train-svhn-dataset/standard-trained-classifier-googlenet-on-clean-svhn-epoch-0011.pkl --dataset svhn >> /home/maggie/mmat/log/SVHN/attack-example-generate/cw/svhn-googlenet-cw-generate-20220630.log 2>&1


