source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar pgd generate 20230325
#----------------------resnet18---------------------------------
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-20230325.log 2>&1


# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.03 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.03 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.0078-max_iter7-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.0078-max_iter7-20230325.log 2>&1



# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.03 --attack_eps_step 0.0078 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.0078-max_iter100-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-eps_step0.0078-max_iter100-20230325.log 2>&1







# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.2 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-20230325.log 2>&1

# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20230325/00030/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished-acc89.96.pkl --dataset cifar10 --attack_eps 0.3 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/resnet18-pgd-generate-20230325.log 2>&1



#-------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.03 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/retrain-oldresnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/retrain-oldresnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/retrain-oldresnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1

CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230325.py run --mode attack --attack_mode pgd --whitebox --exp_name resnet18-cifar10 --cla_model resnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/resnet18-cifar10/20210908/00017-testacc-0.7779/train-cifar10-dataset/standard-trained-classifier-resnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.01 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/pgd/newgen/retrain-oldresnet18-pgd-generate-eps_step0.1-max_iter100-20230325.log 2>&1