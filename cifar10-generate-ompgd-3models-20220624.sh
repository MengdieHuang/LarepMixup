source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar ompgd generate 20220624
#----------------------preactresnet18---------------------------------
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20220624.py run --mode attack --latentattack --attack_mode pgd --whitebox --exp_name preactresnet18-cifar10 --cla_model preactresnet18 --cla_network_pkl /home/maggie/mmat/result/train/cla-train/preactresnet18-cifar10/20210924/00001-teatacc-87.37/train-cifar10-dataset/standard-trained-classifier-preactresnet18-on-clean-cifar10-epoch-0011.pkl --gen_network_pkl /home/maggie/mmat/result/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_dataset /home/maggie/mmat/result/project/stylegan2ada-cifar10/20210913/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.02 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/ompgd/preactresnet18-ompgd-generate-20220624.log 2>&1

#----------------------preactresnet34---------------------------------


#----------------------preactresnet50---------------------------------
