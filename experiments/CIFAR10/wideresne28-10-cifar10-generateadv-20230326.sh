source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# ----------------------wideresnet28_10-acc96.70---------------------------------
#1 cifar10 fgsm wideresnet28_10-acc96.70 eps=8/255=0.031
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode fgsm --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-fgsm-eps0.031-20230326.log 2>&1

#1 cifar10 om-fgsm wideresnet28_10-acc96.70 eps=8/255=0.031
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --latentattack --attack_mode fgsm --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_dataset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --attack_eps 0.031 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-omfgsm-eps0.031-20230326.log 2>&1

# cifar10 autoattack wideresnet28_10-acc96.70 eps=8/255=0.031  step size=2/255=0.0078  max_iter buzhichishezhi 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode autoattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-autoattack-eps0.031-eps_step0.0078-20230326.log 2>&1

# cifar10 deepfool wideresnet28_10-acc96.70 eps=8/255=0.031  step size=2/255=0.0078  max_iter 10 yuanlaishi100
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode deepfool --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-deepfool-eps0.031-max_iter10-20230326.log 2>&1


# cifar10 cw wideresnet28_10-acc96.70 eps=8/255=0.031  step size=2/255=0.0078  confidence 0 max_iter 10 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230326.py run --mode attack --attack_mode cw --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --confidence 0 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-cw-conf0.00-max_iter10-20230326.log 2>&1