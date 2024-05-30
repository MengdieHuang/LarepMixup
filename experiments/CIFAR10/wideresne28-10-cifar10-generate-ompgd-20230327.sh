source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

#buchong cifar10 om-fgsm wideresnet28_10-acc96.70 eps=0.05 step_size=0.05
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --latentattack --attack_mode fgsm --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.05 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-omfgsm-eps0.05-20230327.log 2>&1

#-----------------------------

# cifar10 ompgd wideresnet28_10-acc96.70 eps=0.05 step_size=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.05-step0.0078-iter20-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=0.1 step_size=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.1 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.1-step0.0078-iter20-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=0.2 step_size=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.2 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.2-step0.0078-iter20-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=0.3 step_size=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.3 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.3-step0.0078-iter20-20230327.log 2>&1