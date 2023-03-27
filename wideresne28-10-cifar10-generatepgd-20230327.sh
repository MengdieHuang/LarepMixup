source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# #buchong cifar10 om-fgsm wideresnet28_10-acc96.70 eps=8/255=0.031
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --latentattack --attack_mode fgsm --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-omfgsm-eps0.031-20230327.log 2>&1

# # ----------------------wideresnet28_10-acc96.70---------------------------------
# # cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 1 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter1-20230327.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 7 saveadvtrain
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --saveadvtrain --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter7-20230327.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 10 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter10-20230327.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter20-20230327.log 2>&1

# # cifar10 pgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 50 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 50 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-pgd-eps0.031-eps_step0.0078-max_iter50-20230327.log 2>&1


#-----------------------------
# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 1 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0078-max_iter1-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.031  max_iter 1 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.031 --attack_max_iter 1 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0031-max_iter1-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 7 saveadvtrain
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --saveadvtrain --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0078-max_iter7-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 10 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0078-max_iter10-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0078-max_iter20-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.031 step size=2/255=0.0078  max_iter 50 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 50 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.031-eps_step0.0078-max_iter50-20230327.log 2>&1

# # cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.02 step size=2/255=0.0078  max_iter 20 
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.02 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.02-eps_step0.0078-max_iter20-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.01 step size=2/255=0.0078  max_iter 20 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.01 --attack_eps_step 0.0078 --attack_max_iter 20 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.01-eps_step0.0078-max_iter20-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.01 step size=2/255=0.0078  max_iter 70 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.01 --attack_eps_step 0.0078 --attack_max_iter 70 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.01-eps_step0.0078-max_iter70-20230327.log 2>&1

# cifar10 ompgd wideresnet28_10-acc96.70 eps=8/255=0.01 step size=2/255=0.0078  max_iter 100 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode pgd --latentattack --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --attack_eps 0.01 --attack_eps_step 0.0078 --attack_max_iter 100 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-ompgd-eps0.01-eps_step0.0078-max_iter100-20230327.log 2>&1