source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# rmt seed1 wideresnet28-10 cifar10 evaluate grey-box acc

# # rmt seed1 eval vanilla PGD(eps031-step0078-iter20)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whitepgd-eps031-step0078-iter20-20230328.log 2>&1

# rmt seed1 eval vanilla FGSM(eps031)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --attack_eps 0.031 --attack_eps_step 0.031 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00001-eps031-acc52.82/attack-cifar10-dataset/samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whitefgsm-eps0.031-20230328.log 2>&1

# # rmt seed1 eval vanilla AutoAttack(eps031,step0078)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode autoattack --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/autoattack/wideresnet28_10-cifar10/20230327/00001-eps031-step0078-acc2.35/attack-cifar10-dataset/samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whiteautoattack-eps0.031-20230328.log 2>&1

# # rmt seed1 eval vanilla Deepfool(eps031,iter10)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-20230328.log 2>&1

# # rmt seed1 eval vanilla CW(conf0,iter10)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode cw --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/cw/wideresnet28_10-cifar10/20230327/00001-conf0-iter10-acc0.14/attack-cifar10-dataset/samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whitecw-eps0.031-20230328.log 2>&1

# # rmt seed1 eval vanilla OM-FGSM(eps0.05-step0.05)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode fgsm --latentattack --attack_eps 0.05 --attack_eps_step 0.05 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/fgsm/wideresnet28_10-cifar10/20230328/00003-om-eps031-acc49.91/attack-cifar10-dataset/latent-attack-samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whiteomfgsm-eps0.05-20230328.log 2>&1

# # rmt seed1 eval vanilla OM-PGD(eps0.05-step0.078,iter20)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230328.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/wideresnet28_10-cifar10/blackbox/20230328/00003-cleacc8980/rmt-cifar10-dataset/rmt-trained-wideresnet28_10-on-cifar10-epoch-0040-cleacc-0.8980-advacc-0.7816.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230328/00005-om-eps05-step0078-iter20-acc31.11/attack-cifar10-dataset/latent-attack-samples/test --batch_size 128 >> /home/maggie/mmat/log/CIFAR10/eval/rmt-seed1-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-20230328.log 2>&1


#------------------------------------------------------------------------------