source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# eval vanilla wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1

# eval patchmixup-seed1 wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/patchmixup/pgd/maskmixup-bernoullisampler/wideresnet28_10-cifar10/blackbox/20230331/00001-cleacc-85.71/patchmixup-cifar10-dataset/patchmixup-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1

# eval patchmixup-seed2 wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-2/defense/patchmixup/pgd/maskmixup-bernoullisampler/wideresnet28_10-cifar10/blackbox/20230331/00000-cleacc-83.59/patchmixup-cifar10-dataset/patchmixup-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 2 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1

# eval patchmixup-seed3 wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-3/defense/patchmixup/pgd/maskmixup-bernoullisampler/wideresnet28_10-cifar10/blackbox/20230331/00000-cleacc-81.18/patchmixup-cifar10-dataset/patchmixup-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 3 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1

# eval patchmixup-seed4 wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-4/defense/patchmixup/pgd/maskmixup-bernoullisampler/wideresnet28_10-cifar10/blackbox/20230331/00000-cleacc-82.00/patchmixup-cifar10-dataset/patchmixup-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 4 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1

# # eval patchmixup-seed5 wideresnet28_10 on cifar10: om-pgd (eps0.05-step0.005,iter100)
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode pgd --latentattack --attack_eps 0.05 --attack_eps_step 0.005 --attack_max_iter 100 --whitebox --cla_model wideresnet28_10 --cla_network_pkl XXX-patchmixup-seed5-cla-XXX --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_testset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220627/00000/project-cifar10-testset --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230401/00000-om-eps0.05-step0.005-iter100-acc23.70/attack-cifar10-dataset/latent-attack-samples/test --batch_size 64 --seed 5 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/patchmixup-wideresnet28_10-cifar10-against-whiteompgd-eps0.05-step0.005-iter100-20230331.log 2>&1


# # wideresnet28_10 linear patchmixup beta(1,1) maskmixup defense against eps=0.031 --seed 5 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230331.py run --mode defense --defense_mode patchmixup --beta_alpha 1 --mix_mode maskmixup --sample_mode bernoullisampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 --seed 5 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-cosanne-patchmixup-linear-mix2-batch128-pgd-iter20-seed-5-20230331.log 2>&1
