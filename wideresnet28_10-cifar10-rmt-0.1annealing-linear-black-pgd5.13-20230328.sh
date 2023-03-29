source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# mix50000 cat cle50000
# # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 1 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230328.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --seed 1 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-rmt-linear-mix2-batch128-pgd-iter20-seed-1-20230328.log 2>&1 

# # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 2 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230328.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --seed 2 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-rmt-linear-mix2-batch128-pgd-iter20-seed-2-20230328.log 2>&1

# # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 3 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230328.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --seed 3 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-rmt-linear-mix2-batch128-pgd-iter20-seed-3-20230328.log 2>&1

# # # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 4 
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230328.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --seed 4 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-rmt-linear-mix2-batch128-pgd-iter20-seed-4-20230328.log 2>&1

# # # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 5 
# CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230328.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --seed 5 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-rmt-linear-mix2-batch128-pgd-iter20-seed-5-20230328.log 2>&1


# # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 1 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230329.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 --seed 1 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-cosanne-rmt-linear-mix2-batch128-pgd-iter20-seed-1-20230329.log 2>&1 

# # wideresnet28_10 linear rmt beta(1,1) basemixup defense against eps=0.031 --seed 2 
CUDA_VISIBLE_DEVICES=0 python -u tasklauncher-20230329.py run --mode defense --defense_mode rmt --beta_alpha 1 --mix_mode basemixup --sample_mode betasampler --mix_w_num 2 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --blackbox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_model stylegan2ada --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --dataset cifar10 --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 128 --epochs 40 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 --seed 2 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-cosanne-rmt-linear-mix2-batch128-pgd-iter20-seed-2-20230329.log 2>&1