source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat


# # wideresnet28_10 linear dmat beta(1,1) basemixup defense against eps=0.031 --seed 0 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230401.py run --mode defense --defense_mode dmat --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --gen_network_pkl /home/data/maggie/result-newhome/train/gen-train/stylegan2ada-cifar10/20210702/00000/cifar10-auto1-batch64-ada-bgc-noresume/network-snapshot-023063.pkl --projected_trainset /home/data/maggie/result-newhome/project/stylegan2ada-cifar10/20220716/00000/project-cifar10-trainset --dataset cifar10 --train_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00031-om-eps031-step0078-iter7-acc49.51/attack-cifar10-dataset/latent-attack-samples/train --train_adv_dataset_2 /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00024-eps031-step0078-iter7-acc12.34/attack-cifar10-dataset/samples/train --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/wideresnet28_10-cifar10/20230327/00028-eps31-step0078-iter20-acc5.13/attack-cifar10-dataset/samples/test --batch_size 64 --epochs 40 --lr 0.01 --optimizer sgd --lr_schedule CosineAnnealingLR --patience 40 --seed 0 >> /home/maggie/mmat/log/CIFAR10/defense/montecarloexp/ler-wideresnet28_10-cifar10-cosanne-dmat-ompgditer7-pgditer7-batch64-pgd-seed-0-20230401.log 2>&1




