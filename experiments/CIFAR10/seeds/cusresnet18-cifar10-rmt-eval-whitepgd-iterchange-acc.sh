source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# XXX-vanilla-cla-XXX
# XXX-rmt-seed1-cla-XXX

# eval vanilla cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter 7)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00005-eps031-step0078-iter7-acc21.53/attack-cifar10-dataset/samples/test --batch_size 128 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1

# eval vanilla cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter 10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00002-eps031-step0078-iter10-acc15.58/attack-cifar10-dataset/samples/test --batch_size 128 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1

# eval vanilla cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter 20)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/cusresnet18-cifar10/20230326/00007-testacc-0.9641/train-cifar10-dataset/standard-trained-classifier-cusresnet18-on-clean-cifar10-finished.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00000-eps031-step0078-iter20-acc10.1/attack-cifar10-dataset/samples/test --batch_size 128 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1


#-------------------------------------

# eval rmt-seed1 cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 7 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/cusresnet18-cifar10/blackbox/20230401/00000-cleacc-96.29/rmt-cifar10-dataset/rmt-trained-cusresnet18-on-cifar10-epoch-0040-cleacc-0.9629-advacc-0.2591.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00005-eps031-step0078-iter7-acc21.53/attack-cifar10-dataset/samples/test --batch_size 128 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1

# eval rmt-seed1 cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 10 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/cusresnet18-cifar10/blackbox/20230401/00000-cleacc-96.29/rmt-cifar10-dataset/rmt-trained-cusresnet18-on-cifar10-epoch-0040-cleacc-0.9629-advacc-0.2591.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00002-eps031-step0078-iter10-acc15.58/attack-cifar10-dataset/samples/test --batch_size 128 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1

# eval rmt-seed1 cusresnet18 on cifar10: pgd (eps0.031-step0.0078-iter)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name cusresnet18-cifar10 --attack_mode pgd --attack_eps 0.031 --attack_eps_step 0.0078 --attack_max_iter 20 --whitebox --cla_model cusresnet18 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/rmt/pgd/basemixup-betasampler/cusresnet18-cifar10/blackbox/20230401/00000-cleacc-96.29/rmt-cifar10-dataset/rmt-trained-cusresnet18-on-cifar10-epoch-0040-cleacc-0.9629-advacc-0.2591.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/pgd/cusresnet18-cifar10/20230328/00000-eps031-step0078-iter20-acc10.1/attack-cifar10-dataset/samples/test --batch_size 128 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/rmt-cusresnet18-cifar10-against-whitepgd-eps0.031-step0.0078-20230331.log 2>&1

