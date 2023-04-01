source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# # eval vanilla wideresnet28_10 on cifar10: deepfool (eps031-iter10)
# CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 0 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

# eval dmat-seed1 wideresnet28_10 on cifar10: deepfool (eps031-iter10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-1/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.81/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 1 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

# eval dmat-seed2 wideresnet28_10 on cifar10: deepfool (eps031-iter10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-2/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.84/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 2 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

# eval dmat-seed3 wideresnet28_10 on cifar10: deepfool (eps031-iter10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-3/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230329/00001-cleacc-96.95/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 3 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

# eval dmat-seed4 wideresnet28_10 on cifar10: deepfool (eps031-iter10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-4/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230330/00001-cleacc-96.93/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 4 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

# eval dmat-seed5 wideresnet28_10 on cifar10: deepfool (eps031-iter10)
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230331.py run --mode eval --exp_name wideresnet28_10-cifar10 --attack_mode deepfool --attack_eps 0.031 --attack_max_iter 10 --whitebox --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/seed-5/defense/dmat/pgd/wideresnet28_10-cifar10/blackbox/20230330/00001-cleacc-96.84/adversarial-trained-classifier-wideresnet28_10-on-cifar10-epoch-0040.pkl --dataset cifar10 --test_adv_dataset /home/data/maggie/result-newhome/attack/deepfool/wideresnet28_10-cifar10/20230327/00000-eps031-iter10-acc2.63/attack-cifar10-dataset/samples/test --batch_size 128 --seed 5 >> /home/maggie/mmat/log/CIFAR10/eval/montecarloexp/dmat-wideresnet28_10-cifar10-against-whitedeepfool-eps0.031-iter10-20230331.log 2>&1

