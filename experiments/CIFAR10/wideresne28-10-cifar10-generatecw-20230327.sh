source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# cifar10 cw wideresnet28_10-acc96.70 eps=8/255=0.031  step size=2/255=0.0078  confidence 0 max_iter 10 
CUDA_VISIBLE_DEVICES=1 python -u tasklauncher-20230327.py run --mode attack --attack_mode cw --whitebox --exp_name wideresnet28_10-cifar10 --cla_model wideresnet28_10 --cla_network_pkl /home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pkl --dataset cifar10 --confidence 0 --attack_max_iter 10 >> /home/maggie/mmat/log/CIFAR10/attack-example-generate/wideresnet28_10-acc96.70-generate-cw-conf0.00-max_iter10-20230327.log 2>&1