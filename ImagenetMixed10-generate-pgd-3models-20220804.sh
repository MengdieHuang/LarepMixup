source ~/.bashrc
# source /home/xieyi/anaconda3/bin/activate mmat
source /root/miniconda3/bin/activate mmat

# imagenetmixed10 pgd generate 20220804
#----------------------preactresnet18---------------------------------
# python tasklauncher-20220804.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --attack_eps 0.02 --batch_size 32 --cpus 12 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/attack-example-generate/pgd/imagenetmixed10-preactresnet18-pgd-generate-20220804.log 2>&1

python tasklauncher-20220807.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --attack_eps 0.02 --batch_size 8 
