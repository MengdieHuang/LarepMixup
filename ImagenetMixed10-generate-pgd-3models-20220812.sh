source ~/.bashrc
source /root/miniconda3/bin/activate mmat

# imagenetmixed10 pgd generate 20220812
#----------------------preactresnet18---------------------------------
# python tasklauncher-20220812.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --attack_eps 0.05 --batch_size 16 >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/attack-example-generate/pgd/imagenetmixed10-preactresnet18-pgd-generate-20220812.log 2>&1

python tasklauncher-20220812.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet18-imagenetmixed10 --cla_model preactresnet18 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl --dataset imagenetmixed10 --attack_eps 0.05 --batch_size 16 
# >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/attack-example-generate/pgd/imagenetmixed10-preactresnet18-pgd-generate-20220812.log 2>&1


# #----------------------preactresnet34---------------------------------
python tasklauncher-20220812.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet34-imagenetmixed10 --cla_model preactresnet34 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet34-imagenetmixed10/standard-trained-classifier-preactresnet34-on-clean-imagenetmixed10-epoch-0022-acc-88.03.pkl --dataset imagenetmixed10 --attack_eps 0.05 --batch_size 16 
# >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/attack-example-generate/pgd/imagenetmixed10-preactresnet34-pgd-generate-20220812.log 2>&1

# #----------------------preactresnet50---------------------------------
python tasklauncher-20220812.py run --save_path /root/autodl-tmp/maggie/result --mode attack --attack_mode pgd --whitebox --exp_name preactresnet50-imagenetmixed10 --cla_model preactresnet50 --cla_network_pkl /root/autodl-tmp/maggie/result/train/cla-train/preactresnet50-imagenetmixed10/standard-trained-classifier-preactresnet50-on-clean-imagenetmixed10-epoch-0022-acc-86.80.pkl --dataset imagenetmixed10 --attack_eps 0.05 --batch_size 16 
# >> /root/autodl-nas/maggie/mmat/log/ImagenetMixed10/attack-example-generate/pgd/imagenetmixed10-preactresnet50-pgd-generate-20220812.log 2>&1