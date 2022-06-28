source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# train stylegan2ada for cifar100
#--------------------
python tasklauncher-20220628.py run --exp_name stylegan2ada-cifar100 --gen_model stylegan2ada --dataset cifar100 --data /root/autodl-tmp/data/cifar100/cifar1004stylegan2ada/cifar100.zip --pretrain_pkl_path /root/autodl-tmp/network/gen-train/stylegan2ada-cifar100/network-snapshot-025000.pkl --epochs 1 --batch_size 64 --mode train --train_mode gen-train 

