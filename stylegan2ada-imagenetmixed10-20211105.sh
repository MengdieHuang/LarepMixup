source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

CUDA_VISIBLE_DEVICES=1 python tasklauncher-20211113.py run --exp_name stylegan2ada-imagenetmixed10 --gen_model stylegan2ada --dataset imagenetmixed10 --data /home/data/maggie/imagenetmixed10/imagenetmixed104stylegan2ada/datasets/imagenetmixed10.zip --epochs 1 --batch_size 32 --mode train --train_mode gen-train --pretrain_pkl_path /home/maggie/mmat/result/train/gen-train/stylegan2ada-imagenetmixed10/20210920/00000/imagenetmixed10-auto1-batch32-ada-bgc-noresume/network-snapshot-000880.pkl