source ~/.bashrc
source /home/xieyi/anaconda3/bin/activate mmat

# train stylegan2ada for cifar100
#--------------------
python tasklauncher-20220701.py run --save_path /maggie/result --exp_name stylegan2ada-imagenetmixed10 --gen_model stylegan2ada --dataset imagenetmixed10 --data ~/autodl-tmp/maggie/data/imagenetmixed10/imagenetmixed104stylegan2ada/datasets/imagenetmixed10.zip --epochs 1 --batch_size 64 --mode train --train_mode gen-train 

