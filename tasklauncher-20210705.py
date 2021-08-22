"""
Author: maggie
Date:   2021-06-18
Place:  Xidian University
@copyright
"""

import yaml
import torch
import datetime
from utils.runid import GetRunID     
import os
from utils.parseargs import parse_arguments
# import tasks.clatrain
# import tasks.claattack
# import tasks.cladefense

from clamodels.classifier import MaggieClassifier
from datas.dataset import MaggieDataset
from datas.dataset import Array2Dataset
from datas.dataloader import MaggieDataloader
from attacks.advattack import AdvAttack
import numpy as np
from utils.savetxt import SaveTxt
from genmodels.mixgenerate import MixGenerate

if __name__ == '__main__':

    # get arguments dictionary
    args = parse_arguments()
    args_dictionary = vars(args)                                                                                    #   转换args为字典类型的对象
    print('args_dictionary=%s' % args_dictionary)        

    # from command yaml file import configure dictionary
    DO_NOT_EXPORT = ['maggie','xieldy'] 
    if args.subcommand == 'load':                                                                                   #   加载yml文件
        print('args.subcommand=%s,load from the yaml config' % args.subcommand)
        config_dictionary = yaml.load(open(args.config))   

        # normalize string format in the yaml file
        for key in config_dictionary:                                                                               #  规范config_dictionary字典中的数据类型
            if type(config_dictionary[key]) == str:
                if config_dictionary[key] == 'true':
                    config_dictionary[key] = True                                                                   #   转字符串型键值'true'为布尔型键值True
                if config_dictionary[key] == 'false':
                    config_dictionary[key] = False                                                                  #   转字符串型键值'false'为布尔型键值False
                if config_dictionary[key] == 'null':
                    config_dictionary[key] = None

        # remove keys belong to the DO_NOT_EXPORT list from the configure dictionary
        for key in config_dictionary:
            if key not in DO_NOT_EXPORT:                                                                            #   将不在DO_NOT_EXPORT列表中的键值对赋值给args.dictionary
                args_dictionary[key] = config_dictionary[key]
            else:                                                                                                   #   key in DO_NOT_EXPORT
                print(f"Please ignore the keys '{key}' from the yaml file !")
        print('args_dictionary from load yaml file =%s' % args_dictionary)        

    elif args.subcommand == 'run':
        print('args.subcommand=%s,run the command line' % args.subcommand)
    
    elif args.subcommand == None:
        raise Exception('args.subcommand=%s,please input the subcommand !' % args.subcommand)   
    
    else:
        raise Exception('args.subcommand=%s,invalid subcommand,please input again !' % args.subcommand)

    # copy arguments dictionary
    args_dictionary_copy = dict(args_dictionary)                                                                    #   拷贝一份args_dictionary字典，删掉备份args_dictionary中禁忌的键和键值，导出备份的args配置文件
    for key in DO_NOT_EXPORT:
        if key in args_dictionary:
            del args_dictionary_copy[key]
    args_dictionary_copy_yaml = yaml.dump(args_dictionary_copy)                                                     #   yaml.dump()导出yml格式配置文件
    #   print('export config:%s' % args_dictionary_copy_yaml)
    
    #-------------------------prepare dict for stylegan2ada train-------------------------  
    # kwargs used in stylegan2ada training
    setup_training_loop_kwargs_list = [
        'gpus','snap','metrics','seed','data','cond','subset','mirror','cfg','gamma',
        'kimg','batch_size','aug','p','target','augpipe','resume','freezed','fp32','nhwc',
        'allow_tf32','nobench','workers']

    stylegan2ada_config_kwargs = dict()
    for key in setup_training_loop_kwargs_list:
        if key in args_dictionary_copy:
            stylegan2ada_config_kwargs[key] = args_dictionary_copy[key]
    
    # cuda
    if torch.cuda.is_available():
        cuda_use = True
        print('Torch cuda is available')
    else:
        cuda_use = False
        raise Exception('Torch cuda is not available')

    # check args wheather none
    if args.mode == None:                                                                        
        raise Exception('args.mode=%s,invalid mode,please input the mode' % args.mode)
    if args.save_path == None:
        raise Exception('args.save_path=%s,please input the save_path' % args.save_path)
    if args.cla_model == None:
        raise Exception('args.cla_model=%s,please input the classifier model' % args.cla_model)
    if args.gen_model == None:
        raise Exception('args.gen_model=%s,please input the generator model' % args.gen_model)
    if  args.exp_name == None:
        raise Exception('args.exp_name=%s,please input the exp_name' % args.exp_name)
    if args.seed == 0:
        print('args.seed=%i' % args.seed)
        save_path = args.save_path                                                                                  #   save_path=/mmat/result/
    else:
        print('args.seed=%i' % args.seed)
        save_path = f'{args.save_path}/{args.seed}'

    # set dataset parameters
    if args.dataset == 'mnist':
        args.n_classes = 10;    args.img_size = 28;     args.channels = 1  
    elif args.dataset == 'kmnist':
        args.n_classes = 10;    args.img_size = 28;     args.channels = 1        
    elif args.dataset == 'cifar10':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 3  
    elif args.dataset == 'cifar100':
        args.n_classes = 100;   args.img_size = 32;     args.channels = 3  
    elif args.dataset == 'imagenet':
        args.n_classes = 1000;  args.img_size = 1024;   args.channels = 3  
    elif args.dataset == 'imagenet10':
        args.n_classes = 10;    args.img_size = 1024;   args.channels = 3  
    elif args.dataset == 'lsun':
        args.n_classes = 10;    args.img_size = 256;    args.channels = 3  
    elif args.dataset == 'stl10':
        args.n_classes = 10;    args.img_size = 96;     args.channels = 3          #   data预处理时从96填充到128
    elif args.dataset == 'svhn':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 3  

    # set path for saving experiment result
    cur=datetime.datetime.now()
    date = f'{cur.year:04d}{cur.month:02d}{cur.day:02d}'
    print("date:",date)

    if args.mode == 'train':
        exp_result_dir = f'{save_path}/{args.mode}/{args.train_mode}/{args.exp_name}/{date}'
    elif args.mode == 'test':
        exp_result_dir = f'{save_path}/{args.mode}/{args.test_mode}/{args.exp_name}/{date}'
    elif args.mode == 'attack':
        exp_result_dir = f'{save_path}/{args.mode}/{args.attack_mode}/{args.exp_name}/{date}'
    elif args.mode == 'interpolate':
        exp_result_dir = f'{save_path}/{args.mode}/{args.mix_mode}/{args.sample_mode}/{args.exp_name}/{date}'
    elif args.mode == 'defense':     
        if args.defense_mode == "at":
            exp_result_dir = f'{save_path}/{args.mode}/{args.defense_mode}/{args.attack_mode}/{args.exp_name}/{date}'
        elif args.defense_mode == "mmat":
            exp_result_dir = f'{save_path}/{args.mode}/{args.defense_mode}/{args.mix_mode}-{args.sample_mode}/{args.exp_name}/{date}'
    else:
        exp_result_dir=f'{save_path}/{args.mode}/{args.exp_name}/{date}'

    # add run id for exp_result_dir
    cur_run_id = GetRunID(exp_result_dir)
    exp_result_dir = os.path.join(exp_result_dir, f'{cur_run_id:05d}')    

    os.makedirs(exp_result_dir, exist_ok=True)
    print('Experiment result save dir: %s' % exp_result_dir)

    # save arguments dictionary as yaml file
    exp_yaml=open(f'{exp_result_dir}/experiment-command.yaml', "w")    
    exp_yaml.write(args_dictionary_copy_yaml)

#------------------------------------------------------
    if args.mode == 'train':
        if args.train_mode =="gen-train":
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.mixgenerate()

#------------------------------------------------------任务-----------------------------------------------------
    elif args.mode == 'defense':

        #   实例dataset
        cle_dataset = MaggieDataset(args)
        cle_train_dataset = cle_dataset.traindataset()
        cle_test_dataset = cle_dataset.testdataset()
        
        #   实例dataloader
        cle_dataloader = MaggieDataloader(args,cle_train_dataset,cle_test_dataset)
        cle_train_dataloader = cle_dataloader.traindataloader()
        cle_test_dataloader = cle_dataloader.testdataloader()

    #-----------------模型处理---------------------
        #   标准训练
        target_classifier = MaggieClassifier(args)

        #   提取dataset中的data array
        # print(f"Attributes of *{args.dataset}* cle_train_dataset:{cle_train_dataset.__dict__.keys()}")
        cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
        # print("cle_x_train[0][0]:",cle_x_train[0][0])                                       #   GPU Tensor 
        # print("cle_y_train[0]:",cle_y_train[0])                                             #   GPU Tensor 

        # cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

        if args.pretrained_on_imagenet == False:
            target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  #   自定义的训练法
            # target_classifier.artmodel().fit(cle_x_train, cle_y_train, nb_epochs=args.epochs, batch_size=args.batch_size)                             #   art的训练法
        
        cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
        print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                                #   *accuary* on testset:75.6900%
        print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

        # cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)                    #   测试一下evaluate和EvaluateClassifier计算结果是否一致
        # print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                                #   *accuary* on testset:75.6900%
        # print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 


    #-----------------对抗攻击---------------------
        attack_classifier = AdvAttack(args,target_classifier.model())
        x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(cle_train_dataloader,cle_test_dataloader,exp_result_dir)          #     GPU Tensor
        adv_test_accuracy, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
        print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
        print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

    #-----------------对抗训练防御---------------------
        if args.defense_mode == "at":
            target_classifier.adversarialtrain(args, cle_x_train,cle_y_train, x_train_adv,y_train_adv, x_test_adv, y_test_adv, target_classifier.artmodel(),exp_result_dir)
            at_adv_test_accuracy, at_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
            print(f'adversarial trained classifier accuary on adversarial testset:{at_adv_test_accuracy * 100:.4f}%' ) 
            print(f'adversarial trained classifier loss on adversarial testset:{at_adv_test_loss}' )         
            at_cle_test_accuracy, at_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'adversarial trained classifier accuary on clean testset:{at_cle_test_accuracy * 100:.4f}%' ) 
            print(f'adversarial trained classifier loss on clean testset:{at_cle_test_loss}' ) 
            
            SaveTxt(args,exp_result_dir,cle_test_accuracy,adv_test_accuracy,at_adv_test_accuracy,at_cle_test_accuracy, cle_test_loss,adv_test_loss,at_adv_test_loss,at_cle_test_loss)

    #-----------------混合训练防御---------------------
        if args.defense_mode == "mmat":
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.mixgenerate()
            
            #   混合样本生成(Tensor)
            x_train_mix, y_train_mix = generate_model.generatedset()

            target_classifier.mmat(args, cle_x_train,cle_y_train, x_train_mix,y_train_mix, x_test_adv,y_test_adv, target_classifier.artmodel(),exp_result_dir)        
            
            mmat_adv_test_accuracy, mmat_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
            print(f'manifold mixup adversarial trained classifier accuary on adversarial testset:{mmat_adv_test_accuracy * 100:.4f}%' ) 
            print(f'manifold mixup adversarial trained classifier loss on adversarial testset:{mmat_adv_test_loss}' )         
            mmat_cle_test_accuracy, mmat_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'manifold mixup adversarial trained classifier accuary on clean testset:{mmat_cle_test_accuracy * 100:.4f}%' ) 
            print(f'manifold mixup adversarial trained classifier loss on clean testset:{mmat_cle_test_loss}' ) 
            SaveTxt(args,exp_result_dir, cle_test_accuracy,adv_test_accuracy,mmat_adv_test_accuracy,mmat_cle_test_accuracy, cle_test_loss,adv_test_loss,mmat_adv_test_loss,mmat_cle_test_loss)
