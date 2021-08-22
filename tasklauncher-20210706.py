""" Author: maggie  Date:   2021-06-18  Place:  Xidian University   @copyright  """

import yaml
import torch
import os
from utils.parseargs import parse_arguments
from utils.parseargs import reset_arguments
from utils.parseargs import check_arguments
from utils.parseargs import set_exp_result_dir
from utils.parseargs import correct_args_dictionary
from utils.parseargs import get_stylegan2ada_args
from utils.parseargs import copy_args_dictionary

from clamodels.classifier import MaggieClassifier
from datas.dataset import MaggieDataset
from datas.dataloader import MaggieDataloader
from attacks.advattack import AdvAttack
from utils.savetxt import SaveTxt
from genmodels.mixgenerate import MixGenerate

if __name__ == '__main__':

    # cuda
    if torch.cuda.is_available():
        # cuda_use = True
        print('Torch cuda is available')
    else:
        # cuda_use = False
        raise Exception('Torch cuda is not available')

    # get arguments dictionary
    args = parse_arguments()
    args_dictionary = vars(args)                                                                                    #   转换args为字典类型的对象
    print('args_dictionary=%s' % args_dictionary)        

    # from command yaml file import configure dictionary
    DO_NOT_EXPORT = ['maggie','xieldy'] 
    args_dictionary = correct_args_dictionary(args,args_dictionary,DO_NOT_EXPORT)

    # copy arguments dictionary
    args_dictionary_copy = copy_args_dictionary(args_dictionary,DO_NOT_EXPORT)
    args_dictionary_copy_yaml = yaml.dump(args_dictionary_copy)                                                     #   yaml.dump()导出yml格式配置文件
    
    # kwargs used in stylegan2ada training    
    stylegan2ada_config_kwargs = get_stylegan2ada_args(args_dictionary_copy)

    check_arguments(args)
    args = reset_arguments(args)

    exp_result_dir = set_exp_result_dir(args)
    os.makedirs(exp_result_dir, exist_ok=True)
    print('Experiment result save dir: %s' % exp_result_dir)

    # save arguments dictionary as yaml file
    exp_yaml=open(f'{exp_result_dir}/experiment-command.yaml', "w")    
    exp_yaml.write(args_dictionary_copy_yaml)

    #-------------Train Generator Network------------
    if args.mode == 'train':

        if args.train_mode =="gen-train":
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.mixgenerate()

    #-------------MM Train Classifier Network------------

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
        target_classifier = MaggieClassifier(args)

        cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
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
            # generate_model.mixgenerate(cle_x_train,cle_y_train)
            generate_model.mixgenerate(cle_train_dataloader)
            
            x_train_mix, y_train_mix = generate_model.generatedset()
            target_classifier.mmat(args, cle_x_train,cle_y_train, x_train_mix,y_train_mix, x_test_adv,y_test_adv, target_classifier.artmodel(),exp_result_dir)        
            mmat_adv_test_accuracy, mmat_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
            print(f'manifold mixup adversarial trained classifier accuary on adversarial testset:{mmat_adv_test_accuracy * 100:.4f}%' ) 
            print(f'manifold mixup adversarial trained classifier loss on adversarial testset:{mmat_adv_test_loss}' )         
            mmat_cle_test_accuracy, mmat_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'manifold mixup adversarial trained classifier accuary on clean testset:{mmat_cle_test_accuracy * 100:.4f}%' ) 
            print(f'manifold mixup adversarial trained classifier loss on clean testset:{mmat_cle_test_loss}' ) 
            SaveTxt(args,exp_result_dir, cle_test_accuracy,adv_test_accuracy,mmat_adv_test_accuracy,mmat_cle_test_accuracy, cle_test_loss,adv_test_loss,mmat_adv_test_loss,mmat_cle_test_loss)
