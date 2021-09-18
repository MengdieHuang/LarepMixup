""" Author: maggie  Date:   2021-06-18  Place:  Xidian University   @copyright  """

from logging import error
import torch
import utils.parseargs
from clamodels.classifier import MaggieClassifier
from datas.dataset import MaggieDataset
from datas.dataloader import MaggieDataloader
from attacks.advattack import AdvAttack
from utils.savetxt import SaveTxt
from genmodels.mixgenerate import MixGenerate
# import utils.stylegan2ada.dnnlib as dnnlib       
# import utils.stylegan2ada.legacy as legacy
import numpy as np
import os
from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
from robustness.tools.vis_tools import show_image_row

if __name__ == '__main__':
    print("\n")
    print("---------------------------------------")

    if torch.cuda.is_available():
        print('Torch cuda is available')
    else:
        raise Exception('Torch cuda is not available')

    args, exp_result_dir, stylegan2ada_config_kwargs = utils.parseargs.main()
    
    cle_dataset = MaggieDataset(args)
    cle_train_dataset = cle_dataset.traindataset()
    cle_test_dataset = cle_dataset.testdataset()
    
    cle_dataloader = MaggieDataloader(args,cle_train_dataset,cle_test_dataset)
    cle_train_dataloader = cle_dataloader.traindataloader()
    cle_test_dataloader = cle_dataloader.testdataloader()


    if args.mode == 'train':
        #   train stylegan2ada  存到本地            //mmat/result/train/gen-train/...
        if args.train_mode =="gen-train":                                               
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
        #   train resnet-34     存到本地            //mmat/result/train/cla-train/...
        elif args.train_mode =="cla-train":    
            target_classifier = MaggieClassifier(args)
            # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

            if args.pretrained_on_imagenet == False:
                target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  #   自定义的训练法
                # target_classifier.artmodel().fit(cle_x_train, cle_y_train, nb_epochs=args.epochs, batch_size=args.batch_size)                             #   art的训练法
            elif args.pretrained_on_imagenet == True:
                target_classifier = target_classifier

            # print("args.pretrained_on_imagenet:",args.pretrained_on_imagenet)
            # target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  #   自定义的训练法
                
            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                    #   *accuary* on testset:75.6900%
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 
            
            
            # cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)                    #   测试一下evaluate和EvaluateClassifier计算结果是否一致
            # print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                                #   *accuary* on testset:75.6900%
            # print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

    elif args.mode == 'attack':
        if args.cla_network_pkl != None:
            print("local cla_network_pkl:",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            
            attack_classifier = AdvAttack(args,learned_model)
            target_model = attack_classifier.targetmodel()    #   target model是待攻击的目标模型

            x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, cle_test_dataloader,cle_train_dataloader)          #     GPU Tensor
            
            adv_test_accuracy, adv_test_loss = attack_classifier.evaluatefromtensor(target_model,x_test_adv,y_test_adv)
            print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

            accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            accuracy_txt.write(str(txt_content))
        
            loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
            loss_txt.write(str(loss_txt_content))    
                    
        elif args.cla_network_pkl == None:
            target_classifier = MaggieClassifier(args)
            if args.pretrained_on_imagenet == False:
                target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  #   自定义的训练法
            
            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                                #   *accuary* on testset:75.6900%
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 
            
            # target_model = target_classifier.model()
            
            learned_model = target_classifier.model()
            attack_classifier = AdvAttack(args,learned_model)
            target_model  = attack_classifier.targetmodel()     #   attack_classifier.targetmodel()返回的是深拷贝后的learned_model

            x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(cle_train_dataloader,cle_test_dataloader,exp_result_dir)          #     GPU Tensor
            
            adv_test_accuracy, adv_test_loss = attack_classifier.evaluatefromtensor(target_model,x_test_adv,y_test_adv)
            print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

            accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{attack_classifier.getexpresultdir()}/standard-trained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            accuracy_txt.write(str(txt_content))

            loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-clean-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{attack_classifier.getexpresultdir()}/standard-trained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'

    elif args.mode == 'project':        
        if args.gen_network_pkl != None:        
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.projectmain(cle_train_dataloader) 
            # generate_model.projectmain(cle_test_dataloader)     #同时修改stylegan2ada.py的line596
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")

    elif args.mode == 'interpolate':
        if args.gen_network_pkl != None:     
            # if args.projected_dataset != None:    
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.interpolatemain() 
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    

    elif args.mode == 'generate':
        if args.gen_network_pkl != None:     
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)                        
            generate_model.generatemain()

    elif args.mode == 'defense':        

        learned_model = torch.load(args.cla_network_pkl)
        target_classifier = MaggieClassifier(args,learned_model)

        # target_classifier = MaggieClassifier(args)

        cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
        cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

        #------------读取对抗样本---------
        if args.defense_mode == "at":
            adv_trainset_path = os.path.join(args.adv_dataset,'train')
            x_train_adv, y_train_adv = target_classifier.getadvset(adv_trainset_path)

        adv_testset_path = os.path.join(args.adv_dataset,'test')
        x_test_adv, y_test_adv = target_classifier.getadvset(adv_testset_path)
    
        # #-----------测试准确率------------
        adv_test_accuracy, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
        print(f'standard trained classifier *accuary* on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
        # print(f'standard trained classifier *loss* on adversarial testset:{adv_test_loss}' )  

        cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
        print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )               
        # print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

        # cle_test_accuracy = 0.9470
        # cle_test_loss = 0
        # adv_test_accuracy = 0.4020
        # adv_test_loss = 0

    #-----------------对抗训练防御---------------------
        if args.defense_mode == "at":

            target_classifier.adversarialtrain(args, cle_x_train,cle_y_train, x_train_adv, y_train_adv, x_test_adv, y_test_adv, target_classifier.artmodel(),exp_result_dir)
                
            
            at_cle_test_accuracy, at_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'adversarial trained classifier accuary on clean testset:{at_cle_test_accuracy * 100:.4f}%' ) 
            print(f'adversarial trained classifier loss on clean testset:{at_cle_test_loss}' ) 

            at_adv_test_accuracy, at_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
            print(f'adversarial trained classifier accuary on adversarial testset:{at_adv_test_accuracy * 100:.4f}%' ) 
            print(f'adversarial trained classifier loss on adversarial testset:{at_adv_test_loss}' )     

            # SaveTxt(args,exp_result_dir,cle_test_accuracy,adv_test_accuracy,at_adv_test_accuracy,at_cle_test_accuracy, cle_test_loss,adv_test_loss,at_adv_test_loss,at_cle_test_loss)

    #-----------------混合训练防御---------------------
        if args.defense_mode == "mmat":
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.mixgenerate(cle_train_dataloader)
            
            #------------读取混合样本--------- 不要全部读，读出要求的比例即可
            if args.aug_mix_rate != 0:
                x_train_mix, y_train_mix = generate_model.generatedset()
            else:
                x_train_mix = None
                y_train_mix = None
            
            #------------混合训练---------
            # target_classifier.mmat(args, cle_x_train, cle_y_train, x_train_mix, y_train_mix, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir, target_classifier.artmodel())   
            target_classifier.mmat(args, cle_x_train, cle_y_train, x_train_mix, y_train_mix, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir) 
           

           #    生成对抗样本
            mmat_learned_model= target_classifier.model()
            attack_classifier = AdvAttack(args, mmat_learned_model)
            mmat_target_model = attack_classifier.targetmodel()
  
            # mmat_x_test_adv, mmat_y_test_adv = attack_classifier.generate(exp_result_dir,cle_test_dataloader)          #     只生成testset对抗样本
            mmat_x_test_adv, mmat_y_test_adv = attack_classifier.generateadvfromtestsettensor(exp_result_dir, cle_x_test, cle_y_test)

            mmat_adv_test_accuracy, mmat_adv_test_loss = target_classifier.evaluatefromtensor(mmat_target_model,mmat_x_test_adv,mmat_y_test_adv)
            print(f'mmat trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'mmat trained classifier loss on adversarial testset:{adv_test_loss}' )    
            mmat_cle_test_accuracy, mmat_cle_test_loss = target_classifier.evaluatefromtensor(mmat_target_model,cle_x_test,cle_y_test)
            print(f'mmat trained classifier accuary on clean testset:{mmat_cle_test_accuracy * 100:.4f}%' ) 
            print(f'mmat trained classifier loss on clean testset:{mmat_cle_test_loss}' ) 
            SaveTxt(args, exp_result_dir, cle_test_accuracy, adv_test_accuracy, mmat_adv_test_accuracy, mmat_cle_test_accuracy, cle_test_loss, adv_test_loss, mmat_adv_test_loss, mmat_cle_test_loss)
    
    
    print("---------------------------------------")
    print("\n")