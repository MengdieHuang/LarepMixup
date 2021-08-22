""" Author: maggie  Date:   2021-06-18  Place:  Xidian University   @copyright  """

import torch
import utils.parseargs
from clamodels.classifier import MaggieClassifier
from datas.dataset import MaggieDataset
from datas.dataloader import MaggieDataloader
from attacks.advattack import AdvAttack
from utils.savetxt import SaveTxt
from genmodels.mixgenerate import MixGenerate

if __name__ == '__main__':

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
    if args.mode == 'train':
        if args.train_mode =="cla-train":    
            target_classifier = MaggieClassifier(args)

            # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
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

    #   Generate Adversarial Examples   存到本地    //mmat/result/attack/...
    elif args.mode == 'attack':

        if args.cla_network_pkl == None:
            target_classifier = MaggieClassifier(args)

            # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
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
        elif args.cla_network_pkl != None:
            #   加载网络 
            target_classifier = 
        attack_classifier = AdvAttack(args,target_classifier.model())
        x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(cle_train_dataloader,cle_test_dataloader,exp_result_dir)          #     GPU Tensor

    # #   Project latent representation       存到本地    //mmat/result/project/...
    # elif args.mode == 'project':        

    #     generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
    #     # generate_model.mixgenerate(cle_x_train,cle_y_train)
    #     generate_model.mixgenerate(cle_train_dataloader)


    # elif args.mode == 'defense':        #   //mmat/result/defense

    #     attack_classifier = AdvAttack(args,target_classifier.model())
    #     x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(cle_train_dataloader,cle_test_dataloader,exp_result_dir)          #     GPU Tensor
    #     adv_test_accuracy, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
    #     print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
    #     print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

    # #-----------------对抗训练防御---------------------
    #     if args.defense_mode == "at":
    #         target_classifier.adversarialtrain(args, cle_x_train,cle_y_train, x_train_adv,y_train_adv, x_test_adv, y_test_adv, target_classifier.artmodel(),exp_result_dir)
    #         at_adv_test_accuracy, at_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
    #         print(f'adversarial trained classifier accuary on adversarial testset:{at_adv_test_accuracy * 100:.4f}%' ) 
    #         print(f'adversarial trained classifier loss on adversarial testset:{at_adv_test_loss}' )         
    #         at_cle_test_accuracy, at_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
    #         print(f'adversarial trained classifier accuary on clean testset:{at_cle_test_accuracy * 100:.4f}%' ) 
    #         print(f'adversarial trained classifier loss on clean testset:{at_cle_test_loss}' ) 
    #         SaveTxt(args,exp_result_dir,cle_test_accuracy,adv_test_accuracy,at_adv_test_accuracy,at_cle_test_accuracy, cle_test_loss,adv_test_loss,at_adv_test_loss,at_cle_test_loss)

    # #-----------------混合训练防御---------------------
    #     if args.defense_mode == "mmat":
    #         generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
    #         # generate_model.mixgenerate(cle_x_train,cle_y_train)
    #         generate_model.mixgenerate(cle_train_dataloader)
            
    #         x_train_mix, y_train_mix = generate_model.generatedset()
    #         target_classifier.mmat(args, cle_x_train,cle_y_train, x_train_mix,y_train_mix, x_test_adv,y_test_adv, target_classifier.artmodel(),exp_result_dir)        
    #         mmat_adv_test_accuracy, mmat_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
    #         print(f'manifold mixup adversarial trained classifier accuary on adversarial testset:{mmat_adv_test_accuracy * 100:.4f}%' ) 
    #         print(f'manifold mixup adversarial trained classifier loss on adversarial testset:{mmat_adv_test_loss}' )         
    #         mmat_cle_test_accuracy, mmat_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
    #         print(f'manifold mixup adversarial trained classifier accuary on clean testset:{mmat_cle_test_accuracy * 100:.4f}%' ) 
    #         print(f'manifold mixup adversarial trained classifier loss on clean testset:{mmat_cle_test_loss}' ) 
    #         SaveTxt(args,exp_result_dir, cle_test_accuracy,adv_test_accuracy,mmat_adv_test_accuracy,mmat_cle_test_accuracy, cle_test_loss,adv_test_loss,mmat_adv_test_loss,mmat_cle_test_loss)