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
import utils.stylegan2ada.dnnlib as dnnlib       
import utils.stylegan2ada.legacy as legacy
import os
import numpy as np
from attacks.perattack import PerAttack



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

    print("cle_train_dataloader.len",len(cle_train_dataloader))
    print("cle_test_dataloader.len",len(cle_test_dataloader))

    """
    cle_train_dataloader.len 2414
    cle_test_dataloader.len 94
    """
    # if args.mode == 'train':
    #     if args.train_mode =="gen-train":                                               
    #         generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
    #     elif args.train_mode =="cla-train":    
    #         target_classifier = MaggieClassifier(args)
    #         # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
    #         # cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

    #         if args.pretrained_on_imagenet == False:
    #             target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')     
                                             
    #         elif args.pretrained_on_imagenet == True:
    #             target_classifier = target_classifier

    #         cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
    #         print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                    #   *accuary* on testset:75.6900%
    #         print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

    if args.mode == 'train':
        if args.train_mode =="gen-train":                                               
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
        elif args.train_mode =="cla-train":    
            target_classifier = MaggieClassifier(args)
            # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

            if args.pretrained_on_imagenet == False:
                # target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')     
                
                target_classifier.newtrain(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                   
                
                                             
            elif args.pretrained_on_imagenet == True:
                target_classifier = target_classifier

            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )
            #   *accuary* on testset:75.6900%
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 
            
    elif args.mode == 'attack':
        if args.latentattack == False:    
            if args.perceptualattack == False:  #   像素层对抗攻击
                
                if args.attack_mode =='cw':     
                    print("confidence:",args.confidence)
                else:
                    print("eps:",args.attack_eps)                   #   0.05

                print("pixel adversarial attack.............")
                print("cla_network_pkl:",args.cla_network_pkl)
                """
                cla_network_pkl: /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl
                """

                learned_model = torch.load(args.cla_network_pkl)
                attack_classifier = AdvAttack(args,learned_model)                
                target_model = attack_classifier.targetmodel()    #   target model是待攻击的目标模型

                #-------------maggie 20230325
                cle_test_accuracy, cle_test_loss = attack_classifier.evaluatefromdataloader(target_model,cle_test_dataloader)
                print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )
                print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 
                #----------------------------      
                                
                print("start generating adv 20230325")
                # x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, cle_test_dataloader,cle_train_dataloader)          
                x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, cle_test_dataloader)          
                # x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, test_dataloader=cle_train_dataloader)     #为了保存方便    

                adv_test_accuracy, adv_test_loss = attack_classifier.evaluatefromtensor(target_model,x_test_adv,y_test_adv)
                print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
                print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

                accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
                txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
                accuracy_txt.write(str(txt_content))
            
                loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
                loss_txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
                loss_txt.write(str(loss_txt_content))    
            
            elif args.perceptualattack == True:  #   像素层感知攻击
                learned_model = torch.load(args.cla_network_pkl)
                attack_classifier = PerAttack(args,learned_model)
                target_model = attack_classifier.targetmodel()    #   target model是待攻击的目标模型

                x_test_per, y_test_per = attack_classifier.generate(exp_result_dir, cle_test_dataloader)          

                target_model.eval()
                per_test_accuracy, per_test_loss = attack_classifier.evaluatefromtensor(target_model, x_test_per, y_test_per)
                print(f'standard trained classifier accuary on perceptual attack testset:{per_test_accuracy * 100:.4f}%' ) 
                print(f'standard trained classifier loss on perceptual attack testset:{per_test_loss}' )    

                accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
                txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-per-{args.dataset}-testset = {per_test_accuracy}\n'
                accuracy_txt.write(str(txt_content))
            
                loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
                loss_txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-per-{args.dataset}-testset = {per_test_loss}\n'
                loss_txt.write(str(loss_txt_content))    
        
        elif args.latentattack == True:  #   表征层对抗攻击
            print("eps:",args.attack_eps)
            print("latent adversarial attack.............")
            print("cla_network_pkl:",args.cla_network_pkl)
            learned_cla_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_cla_model)
            cle_w_test, cle_y_test = target_classifier.getproset(args.projected_dataset)
            cle_y_test = cle_y_test[:,0]    #这个不能注释
            print("cle_w_test.shape:",cle_w_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            # raise error("maggie stop")


            cla_net = learned_cla_model
            cla_net.cuda()
            cla_net.eval()                

            device = torch.device('cuda')
            with dnnlib.util.open_url(args.gen_network_pkl) as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                                                
            
            gan_net = G.synthesis     
            gan_net.cuda()
            gan_net.eval()

            merge_model = torch.nn.Sequential(gan_net, cla_net)
            latent_attacker = AdvAttack(args,merge_model)
            
            adv_x_test, adv_y_test = latent_attacker.generatelatentadv(exp_result_dir, cle_test_dataloader, cle_w_test, cle_y_test, gan_net)             
            adv_test_accuracy, adv_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
            print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

            #   20220809
            # gan_net.eval()
            # testset_total_num = int(cle_w_test.size(0))
            # batch_size = args.batch_size
            # batch_num = int(np.ceil( int(testset_total_num) / float(batch_size) ) )
            # cle_x_test = []
            # for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
            #     cle_w_batch = cle_w_test[batch_index * batch_size : (batch_index + 1) * batch_size]
            #     cle_x_batch = gan_net(cle_w_batch.cuda())        
            #     cle_x_test.append(cle_x_batch)

            # cle_x_test = torch.cat(cle_x_test, dim=0)        
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)
            # cle_test_accuracy, cle_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),cle_x_test, cle_y_test)
            # print(f'standard trained classifier accuary on clean w testset:{cle_test_accuracy * 100:.4f}%' ) 
            # print(f'standard trained classifier loss on clean w testset:{cle_test_loss}' )    

            # accuracy_txt=open(f'{latent_attacker.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            # txt_content = f'{latent_attacker.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            # accuracy_txt.write(str(txt_content))
        
            # loss_txt=open(f'{latent_attacker.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
            # loss_txt_content = f'{latent_attacker.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
            # loss_txt.write(str(loss_txt_content))  

        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    
                    
    elif args.mode == 'project':        
        if args.gen_network_pkl != None:        
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            # print("projecting training set")                        #   20220714 投影CIFAR10训练集
            # generate_model.projectmain(cle_train_dataloader) 

            print("projecting test set-----为了避免在跑代码的中断，文件夹仍然名为trainset")
            generate_model.projectmain(cle_test_dataloader)     #20220624 投影CIFAR10测试集

        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")

    elif args.mode == 'interpolate':
        if args.gen_network_pkl != None:     
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.interpolatemain() 
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    

    elif args.mode == 'generate':
        if args.gen_network_pkl != None:     
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)                        
            generate_model.generatemain()
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    

    elif args.mode == 'defense':        
        if args.defense_mode == "rmt":
            print("adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            # model
            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本投影训练集
            print("args.projected_dataset",args.projected_dataset)
            cle_w_train, cle_y_train = target_classifier.getproset(args.projected_dataset)
            print("cle_w_train.shape:",cle_w_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)

            # 干净样本测试集
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   bug
            print(f'Accuary of before rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before rmt trained classifier clean testset:{cle_test_loss}' ) 

            if args.cla_model in ['preactresnet18','preactresnet34','preactresnet50'] and args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对PreAct的非FGSM的攻击 只评估精度 
                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_mode == "om-pgd":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对非PreAct的OM-PGD的攻击 只评估精度 
                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_eps != 0.02:    
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对非PreAct的eps不等于0.02的攻击 只评估精度 

                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_mode in ["fog","snow","elastic","jpeg"]:
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对非PreAct的perceptual攻击 只评估精度 
                raise error            

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)
            
            # train
            target_classifier.rmt(args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs)

            #------------test after rmt------------
            # evaluate acc and loss on clean pixel testset 
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of rmt trained classifier on clean testset:{cle_test_loss}' ) 
           
            # evaluate acc and loss on adversarial pixel testset 
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of rmt trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of rmt trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of rmt trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of rmt trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='at':
            print("adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            # model
            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            # 对抗样本训练集
            # print("args.train_adv_dataset",args.train_adv_dataset)
            adv_trainset_path = args.train_adv_dataset

            adv_x_train, adv_y_train = target_classifier.getadvset(adv_trainset_path)
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)            
            # adv_x_train=adv_x_train[:25397]
            # adv_y_train=adv_y_train[:25397]

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset

            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            print("adv_x_test.shape:",adv_x_test.shape)
            print("adv_y_test.shape:",adv_y_test.shape)  

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   bug
            print(f'Accuary of before adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before adversarial trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before at trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before at trained classifier on adv testset:{adv_test_loss}' ) 
                raise error
            
            target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)

            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of adversarial trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of adversarial trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of adversarial trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of adversarial trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='inputmixup':
            # model
            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # clean pixel testset
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()
            print("cle_y_train.shape:",cle_y_train.shape)
            
            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   bug
            print(f'Accuary of before inputmixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before inputmixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before inputmixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before inputmixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.inputmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of inputmixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of inputmixup trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of inputmixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of inputmixup trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of inputmixup trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of inputmixup trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='manifoldmixup':
            print("manifold mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            # model
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)


            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   在干净样本测试集上评估精度
            print(f'Accuary of before manifold mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before manifold mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)     #   在对抗样本测试集上评估精度
                print(f'Accuary of before manifold mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before manifold mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.manifoldmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of manifold trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of manifold trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of manifold trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of manifold trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of manifold trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of manifold trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='patchmixup':
            print("patch mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            # model
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   在干净样本测试集上评估精度
            print(f'Accuary of before patch mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before patch mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)     #   在对抗样本测试集上评估精度
                print(f'Accuary of before patch mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before patch mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.patchmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of patch trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of patch trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of patch trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of patch trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of patch trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of patch trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='puzzlemixup':
            print("puzzle mixup")
            print("lr:",args.lr)
            print("args.attack_mode:",args.attack_mode)
            print("cla_network_pkl:",args.cla_network_pkl)

            # model
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   在干净样本测试集上评估精度
            print(f'Accuary of before puzzle mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before puzzle mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)     #   在对抗样本测试集上评估精度
                print(f'Accuary of before puzzle mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before puzzle mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.puzzlemixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of puzzle trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of puzzle trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of puzzle trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of puzzle trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of puzzle trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of puzzle trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='cutmixup':
            print("cut mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            # model
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   在干净样本测试集上评估精度
            print(f'Accuary of before cut mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before cut mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)     #   在对抗样本测试集上评估精度
                print(f'Accuary of before cut mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before cut mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.cutmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of cut mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of cut mixup trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of cut mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of cut mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of cut mixup trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of cut mixup trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='dmat':
            print("dual manifold adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            # model
            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # 干净样本测试集
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            # 对抗样本训练集1 ompgd/omfgsm
            print("args.train_adv_dataset",args.train_adv_dataset)
            adv_trainset_path = args.train_adv_dataset
            adv_x_train, adv_y_train = target_classifier.getadvset(adv_trainset_path)
            adv_x_train=adv_x_train[:25000]
            adv_y_train=adv_y_train[:25000]
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)            

            #-----------20220809------------
            # 对抗样本训练集2 pgd/fgsm
            print("args.train_adv_dataset_2",args.train_adv_dataset_2)
            adv_trainset_path_2 = args.train_adv_dataset_2
            adv_x_train_2, adv_y_train_2 = target_classifier.getadvset(adv_trainset_path_2)
            adv_x_train_2=adv_x_train_2[:25000]
            adv_y_train_2=adv_y_train_2[:25000]            
            print("adv_x_train_2.shape:",adv_x_train_2.shape)
            print("adv_y_train_2.shape:",adv_y_train_2.shape)  
              
            adv_x_train = torch.cat([adv_x_train, adv_x_train_2], dim=0)
            adv_y_train = torch.cat([adv_y_train, adv_y_train_2], dim=0)  
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)  
            #-------------------------------

            # 对抗样本测试集
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset

            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            print("adv_x_test.shape:",adv_x_test.shape)
            print("adv_y_test.shape:",adv_y_test.shape)  

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   bug
            print(f'Accuary of before dual manifold adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before dual manifold adversarial trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_loss}' )           
                raise error("maggie stop here")
            
            target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)

            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of dual manifold adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of dual manifold adversarial trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
            if args.whitebox == True:
                # white box adversarial pixel testset acc and loss
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of dual manifold adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of dual manifold adversarial trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                # black box adversarial pixel testset acc and loss
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of dual manifold adversarial trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of dual manifold adversarial trained classifier on black-box adv testset:{adv_test_loss}' ) 
        
    print("---------------------------------------")
    print("\n")