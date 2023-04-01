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
import time


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

    print("args.batch_size:",args.batch_size)
    print("batchnum=cle_train_dataloader.len",len(cle_train_dataloader))
    print("batchnum=cle_test_dataloader.len",len(cle_test_dataloader))

    if args.mode == 'train':
        if args.train_mode =="gen-train":                                               
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
        elif args.train_mode =="cla-train":    
            target_classifier = MaggieClassifier(args)

            if args.pretrained_on_imagenet == False:
                # target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')     
                
                target_classifier.newtrain(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                   
                                            
            elif args.pretrained_on_imagenet == True:
                target_classifier = target_classifier

            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 
          
    elif args.mode == 'attack':
        if args.latentattack == False:    
            print("cla_network_pkl:",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            
            if args.perceptualattack == False:  #   像素层对抗攻击
                print("-----------------generate pixel-level adversarial examples----------------")
                if args.attack_mode =='cw':     
                    print("confidence:",args.confidence)
                else:
                    print("eps:",args.attack_eps)                   #   0.05

                attack_classifier = AdvAttack(args,learned_model)                
                target_model = attack_classifier.targetmodel()    #   target model是待攻击的目标模型

                cle_test_accuracy, cle_test_loss = attack_classifier.evaluatefromdataloader(target_model,cle_test_dataloader)
                print(f'Accuary of load {args.cla_model} classifier on clean testset:{cle_test_accuracy * 100:.4f}%' ) 
                print(f'Loss of load {args.cla_model} classifier on clean testset:{cle_test_loss}' ) 
                                                    
                print("-----------------start generate pixel-level adversarial examples----------------")       
                print("args.saveadvtrain:",args.saveadvtrain)
                if args.saveadvtrain == False:
                    # 默认只生成test adv
                    x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, cle_test_dataloader)         
                elif args.saveadvtrain == True:
                    x_train_adv, y_train_adv, x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, cle_test_dataloader,cle_train_dataloader)          
                     
            elif args.perceptualattack == True:  #   像素层感知攻击
                print("-----------------generate pixel-level perceptual examples----------------")                
                attack_classifier = PerAttack(args,learned_model)
                target_model = attack_classifier.targetmodel()    #   target model是待攻击的目标模型

                cle_test_accuracy, cle_test_loss = attack_classifier.evaluatefromdataloader(target_model,cle_test_dataloader)
                print(f'Accuary of load {args.cla_model} classifier on clean testset:{cle_test_accuracy * 100:.4f}%' ) 
                print(f'Loss of load {args.cla_model} classifier on clean testset:{cle_test_loss}' ) 

                print("-----------------start generate pixel-level perceptual examples----------------")
                x_test_per, y_test_per = attack_classifier.generate(exp_result_dir, cle_test_dataloader)          

            target_model.eval()
            adv_test_accuracy, adv_test_loss = attack_classifier.evaluatefromtensor(target_model,x_test_adv,y_test_adv)  
            print(f'Accuary of load {args.cla_model} classifier on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'Loss of load {args.cla_model} classifier on adversarial testset:{adv_test_loss}' ) 

            accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/load-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset.txt', "w")    
            txt_content = f'{attack_classifier.getexpresultdir()}/load-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            accuracy_txt.write(str(txt_content))
        
            loss_txt=open(f'{attack_classifier.getexpresultdir()}/load-{args.cla_model}-loss-on-adv-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{attack_classifier.getexpresultdir()}/load-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
            loss_txt.write(str(loss_txt_content)) 
            
        
        elif args.latentattack == True:  #   表征层对抗攻击
            print("-----------------generate representation-level perceptual examples----------------")
            print("eps:",args.attack_eps)
            print("cla_network_pkl:",args.cla_network_pkl)
            learned_cla_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_cla_model)

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

            cle_test_accuracy, cle_test_loss = latent_attacker.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'Accuary of load {args.cla_model} classifier on clean testset:{cle_test_accuracy * 100:.4f}%' ) 
            print(f'Loss of load {args.cla_model} classifier on clean testset:{cle_test_loss}' )             

            print("--------start generate representation-level adversarial examples------")
            print("args.saveadvtrain:",args.saveadvtrain)
            if args.saveadvtrain == True and args.projected_trainset != None:
                
                print("(args.projected_trainset:",args.projected_trainset)
                target_classifier._args.projected_dataset = args.projected_trainset  #   getproset需要用到_args.projected_dataset         
                cle_w_train, cle_y_train = target_classifier.getproset(target_classifier._args.projected_dataset)
                cle_y_train = cle_y_train[:,0]    #这个不能注释
                print("cle_w_train.shape:",cle_w_train.shape)
                print("cle_y_train.shape:",cle_y_train.shape)
                # adv_x_train, adv_y_train = latent_attacker.generatelatentadv(exp_result_dir, cle_w_train, cle_y_train, gan_net)       
                latent_attacker.generatelatentadv(exp_result_dir, cle_w_train, cle_y_train, gan_net)       

                cle_w_train=None 
                cle_y_train=None  
                # latent_attacker._args.projected_trainset = None   # 强制使其在后面变为0 避免误输入trainset
            
            
            latent_attacker._args.projected_trainset = None   # 强制使其在后面变为0
            print("(args.projected_testset:",args.projected_testset)   
            target_classifier._args.projected_dataset = args.projected_testset  
            cle_w_test, cle_y_test = target_classifier.getproset(target_classifier._args.projected_dataset)
            cle_y_test = cle_y_test[:,0]    #这个不能注释
            print("cle_w_test.shape:",cle_w_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
                        
            adv_x_test, adv_y_test = latent_attacker.generatelatentadv(exp_result_dir, cle_w_test, cle_y_test, gan_net)                   
            
            target_classifier.model().eval()          
            adv_test_accuracy, adv_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),adv_x_test, adv_y_test)
            print(f'Accuary of load {args.cla_model} classifier on OM adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'Loss of load {args.cla_model} classifier on OM adversarial testset:{adv_test_loss}' ) 

            accuracy_txt=open(f'{latent_attacker.getexpresultdir()}/load-{args.cla_model}-accuracy-on-omadv-{args.dataset}-testset.txt', "w")    
            txt_content = f'{latent_attacker.getexpresultdir()}/load-{args.cla_model}-accuracy-on-omadv-{args.dataset}-testset = {adv_test_accuracy}\n'
            accuracy_txt.write(str(txt_content))
        
            loss_txt=open(f'{latent_attacker.getexpresultdir()}/load-{args.cla_model}-loss-on-omadv-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{latent_attacker.getexpresultdir()}/load-{args.cla_model}-loss-on-omadv-{args.dataset}-testset = {adv_test_loss}\n'
            loss_txt.write(str(loss_txt_content)) 
            
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
        print("args.lr:",args.lr)
        print("args.attack_mode:",args.attack_mode)
        print("args.cla_network_pkl:", args.cla_network_pkl)
        
        if  args.cla_network_pkl is None:
            print("load unlearned model !!!!!")
            target_classifier = MaggieClassifier(args)  #加载初始模型                      
        else:
            print("load learned model !!!!!!")
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)     
            
        # raise error
        # 干净样本测试集
        cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
        print("cle_x_test.shape:",cle_x_test.shape)
        print("cle_y_test.shape:",cle_y_test.shape)
                       
        # 对抗样本测试集
        print("args.test_adv_dataset",args.test_adv_dataset)
        adv_testset_path = args.test_adv_dataset
        adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
        print("adv_x_test.shape:",adv_x_test.shape)
        print("adv_y_test.shape:",adv_y_test.shape) 

        # clean pixel testset acc and loss
        cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
        print(f'Accuary of load {args.cla_model} classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
        print(f'Loss of load {args.cla_model} classifier on clean testset:{cle_test_loss}' ) 
                              
        # adv testset acc and loss
        adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
        
        print(f'Accuary of load {args.cla_model} classifier on load adv testset:{adv_test_acc * 100:.4f}%' ) 
        print(f'Loss of load {args.cla_model} classifier on load adv testset:{adv_test_loss}' ) 
                                                                                        
        if args.defense_mode == "rmt":
            print("-----------------defense with representation mixup training----------------")

            # 干净样本投影训练集
            print("args.projected_trainset",args.projected_trainset)
            target_classifier._args.projected_dataset = args.projected_trainset  #   getproset需要用到_args.projected_dataset         
            cle_w_train, cle_y_train = target_classifier.getproset(target_classifier._args.projected_dataset)
            print("cle_w_train.shape:",cle_w_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)

            # 什么时候不用train
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

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_mode in ["fog","snow","elastic","jpeg"]:
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对非PreAct的perceptual攻击 只评估精度 
                raise error            

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_eps != 0.02 and args.attack_mode != 'pgd':    
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                # 针对非PreAct的eps不等于0.02的攻击 只评估精度 

                raise error
            
            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)
            print("args.lr_schedule:",args.lr_schedule)
            print("args.optimizer:",args.optimizer)
            
            # train
            if args.lr_schedule == 'CosineAnnealingLR':
                print("rmt sgd cosine annel ing !!!!")
                target_classifier.rmt_sgd_cos(args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs)
            else:
                print("rmt(0.1 annel) ing  !!!!") # 0.1annel更有效,最终选她
                target_classifier.rmt(args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs)
                
        elif args.defense_mode =='inputmixup':
            print("-----------------defense with input mixup training----------------")

            # clean pixel testset
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()
            print("cle_y_train.shape:",cle_y_train.shape)

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
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

        elif args.defense_mode =='manifoldmixup':
            print("-----------------defense with manifold mixup training----------------")

            # print("args.cla_network_pkl:", args.cla_network_pkl)
        
            # if  args.cla_network_pkl is None:
            #     print("load unlearned model !!!!!")
            #     target_classifier = MaggieClassifier(args)  #加载初始模型                      
            # else:
            #     print("load learned model !!!!!!")
            #     # learned_model = torch.load(args.cla_network_pkl)
            #     # # path = os.path.join(self.save_path, 'best_network.pth')
                
            #     # torch.save(learned_model.state_dict(), '/home/data/maggie/result-newhome/train/cla-train/wideresnet28_10-cifar10/20230326/00002-testacc-0.9670/train-cifar10-dataset/standard-trained-classifier-wideresnet28_10-on-clean-cifar10-finished.pth')	# 存成参数后重新读取
                
            #     target_classifier = MaggieClassifier(args)               
            #     target_classifier.model().load_state_dict(torch.load(args.cla_network_pkl))
                                        
            #     # target_classifier = MaggieClassifier(args,learned_model)     



            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
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

        elif args.defense_mode =='patchmixup':
            print("-----------------defense with patch mixup training----------------")

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
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

        elif args.defense_mode =='puzzlemixup':
            print("-----------------defense with puzzle mixup training----------------")

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
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

        elif args.defense_mode =='cutmixup':
            print("-----------------defense with cut mixup training----------------")

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  #   标签转为one hot
            print("cle_y_train.shape:",cle_y_train.shape)

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
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

        elif args.defense_mode =='at':
            print("-----------------defense with adversarial training----------------")

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)

            # 对抗样本训练集
            # print("args.train_adv_dataset",args.train_adv_dataset)
            adv_trainset_path = args.train_adv_dataset
            adv_x_train, adv_y_train = target_classifier.getadvset(adv_trainset_path)
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)            
            # adv_x_train=adv_x_train[:25397]
            # adv_y_train=adv_y_train[:25397]

            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before at trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before at trained classifier on adv testset:{adv_test_loss}' ) 
                raise error
            
            target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)
                
        elif args.defense_mode =='dmat':
            print("-----------------defense with dual manifold adversarial training----------------")

            # 干净样本训练集
            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            # # 干净样本测试集
            # print("cle_x_test.shape:",cle_x_test.shape)
            # print("cle_y_test.shape:",cle_y_test.shape)

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

            # 干净样本投影训练集
            print("args.projected_trainset",args.projected_trainset)
            target_classifier._args.projected_dataset = args.projected_trainset  #   getproset需要用到_args.projected_dataset         
            cle_w_train, cle_y_train = target_classifier.getproset(target_classifier._args.projected_dataset)
            print("cle_w_train.shape:",cle_w_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            
            if args.attack_mode != "fgsm" and args.attack_mode != "pgd":
                # adv pixel testset acc and loss
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_loss}' )           
                raise error("maggie stop here")
            
            # target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)



            # train
            if args.lr_schedule == 'CosineAnnealingLR':
                print("rmt sgd cosine annel ing !!!!")
                
                # target_classifier.advtrain_sgd_cos(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)

                target_classifier.advtrain_sgd_cos_genom(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir, stylegan2ada_config_kwargs,cle_w_train, cle_y_train)
                
                
                
            else:
                print("rmt(0.1 annel) ing  !!!!") # 0.1annel更有效,最终选她
                
                target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)
                

        print("-----------------defense training finished----------------")    
        
        target_classifier.model().eval()
        # evaluate acc and loss on clean pixel testset 
        cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
        print(f'Accuary of robust trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
        print(f'Loss of robust trained classifier on clean testset:{cle_test_loss}' ) 

        # evaluate acc and loss on adversarial testset 
        if args.whitebox == True:
            attack_classifier = AdvAttack(args, target_classifier.model())
            target_model = attack_classifier.targetmodel()
            adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
            adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
            print(f'Accuary of {args.defense_mode} trained {args.cla_model} classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
            print(f'Loss of {args.defense_mode} trained {args.cla_model} classifier on white-box adv testset:{adv_test_loss}' ) 

        elif args.blackbox == True:
            # adv_x_test, adv_y_test = adv_x_test, adv_y_test #original adv testset
            adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
            print(f'Accuary of {args.defense_mode} trained {args.cla_model} classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
            print(f'Loss of {args.defense_mode} trained {args.cla_model} classifier on black-box adv testset:{adv_test_loss}' ) 
    
    elif args.mode =='eval':
        print("args.attack_mode:",args.attack_mode)
        print("args.cla_network_pkl:", args.cla_network_pkl)
        print("args.seed:",args.seed)
        print("****************")
        print("args.attack_mode:",args.attack_mode)
        print("args.attack_eps:",args.attack_eps)
        print("args.attack_eps:",args.attack_eps)
        print("args.attack_max_iter:",args.attack_max_iter)
        print("****************")        
        
        print("load learned model !!!!!!")
        learned_model = torch.load(args.cla_network_pkl)
        target_classifier = MaggieClassifier(args,learned_model)     
            
        # 干净样本测试集
        cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
        print("cle_x_test.shape:",cle_x_test.shape)
        print("cle_y_test.shape:",cle_y_test.shape)

        # clean pixel testset acc and loss
        cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
        print(f'Accuary of load {args.cla_model} classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
        print(f'Loss of load {args.cla_model} classifier on clean testset:{cle_test_loss}' )            
        print("-------------------------------------------") 

        # 对抗样本测试集
        print("args.test_adv_dataset",args.test_adv_dataset)
        adv_testset_path = args.test_adv_dataset
        adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
        print("adv_x_test.shape:",adv_x_test.shape)
        print("adv_y_test.shape:",adv_y_test.shape)                         
        
        # adv_x_test, adv_y_test = adv_x_test, adv_y_test #original adv testset
        adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
        print(f'Accuary of {args.defense_mode} load {args.cla_model} classifier on load adv testset:{adv_test_acc * 100:.4f}%' ) 
        print(f'Loss of {args.defense_mode} load {args.cla_model} classifier on load adv testset:{adv_test_loss}' )                               
        print("-------------------------------------------") 
                              
        # evaluate acc and loss on adversarial testset 
        if args.whitebox == True:
            
            if args.latentattack ==True:
                
                cla_net = learned_model

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
                
                latent_attacker._args.projected_trainset = None   # 强制使其在后面变为0
                # print("(args.projected_testset:",args.projected_testset)   
                target_classifier._args.projected_dataset = args.projected_testset  
                cle_w_test, cle_y_test = target_classifier.getproset(target_classifier._args.projected_dataset)
                cle_y_test = cle_y_test[:,0]    #这个不能注释
                # print("cle_w_test.shape:",cle_w_test.shape)
                # print("cle_y_test.shape:",cle_y_test.shape)                

                adv_x_test, adv_y_test = latent_attacker.generatelatentadv(exp_result_dir, cle_w_test, cle_y_test, gan_net)                   
                
                target_classifier.model().eval()          
                adv_test_accuracy, adv_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),adv_x_test, adv_y_test)
                print(f'Accuary of load {args.cla_model} classifier on real generate OM adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
                print(f'Loss of load {args.cla_model} classifier on real generate OM adversarial testset:{adv_test_loss}' )                 
                
                
            else:
            
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of load {args.cla_model} classifier on real generate white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of load {args.cla_model} classifier on real generate white-box adv testset:{adv_test_loss}' ) 
                
                # adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                # print(f'Accuary of {args.defense_mode} load {args.cla_model} classifier on load adv testset:{adv_test_acc * 100:.4f}%' ) 
                # print(f'Loss of {args.defense_mode} load {args.cla_model} classifier on load adv testset:{adv_test_loss}' )                                           
            
            
            
            
            
            
            
            
            
            # print("-------------------------------------------") 

        # elif args.blackbox == True:
        #     # 对抗样本测试集
        #     # print("args.test_adv_dataset",args.test_adv_dataset)
        #     # adv_testset_path = args.test_adv_dataset
        #     # adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
        #     # print("adv_x_test.shape:",adv_x_test.shape)
        #     # print("adv_y_test.shape:",adv_y_test.shape)                         
            
        #     # # adv_x_test, adv_y_test = adv_x_test, adv_y_test #original adv testset
        #     # adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
        #     # print(f'Accuary of {args.defense_mode} load {args.cla_model} classifier on load black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
        #     # print(f'Loss of {args.defense_mode} load {args.cla_model} classifier on load black-box adv testset:{adv_test_loss}' )                               
        #     print("------------See accuray on load dataset-------------------------------") 
                
                                 
    print("---------------------------------------")
    print("\n")