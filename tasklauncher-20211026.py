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

    if args.mode == 'train':
        if args.train_mode =="gen-train":                                               
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
        elif args.train_mode =="cla-train":    
            target_classifier = MaggieClassifier(args)
            # cle_x_train, cle_y_train = target_classifier.settensor(cle_train_dataloader)
            # cle_x_test, cle_y_test = target_classifier.settensor(cle_test_dataloader)

            if args.pretrained_on_imagenet == False:
                target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  
            elif args.pretrained_on_imagenet == True:
                target_classifier = target_classifier

            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                    #   *accuary* on testset:75.6900%
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

    elif args.mode == 'attack':
        if args.latentattack == False:    
            if args.perceptualattack == False:  #   像素层对抗攻击
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
            print("latent adversarial attack.............")
            learned_cla_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_cla_model)
            cle_w_test, cle_y_test = target_classifier.getproset(args.projected_dataset)
            cle_w_test = cle_w_test[:10]
            cle_y_test = cle_y_test[:10]
            cle_y_test = cle_y_test[:,0]
    
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


            gan_net.eval()
            testset_total_num = int(cle_w_test.size(0))
            batch_size = args.batch_size
            batch_num = int(np.ceil( int(testset_total_num) / float(batch_size) ) )
            cle_x_test = []
            for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
                cle_w_batch = cle_w_test[batch_index * batch_size : (batch_index + 1) * batch_size]
                cle_x_batch = gan_net(cle_w_batch.cuda())        
                cle_x_test.append(cle_x_batch)

            cle_x_test = torch.cat(cle_x_test, dim=0)        
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_test_accuracy, cle_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),cle_x_test, cle_y_test)
            print(f'standard trained classifier accuary on clean testset:{cle_test_accuracy * 100:.4f}%' ) 
            print(f'standard trained classifier loss on clean testset:{cle_test_loss}' )    

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
            generate_model.projectmain(cle_train_dataloader) 
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
            
            # model
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

            # data
            # clean representation trainset
            cle_w_train, cle_y_train = target_classifier.getproset(args.projected_dataset)

            # clean pixel testset
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            # raw_x_train, raw_y_train = target_classifier.getrawset(cle_train_dataloader)

            # adversarial testset
            print("args.adv_dataset：",args.adv_dataset)
            adv_testset_path = os.path.join(args.adv_dataset,'test')
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     #   bug
            print(f'Accuary of before rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before mmat trained classifier clean testset:{cle_test_loss}' ) 

            # # adv pixel testset acc and loss
            # adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
            # print(f'Accuary of before rmt trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
            # print(f'Loss of before rmt trained classifier on white-box adv testset:{adv_test_loss}' ) 
            # # raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)

            
            # train
            target_classifier.rmt(args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs)

            # test
            # clean pixel testset acc and loss
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of rmt trained classifier on clean testset:{cle_test_loss}' ) 
           
            # adversarial pixel testset acc and loss
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

        else:
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = MaggieClassifier(args,learned_model)

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

            # raise error

            if args.defense_mode == "at":

                target_classifier.adversarialtrain(args, cle_x_train,cle_y_train, x_train_adv, y_train_adv, x_test_adv, y_test_adv, target_classifier.artmodel(),exp_result_dir)
                    
                
                at_cle_test_accuracy, at_cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
                print(f'adversarial trained classifier accuary on clean testset:{at_cle_test_accuracy * 100:.4f}%' ) 
                print(f'adversarial trained classifier loss on clean testset:{at_cle_test_loss}' ) 

                at_adv_test_accuracy, at_adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),x_test_adv,y_test_adv)
                print(f'adversarial trained classifier accuary on adversarial testset:{at_adv_test_accuracy * 100:.4f}%' ) 
                print(f'adversarial trained classifier loss on adversarial testset:{at_adv_test_loss}' )     

                # SaveTxt(args,exp_result_dir,cle_test_accuracy,adv_test_accuracy,at_adv_test_accuracy,at_cle_test_accuracy, cle_test_loss,adv_test_loss,at_adv_test_loss,at_cle_test_loss)

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

                target_classifier.mmat(args, cle_x_train, cle_y_train, x_train_mix, y_train_mix, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir) 

                # target_classifier.mmat(args, cle_x_train, cle_y_train, x_train_mix, y_train_mix, cle_x_test, cle_y_test, x_test_adv, y_test_adv, exp_result_dir, target_classifier.artmodel())              

            #    生成对抗样本
                mmat_learned_model= target_classifier.model()
                attack_classifier = AdvAttack(args, mmat_learned_model)
                mmat_target_model = attack_classifier.targetmodel()
    
                # mmat_x_test_adv, mmat_y_test_adv = attack_classifier.generate(exp_result_dir,cle_test_dataloader)          #     只生成testset对抗样本
                mmat_x_test_adv, mmat_y_test_adv = attack_classifier.generateadvfromtestsettensor(exp_result_dir, cle_x_test, cle_y_test)

                mmat_adv_test_accuracy, mmat_adv_test_loss = target_classifier.evaluatefromtensor(mmat_target_model,mmat_x_test_adv,mmat_y_test_adv)
                print(f'mmat trained classifier accuary on adversarial testset:{mmat_adv_test_accuracy * 100:.4f}%' ) 
                print(f'mmat trained classifier loss on adversarial testset:{mmat_adv_test_loss}' )    
                mmat_cle_test_accuracy, mmat_cle_test_loss = target_classifier.evaluatefromtensor(mmat_target_model,cle_x_test,cle_y_test)
                print(f'mmat trained classifier accuary on clean testset:{mmat_cle_test_accuracy * 100:.4f}%' ) 
                print(f'mmat trained classifier loss on clean testset:{mmat_cle_test_loss}' ) 
                SaveTxt(args, exp_result_dir, cle_test_accuracy, adv_test_accuracy, mmat_adv_test_accuracy, mmat_cle_test_accuracy, cle_test_loss, adv_test_loss, mmat_adv_test_loss, mmat_cle_test_loss)
        
    print("---------------------------------------")
    print("\n")