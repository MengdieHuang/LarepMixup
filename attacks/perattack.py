import attacks.fog
import attacks.snow
import attacks.elastic
import attacks.gabor
import attacks.jpeg
import copy
import os
import torch
import numpy as np
from utils.savepng import save_image
from torch import LongTensor

Tensor = torch.Tensor


class PerAttack():
    r"""
        class of the perceptual attack 
        attributes:
        self._args
        self._model
        self._loss
        self._optimizer
        self._targetmodel
        self._test_dataloader
        self._whitebox
        self._artmodel
        self._advgenmodel

        methods:
        self.__init__()
        self.__getartmodel__()
        self.__getadvgenmodel__()
   
    """
    def __init__(self, args, learned_model) -> None:                 
        print('initlize perceptual classifier')
        self._args = args
        self._targetmodel = learned_model  
        self._model = copy.deepcopy(learned_model)
        self._attacker = self.__getperattackmodel__()

    def targetmodel(self):
        return self._targetmodel

    def __getperattackmodel__(self):   # 像素层感知攻击
        self._model.cuda()
        self._model.eval()   
        if self._args.attack_mode =='fog':
            print("pixel fog attack")
            attacker = attacks.fog.FogAttack(predict=self._model, nb_iters=200 , eps_max =128, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='snow':
            print("pixel snow attack")  
            attacker = attacks.snow.SnowAttack(predict=self._model, nb_iters=200 , eps_max =0.0625, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='elastic':
            print("pixel elastic attack") 
            attacker = attacks.elastic.ElasticAttack(predict=self._model, nb_iters=200 , eps_max =0.5, step_size =0.035355339059327376, resolution =self._args.img_size )

        elif self._args.attack_mode =='gabor':
            print("pixel gabor attack")  
            attacker = attacks.gabor.GaborAttack(predict=self._model, nb_iters=200 , eps_max =12.5, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='jpeg':
            print("pixel jpeg attack")   
            attacker = attacks.jpeg.JPEGAttack(predict=self._model, nb_iters=200 , eps_max =32, step_size =72.40773439350247, opt ='l1', resolution =self._args.img_size )  #   0.0000%
            # attacker = attacks.jpeg.JPEGAttack(predict=self._model, nb_iters=200 , eps_max =1, step_size =72.40773439350247, opt ='l1', resolution =self._args.img_size ) #   73.4375%
            # attacker = attacks.jpeg.JPEGAttack(predict=self._model, nb_iters=200 , eps_max =16, step_size =72.40773439350247, opt ='l1', resolution =self._args.img_size )    #   9.3750%

        return attacker
    
    def generate(self,exp_result_dir, cle_test_dataloader):
        self._test_dataloader = cle_test_dataloader 
        self._exp_result_dir = exp_result_dir

        self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)            

        print('generating testset perceptual attack examples...')
        per_x_test = []
        per_y_test = []
        for batch_index, (images, labels) in enumerate(self._test_dataloader):
            
            # if batch_index < 2 :
            #     print(f"batch_index = {batch_index:04d}/{len(self._test_dataloader):04d}")
            #     if batch_index == 0 :
            #         print("images.shape:",images.shape)             
            #         print("labels.shape:",labels.shape)             
            #         print("labels[:10]:",labels[:10]) 
            #         """
            #         images.shape: torch.Size([32, 3, 32, 32])
            #         labels.shape: torch.Size([32])
            #         labels[:10]: tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1])
            #         """
            #     images = images.cuda()
            #     labels = labels.cuda()
            #     per_x_batch = self._attacker.perturb(images,labels)
            #     per_y_batch = labels
            #     per_x_test.append(per_x_batch)
            #     per_y_test.append(per_y_batch)

            print(f"batch_index = {batch_index:04d}/{len(self._test_dataloader):04d}")
            images = images.cuda()
            labels = labels.cuda()
            per_x_batch = self._attacker.perturb(images,labels)
            per_y_batch = labels
            per_x_test.append(per_x_batch)
            per_y_test.append(per_y_batch)

        per_x_test = torch.cat(per_x_test, dim=0) 
        per_y_test = torch.cat(per_y_test, dim=0)    

        print("per_x_test.shape:",per_x_test.shape)             
        print("per_y_test.shape:",per_y_test.shape)
        print("per_y_test[:10]:",per_y_test[:10]) 

        """
        per_x_test.shape: torch.Size([64, 3, 32, 32])
        per_y_test.shape: torch.Size([64])
        per_y_test[:10]: tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1], device='cuda:0')        
        """

        self._x_test_per = per_x_test
        self._y_test_per = per_y_test
        self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

        print("self._x_test.shape:",self._x_test.shape)             
        print("self._y_test.shape:",self._y_test.shape)
        print("self._y_test[:10]:",self._y_test[:10]) 
        """
        self._x_test.shape: torch.Size([10000, 3, 32, 32])
        self._y_test.shape: torch.Size([10000])
        self._y_test[:10]: tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1], device='cuda:0')
        """

        self.__saveperpng__()

        return self._x_test_per, self._y_test_per         #   GPU tensor        


    def __labelnames__(self):
        opt = self._args
        # print("opt.dataset:",opt.dataset)
        
        label_names = []
        
        if opt.dataset == 'cifar10':
            label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
            #   label_names = ['飞机'，'汽车'，'鸟'，'猫'，'鹿'，'狗'，'青蛙'，'马'，'船'，'卡车']

        elif opt.dataset == 'cifar100': # = cle_train_dataloader.dataset.classes
            label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        
        elif opt.dataset =='svhn':
            label_names = ['0','1','2','3','4','5','6','7','8','9']

        elif opt.dataset =='kmnist':
            label_names = ['0','1','2','3','4','5','6','7','8','9']
        
        elif opt.dataset =='stl10': # cle_train_dataloader.dataset.classes 标签序号是0-9, dataloader 已调整数字0的标签为0
            label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
            #   label_names = ['飞机'，'鸟'，'汽车'，'猫'，'鹿'，'狗'，'马'，'猴子'，'船'，'卡车'] 
        
        elif opt.dataset =='imagenetmixed10':
            label_names = ['dog,','bird','insect','monkey','car','feline','truck','fruit','fungus','boat']        
            #   label_names = ['狗，'，'鸟'，'昆虫'，'猴子'，'汽车'，'猫'，'卡车'，'水果'，'真菌'，'船']
        else:
            raise Exception(" label name get wrong")            
        
        return label_names

    def __saveperpng__(self):
        if self._args.latentattack == False:      # 像素层攻击样本
            classification = self.__labelnames__() 
            os.makedirs(f'{self._exp_result_dir}/samples/test/',exist_ok=True)    
            print(f"Saving {self._args.dataset} testset perceptual attack examples...")
            for img_index, _ in enumerate(self._x_test_per):
                save_per_img = self._x_test_per[img_index]
                save_cle_img = self._x_test[img_index]
                img_true_label = self._y_test_per[img_index]

                np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-per-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_per_img.cpu().detach().numpy())      
                
                # save_image(save_per_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-per-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)
                # save_image(save_cle_img, f'{self._exp_result_dir}/samples/test/{img_index:08d}-cle-{img_true_label}-{classification[int(img_true_label)]}.png', nrow=5, normalize=True)  

    def __getsettensor__(self,dataloader)->"Tensor":

        xset_tensor  = self.__getxsettensor__(dataloader)
        yset_tensor = self.__getysettensor__(dataloader)

        return xset_tensor, yset_tensor
    
    def __getxsettensor__(self,dataloader)->"Tensor":

        # print("dataloader.dataset.data[0]:",dataloader.dataset.data[0])             #   dataloader.dataset.data[0]: [[[ 59  62  63]

        if self._args.dataset == 'cifar10':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                         

        elif self._args.dataset == 'cifar100':
            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
                                        

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000

            xset_tensor = []
            # for img_index in range(len(dataloader.dataset)):
            for img_index in range(jieduan_num):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
            
        elif self._args.dataset == 'svhn':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)   

        elif self._args.dataset == 'kmnist':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)  

        return xset_tensor.cuda()                                       #   xset_tensor原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接

    def __getysettensor__(self,dataloader)->"Tensor":

        if self._args.dataset == 'cifar10':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'cifar100':

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           
            # print("yset_tensor.type:", type(yset_tensor))                                         
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000
            # y_ndarray = []       
            # datasetset_len = len(dataloader.dataset)
            # print('datasetset len:',datasetset_len)

            # for index in range(jieduan_num):
            # # for index in range(datasetset_len):

            #     _, label = dataloader.dataset.__getitem__(index)
            #     y_ndarray.append(label)      

            yset_tensor = []
            # for img_index in range(len(dataloader.dataset)):
            for img_index in range(jieduan_num):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           
            # print("yset_tensor.type:", type(yset_tensor))                       #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)                      #   yset_tensor.shape: torch.Size([1000])

        elif self._args.dataset == 'svhn':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)

        elif self._args.dataset == 'kmnist':
        #     y_ndarray = dataloader.dataset.targets
        #     print("y_ndarray.type:", type(y_ndarray))

            # y_ndarray = y_ndarray[:jieduan_num]

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           #   list型转为tensor
            # print("yset_tensor.type:", type(yset_tensor))                   #   yset_tensor.type: <class 'torch.Tensor'>
            # print("yset_tensor.shape:", yset_tensor.shape)


        return yset_tensor.cuda()       #   yset_tensor 原本是CPU Tensor, 转成GPU Tenso,便于后面与mix样本拼接

    def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        classifier.eval()   #   eval mode        
        if torch.cuda.is_available():
            classifier.cuda()             
        
        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        cla_model_name=self._args.cla_model

        # print("x_set.shape:",x_set.shape)           #   x_set.shape: torch.Size([26032, 3, 32, 32])
        # print("y_set.shape:",y_set.shape)           #   y_set.shape: torch.Size([26032])
        # print("testset_total_num:",testset_total_num)       #   testset_total_num: 26032
        # print("batch_num:",batch_num)                   #   batch_num: 813.5
        # print("batch_size:",batch_size)       #  
        # print("cla_model_name:",cla_model_name)       #   cla_model_name: alexnet

        eva_loss = torch.nn.CrossEntropyLoss()
        epoch_correct_num = 0
        epoch_total_loss = 0

        for batch_index in range(batch_num):                                                #   进入batch迭代 共有num_batch个batch
            images = x_set[batch_index * batch_size : (batch_index + 1) * batch_size]
            labels = y_set[batch_index * batch_size : (batch_index + 1) * batch_size]                                                

            imgs = images.cuda()
            labs = labels.cuda()

            with torch.no_grad():

                if cla_model_name == 'inception_v3':
                    output, aux = classifier(imgs)
                
                elif cla_model_name == 'googlenet':
                    # if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10':  #   只有imagenet和svhn kmnist搭配googlenet时是返回一个值
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10':  #   只有imagenet和svhn kmnist搭配googlenet时是返回一个值
                        output = classifier(imgs)
                    else:
                        output, aux1, aux2 = classifier(imgs)
                else:
                    output = classifier(imgs)         
                                
                loss = eva_loss(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                
                batch_same_num = (predicted_label_index == labs).sum().item()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss


        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  
        classifier.train()

        return test_accuracy, test_loss

    def getexpresultdir(self):
        return self._exp_result_dir
    
