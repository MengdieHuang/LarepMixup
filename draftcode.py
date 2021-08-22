"""class classifier"""
        # # initilize the model architecture
        # if self.opt.network_pkl == None:
        #     # print('no trained classifier !')

        #     if self.opt.pretrained_on_imagenet == False:
        #         self.classify_model = CustomClassifier(self.opt.model, self.opt.n_classes, False)

        #     elif self.opt.pretrained_on_imagenet == True:
        #         # print('self.opt.model=%s'%self.opt.model)
        #         self.classify_model = torchvision.models.__dict__[self.opt.model]

        # elif self.opt.network_pkl != None:
        #     # print(f'load trained classifier from {self.opt.network_pkl}')
        #     self.classify_model = torch.load(self.opt.network_pkl)
        
        # # initilize the loss function
        # self.classify_loss = torch.nn.CrossEntropyLoss()

        # # initilize the optimizer
        # self.classify_optimizer = torch.optim.Adam(self.classify_model.parameters(), lr=self.opt.lr)

        # # initilize the dataloader
        # self.train_dataloader, self.test_dataloader = self.data()

    # def data(self):
    #     if self.opt.dataset == 'mnist':
    #         return datasets.dataload.LoadMNIST(self.opt)
    #     elif self.opt.dataset == 'kmnist':
    #         return datasets.dataload.LoadKMNIST(self.opt)
    #     elif self.opt.dataset == 'cifar10':
    #         return datasets.dataload.LoadCIFAR10(self.opt)
    #     elif self.opt.dataset == 'cifar100':
    #         return datasets.dataload.LoadCIFAR100(self.opt)
    #     elif self.opt.dataset == 'imagenet':
    #         return datasets.dataload.LoadIMAGENET(self.opt)
    #     elif self.opt.dataset == 'lsun':
    #         return datasets.dataload.LoadLSUN(self.opt)
    #     elif self.opt.dataset == 'stl10':
    #         return datasets.dataload.LoadSTL10(self.opt)

"""加载torchvision models"""
# def get_models_last(model):
#     last_name = list(model._modules.keys())[-1]
#     last_module = model._modules[last_name]
#     return last_module, last_name

# class CustomClassifier(torch.nn.Module):
#     def __init__(self, arch: str, num_classes: int, pretrained: bool = True):
#         super().__init__()
#         if pretrained:
#            self.model = torchvision.models.__dict__[arch](pretrained = pretrained)
#         else:
#            self.model = torchvision.models.__dict__[arch]()
        
#         # last_module, last_name = get_models_last(self.model)
#         last_module, last_name = get_models_last(self.model)
       
#         if isinstance(last_module, torch.nn.Linear):
#             n_features = last_module.in_features
#             self.model._modules[last_name] = torch.nn.Linear(n_features, num_classes)
#         elif isinstance(last_module, torch.nn.Sequential):
#             # seq_last_module, seq_last_name = get_models_last(last_module)
#             seq_last_module, seq_last_name = get_models_last(last_module)

#             n_features = seq_last_module.in_features
#             last_module._modules[seq_last_name] = torch.nn.Linear(n_features, num_classes)

#         #just for test
#         self.last = list(self.model.named_modules())[-1][1]

#     def forward(self, input_neurons):
#         # TODO: add dropout layers, or the likes.
#         output_predictions = self.model(input_neurons)
#         return output_predictions

"""训练分类器"""
#    def train(self,exp_result_dir):
#         print('train classifier')

#         epochs = self.opt.epochs
#         learn_rate = self.opt.lr
#         optimizer = self.classify_optimizer
#         classifier = self.classify_model
#         classify_loss = self.classify_loss
#         train_dataloader = self.train_dataloader
#         test_dataloader = self.test_dataloader

#         global_train_acc = []
#         global_test_acc = []
#         global_train_loss = []
#         global_test_loss = []

#         trainset_total_num = len(train_dataloader.dataset)
#         print("train set total num:",trainset_total_num)     #   cifar10 500000
       
#         if torch.cuda.is_available():
#             classify_loss.cuda()
#             classifier.cuda()

#         for epoch_index in range(epochs):
#             adjust_learning_rate(learn_rate,optimizer,epoch_index)       
#             epoch_correct_num = 0
#             epoch_total_loss = 0
#             for batch_index, (images, labels) in enumerate(train_dataloader):
#                 imgs = images.cuda()
#                 labs = labels.cuda()
#                 optimizer.zero_grad()
#                 output = classifier(imgs)
#                 loss = classify_loss(output,labs)
#                 loss.backward()
#                 optimizer.step()
#                 print("[Epoch %d/%d] [Batch %d/%d] [Classify loss: %f] " % (epoch_index, epochs, batch_index, len(train_dataloader), loss.item()))

#                 #--------在当前训练集batch上的准确率-------------
#                 _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index
#                 batch_same_num = (predicted_label_index == labs).sum().item()     #   当前batch的正确数目
#                 epoch_correct_num += batch_same_num                                     #   加上本轮之前所有batch的正确数目总和
#                 epoch_total_loss += loss

#             #--------当前epoch分类模型在当前epoch上的准确率-------------            
#             epoch_train_accuarcy = epoch_correct_num / trainset_total_num
#             print(f'Classifier accuary on the {epoch_index} epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
#             global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list
            
#             if epoch_train_accuarcy > 0.9:
#                 print(f'epoch index = {epoch_index} epoch_train_accuarcy = {epoch_train_accuarcy}%')

#             #--------当前epoch分类模型在当前epoch上的损失-------------                            
#             epoch_train_loss = epoch_total_loss / len(train_dataloader)

#             print(f'Classifier loss on the {epoch_index} epoch training examples:{epoch_train_loss:.4f}' )   
#             global_train_loss.append(epoch_train_loss)

#             #--------当前epoch分类模型在测试集整体上的准确率------------- 
#             epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(classifier,classify_loss,test_dataloader)
#             print(f'{epoch_index} epoch Classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )   
#             global_test_acc.append(epoch_test_accuracy)   #   每个epoch训练完后的最新准确率list
#             global_test_loss.append(epoch_test_loss)

#             if epoch_index % 50 == 0:
#                 torch.save(classifier,f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-on-clean-{self.opt.dataset}-epoch-{epoch_index:04d}.pkl')

#         torch.save(classifier,f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-on-clean-{self.opt.dataset}-finished.pkl')

#         accuracy_png_name = f'standard trained classifier {self.opt.model} accuracy on clean {self.opt.dataset}'
#         SaveAccuracyCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_acc,global_test_acc,accuracy_png_name)
        
#         # accuracy_txt=open(f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}.txt', "w")    
#         # accuracy_txt_content = f"{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}-trainset = {global_train_acc[-1]} \n{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}-testset = {global_test_acc[-1] }"
#         # accuracy_txt.write(str(accuracy_txt_content))

#         loss_png_name = f'standard trained classifier {self.opt.model} loss on clean{self.opt.dataset}'
#         SaveLossCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_loss,global_test_loss,loss_png_name)
        
#         # loss_txt=open(f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}.txt', "w")    
#         # loss_txt_content = f"{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}-trainset = {global_train_loss[-1]} \n{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}-testset = {global_test_loss[-1] }"
#         # loss_txt.write(str(loss_txt_content))

#         return self.classify_model

"""加载data"""
    # def __gettraindataloader__(self) -> "torch.utils.data.DataLoader":
        
    #     train_dataloader = self._args.train_dataloader
        
    #     return train_dataloader

    # def __gettestdataloader__(self) -> "torch.utils.data.DataLoader":
        
    #     test_dataloader = self._args.test_dataloader
        
    #     return test_dataloader

"""评估"""
    # def evaluate(self,classify_model,test_dataloader,exp_result_dir):       
    #     classify_loss= self.classify_loss
        
    #     if torch.cuda.is_available():
    #         classify_model.cuda()
    #         classify_loss.cuda()
    #     test_accuracy, test_loss = EvaluateAccuracy(classify_model,classify_loss,test_dataloader)
    #     # print(f'classifier *accuary* on testset:{test_accuracy * 100:.4f}%' ) 
    #     # print(f'classifier *loss* on testset:{test_loss}' ) 

    #     return test_accuracy, test_loss

"""advattack"""
# class AdvAttackClassifier:
#     def __init__(self,opt,classify_model):
#         # initialize the parameters
#         self.opt = opt   

#         # initilize the target model architecture
#         self.target_classify_model = classify_model  

#         # initilize the attack model architecture
#         self.white_box = True                       #   white box attack or black box attack
#         if self.white_box == True:
#             self.attack_classify_model = classify_model
#         elif self.white_box == False:
#             self.attack_classify_model = torchvision.models.resnet34(pretrained=True)

#         # initilize the loss function
#         self.classify_loss = torch.nn.CrossEntropyLoss()
        
#         # initilize the optimizer
#         self.classify_optimizer =  torch.optim.Adam(self.attack_classify_model.parameters(), lr=self.opt.lr)
    
#         #initilize the art package of attack model (to use art attack algorithms)
#         self.art_attack_classify_model = self.GetArtClassifier()

#         # get the trained art package of attack model
#         self.art_estimator_model = self.GetAdvtAttackClassifier()

#     def GetArtClassifier(self):

#         if torch.cuda.is_available():
#             self.classify_loss.cuda()
#             self.attack_classify_model.cuda()      
        
#         data_raw = False                                        #   是否在之前对数据集进行过归一化
#         if data_raw == True:
#             min_pixel_value = 0.0
#             max_pixel_value = 255.0
#         else:
#             min_pixel_value = 0.0
#             max_pixel_value = 1.0        

#         art_attack_classify_model = PyTorchClassifier(
#             model=self.attack_classify_model,
#             clip_values=(min_pixel_value, max_pixel_value),
#             loss=self.classify_loss,
#             optimizer=self.classify_optimizer,
#             input_shape=(self.opt.channels, self.opt.img_size, self.opt.img_size),
#             nb_classes=self.opt.n_classes,
#         )             
#         return art_attack_classify_model

#     def GetAdvtAttackClassifier(self):
#         if self.opt.attack_mode == 'fgsm':                              #   FGSM攻击
#             print('generating FGSM examples')
#             return FastGradientMethod(estimator=self.art_attack_classify_model, eps=0.2, targeted=False)    #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self.opt.attack_mode =='deepfool':                         #   DeepFool攻击
#             return DeepFool(classifier=self.art_attack_classify_model, epsilon=0.2)                         #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self.opt.attack_mode =='bim':                              #   BIM攻击
#             return BasicIterativeMethod(estimator=self.art_attack_classify_model, eps=0.2, targeted=False)  #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self.opt.attack_mode =='cw':                               #   CW攻击
#             return CarliniL2Method(classifier=self.art_attack_classify_model, targeted=False)               #   estimator: A trained classifier. eps: Attack step size (input variation).
#         elif self.opt.attack_mode == None:
#             raise Exception('please input the attack mode')   

#     def generate(self,exp_result_dir):
#         # initilize the dataloader
#         self.train_dataloader, self.test_dataloader = self.data()

#         # initilize the data ndArray
#         self.x_train,self.y_train,self.x_test,self.y_test = self.GetCleanDataArray()  

#         print('generating adversarial examples')
#         self.x_train_adv = self.art_estimator_model.generate(x = self.x_train, y = self.y_train)
#         self.y_train_adv = self.y_train
#         self.x_test_adv = self.art_estimator_model.generate(x = self.x_test, y = self.y_test)
#         self.y_test_adv = self.y_test
#         print('finished generate adversarial examples')
#         # print('x_test_adv:',self.x_test_adv[:3])
#         # print('x_test:',self.x_test[:3])

#         if torch.cuda.is_available():
#             Tensor = torch.cuda.FloatTensor 
#         else:
#             Tensor = torch.FloatTensor

#         # save png
#         for img_index, img in enumerate(self.x_test_adv):
#             if img_index % 1000 == 0:
#                 save_adv_img = self.x_test_adv[img_index:img_index+25]
#                 save_adv_img = Tensor(save_adv_img)
#                 save_cle_img = self.x_test[img_index:img_index+25]
#                 save_cle_img = Tensor(save_cle_img)
#                 print('save_adv_img.shape:',save_adv_img.shape)
#                 if save_adv_img.size(0) == 25 :
#                     save_image(save_adv_img, f'{exp_result_dir}/cle-testset-{img_index}.png', nrow=5, normalize=True)
#                     save_image(save_cle_img, f'{exp_result_dir}/adv-testset-{img_index}.png', nrow=5, normalize=True)       

#         self.train_set = AugCIFAR10(                                             #   用 torchvision.datasets.MNIST类的构造函数返回值给DataLoader的参数 dataset: torch.utils.data.dataset.Dataset[T_co]赋值 https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
#             "/home/data/maggie/cifar10",
#             train=True,                                             #   从training.pt创建数据集
#             download=False,                                          #   自动从网上下载数据集
#             transform=transforms.Compose(
#                 [
#                     transforms.Resize(self.opt.img_size), 
#                     transforms.CenterCrop(self.opt.img_size),
#                     transforms.ToTensor(), 
#                     # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                 ]
#             ),
#         )

#         self.test_set = AdvCIFAR10(                                             #   用 torchvision.datasets.MNIST类的构造函数返回值给DataLoader的参数 dataset: torch.utils.data.dataset.Dataset[T_co]赋值 https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
#             "/home/data/maggie/cifar10",
#             train=False,                                             #   从training.pt创建数据集
#             download=False,                                          #   自动从网上下载数据集
#             transform=transforms.Compose(
#                 [
#                     transforms.Resize(self.opt.img_size), 
#                     transforms.CenterCrop(self.opt.img_size),
#                     transforms.ToTensor(), 
#                     # transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
#                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#                 ]
#             ),
#         )

#         # self.train_set = self.train_dataloader.dataset
#         # self.test_set = self.test_dataloader.dataset

#         self.train_set.AugmentData(self.x_train_adv,self.y_train_adv)
#         self.test_set.AdvReplaceData(self.x_test_adv,self.y_test_adv)
#         self.aug_train_dataloader = datasets.dataload.LoadCIFAR10Train(self.opt,self.train_set)
#         self.adv_test_dataloader = datasets.dataload.LoadCIFAR10Test(self.opt,self.test_set)

#         return self.aug_train_dataloader, self.adv_test_dataloader


#     def data(self):
#         if self.opt.dataset == 'mnist':
#             return datasets.dataload.LoadMNIST(self.opt)
#         elif self.opt.dataset == 'kmnist':
#             return datasets.dataload.LoadKMNIST(self.opt)
#         elif self.opt.dataset == 'cifar10':
#             return datasets.dataload.LoadCIFAR10(self.opt)
#         elif self.opt.dataset == 'cifar100':
#             return datasets.dataload.LoadCIFAR100(self.opt)
#         elif self.opt.dataset == 'imagenet':
#             return datasets.dataload.LoadIMAGENET(self.opt)
#         elif self.opt.dataset == 'lsun':
#             return datasets.dataload.LoadLSUN(self.opt)
#         elif self.opt.dataset == 'stl10':
#             return datasets.dataload.LoadSTL10(self.opt)

#     def GetCleanDataArray(self):
#         train_dataloader = self.train_dataloader
#         test_dataloader = self.test_dataloader

#         if self.opt.dataset == 'cifar10':
#             jieduan_num = 1000
#             # print("train_dataloader.dataset.__dict__.keys:",train_dataloader.dataset.__dict__.keys())
#             total_num = len(train_dataloader.dataset)
#             x_train = train_dataloader.dataset.data                                             #   shape是（50000，32，32，3）
#             y_train = train_dataloader.dataset.targets

#             # x_train = x_train[:jieduan_num]
#             # y_train = y_train[:jieduan_num]

#             x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)    
#             print('trainset total_num :',total_num)
#             print('x_train.shape:',x_train.shape)


#             total_num = len(test_dataloader.dataset)
#             x_test = test_dataloader.dataset.data                                             #   shape是（50000，32，32，3）
#             y_test = test_dataloader.dataset.targets

#             # x_test = x_test[:jieduan_num]
#             # y_test = y_test[:jieduan_num]

#             x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)    
#             print('testset total_num :',total_num)
#             print('x_test.shape:',x_test.shape)    

#         elif self.opt.dataset == 'imagenet':
#             jieduan_num = 100
#             x_train = []                                             #   shape是（50000，32，32，3）
#             y_train = []       
#             train_set_len = len(train_dataloader.dataset)
#             print('trainset len:',train_set_len)

#             for index in range(jieduan_num):
#                 img, label = train_dataloader.dataset.__getitem__(index)
#                 # print("img.shape:",img.shape)       #   [3,1024,1024]
#                 x_train.append(img)
#                 y_train.append(label)      
#             x_train = torch.stack(x_train) 
#             print("x_train shape:",x_train.shape)    
#             x_train = x_train.numpy()
#             print("x_train shape:",x_train.shape)    

#             x_test = []
#             y_test = []
#             test_set_len = len(test_dataloader.dataset)
#             print('testlen len:',test_set_len)
#             for index in range(test_set_len):
#                 img, label = test_dataloader.dataset.__getitem__(index)
#                 x_test.append(img)                                          #   张量list
#                 y_test.append(label)

#             x_test = torch.stack(x_test) 
#             print("x_test shape:",x_test.shape)    
#             x_test = x_test.numpy()
#             print("x_test shape:",x_test.shape)    
        
#         return x_train,y_train,x_test,y_test


#     def evaluate(self,classify_model,test_dataloader,exp_result_dir):
#         print('evaluate the trained model on adversarial examples')
#         classify_loss= self.classify_loss

#         if torch.cuda.is_available():
#             classify_model.cuda()
#             classify_loss.cuda()

#         test_accuracy, test_loss = EvaluateAccuracy(classify_model,classify_loss,test_dataloader)

#         return test_accuracy,test_loss
        


