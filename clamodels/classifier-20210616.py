"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import torchvision
import torch
from evaluations.accuracy import EvaluateAccuracy
import datasets.dataload  
from utils.saveplt import SaveAccuracyCurve
from utils.saveplt import SaveLossCurve
from utils.savetxt import SaveAccuracyTxt
from utils.savetxt import SaveLossTxt


def smooth_step(a,b,c,d,epoch_index):
    if epoch_index <= a:        #   <=10
        return 0.01
    if a < epoch_index <= b:    #   10~25
        # return (((epoch_index-a)/(b-a))*(level_m-level_s)+level_s)
        return 0.001
    if b < epoch_index<=c:      #   25~30
        return 0.1
    if c < epoch_index<=d:      #   30~40
        return 0.01
    if d < epoch_index:         #   40~50
        return 0.0001

def adjust_learning_rate(learning_rate, classify_optimizer, epoch_index):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""#每隔30epoch除以一次10
    lr1 = learning_rate * (0.1 ** (epoch_index // 10))            #   30//30=1 31//30=1 60//30=2 返回整数部分
    lr2 = smooth_step(10,20,30,40,epoch_index)

    for param_group in classify_optimizer.param_groups:
        param_group['lr'] = lr1
        # param_group['lr'] = lr2

def get_models_last(model):
    last_name = list(model._modules.keys())[-1]
    last_module = model._modules[last_name]
    return last_module, last_name

class CustomClassifier(torch.nn.Module):
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
           self.model = torchvision.models.__dict__[arch](pretrained = pretrained)
        else:
           self.model = torchvision.models.__dict__[arch]()
        
        # last_module, last_name = get_models_last(self.model)
        last_module, last_name = get_models_last(self.model)
       
        if isinstance(last_module, torch.nn.Linear):
            n_features = last_module.in_features
            self.model._modules[last_name] = torch.nn.Linear(n_features, num_classes)
        elif isinstance(last_module, torch.nn.Sequential):
            # seq_last_module, seq_last_name = get_models_last(last_module)
            seq_last_module, seq_last_name = get_models_last(last_module)

            n_features = seq_last_module.in_features
            last_module._modules[seq_last_name] = torch.nn.Linear(n_features, num_classes)

        #just for test
        self.last = list(self.model.named_modules())[-1][1]

    def forward(self, input_neurons):
        # TODO: add dropout layers, or the likes.
        output_predictions = self.model(input_neurons)
        return output_predictions

class MaggieClassifier:
    r"""    
        class of the target classifier
        attributes:
        self.opt
        self.classify_model
        self.classify_loss
        self.classify_optimizer
        self.train_dataloader
        self.test_dataloader
        self.init()
        self.train()
        self.data()
        self.evaluate()

    """

    def __init__(self,args) -> None:                 # 双下划线表示只有Classifier类本身可以访问   ->后是对函数返回值的注释，None表明无返回值
        print('initlize classifier')

        # initilize the parameters
        # self.opt = opt
        self._args = args

        self._model = self.__getmodel__()



        # initilize the model architecture
        if self.opt.network_pkl == None:
            # print('no trained classifier !')

            if self.opt.pretrained_on_imagenet == False:
                self.classify_model = CustomClassifier(self.opt.model, self.opt.n_classes, False)

            elif self.opt.pretrained_on_imagenet == True:
                # print('self.opt.model=%s'%self.opt.model)
                self.classify_model = torchvision.models.__dict__[self.opt.model]

        elif self.opt.network_pkl != None:
            # print(f'load trained classifier from {self.opt.network_pkl}')
            self.classify_model = torch.load(self.opt.network_pkl)
        
        # initilize the loss function
        self.classify_loss = torch.nn.CrossEntropyLoss()

        # initilize the optimizer
        self.classify_optimizer = torch.optim.Adam(self.classify_model.parameters(), lr=self.opt.lr)

        # initilize the dataloader
        self.train_dataloader, self.test_dataloader = self.data()

    def data(self):
        if self.opt.dataset == 'mnist':
            return datasets.dataload.LoadMNIST(self.opt)
        elif self.opt.dataset == 'kmnist':
            return datasets.dataload.LoadKMNIST(self.opt)
        elif self.opt.dataset == 'cifar10':
            return datasets.dataload.LoadCIFAR10(self.opt)
        elif self.opt.dataset == 'cifar100':
            return datasets.dataload.LoadCIFAR100(self.opt)
        elif self.opt.dataset == 'imagenet':
            return datasets.dataload.LoadIMAGENET(self.opt)
        elif self.opt.dataset == 'lsun':
            return datasets.dataload.LoadLSUN(self.opt)
        elif self.opt.dataset == 'stl10':
            return datasets.dataload.LoadSTL10(self.opt)

    def train(self,exp_result_dir):
        print('train classifier')

        epochs = self.opt.epochs
        learn_rate = self.opt.lr
        optimizer = self.classify_optimizer
        classifier = self.classify_model
        classify_loss = self.classify_loss
        train_dataloader = self.train_dataloader
        test_dataloader = self.test_dataloader

        global_train_acc = []
        global_test_acc = []
        global_train_loss = []
        global_test_loss = []

        trainset_total_num = len(train_dataloader.dataset)
        print("train set total num:",trainset_total_num)     #   cifar10 500000
       
        if torch.cuda.is_available():
            classify_loss.cuda()
            classifier.cuda()

        for epoch_index in range(epochs):
            adjust_learning_rate(learn_rate,optimizer,epoch_index)       
            epoch_correct_num = 0
            epoch_total_loss = 0
            for batch_index, (images, labels) in enumerate(train_dataloader):
                imgs = images.cuda()
                labs = labels.cuda()
                optimizer.zero_grad()
                output = classifier(imgs)
                loss = classify_loss(output,labs)
                loss.backward()
                optimizer.step()
                print("[Epoch %d/%d] [Batch %d/%d] [Classify loss: %f] " % (epoch_index, epochs, batch_index, len(train_dataloader), loss.item()))

                #--------在当前训练集batch上的准确率-------------
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index
                batch_same_num = (predicted_label_index == labs).sum().item()     #   当前batch的正确数目
                epoch_correct_num += batch_same_num                                     #   加上本轮之前所有batch的正确数目总和
                epoch_total_loss += loss

            #--------当前epoch分类模型在当前epoch上的准确率-------------            
            epoch_train_accuarcy = epoch_correct_num / trainset_total_num
            print(f'Classifier accuary on the {epoch_index} epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list
            
            if epoch_train_accuarcy > 0.9:
                print(f'epoch index = {epoch_index} epoch_train_accuarcy = {epoch_train_accuarcy}%')

            #--------当前epoch分类模型在当前epoch上的损失-------------                            
            epoch_train_loss = epoch_total_loss / len(train_dataloader)

            print(f'Classifier loss on the {epoch_index} epoch training examples:{epoch_train_loss:.4f}' )   
            global_train_loss.append(epoch_train_loss)

            #--------当前epoch分类模型在测试集整体上的准确率------------- 
            epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(classifier,classify_loss,test_dataloader)
            print(f'{epoch_index} epoch Classifier accuary on the entire testing examples:{epoch_test_accuracy*100:.4f}%' )   
            global_test_acc.append(epoch_test_accuracy)   #   每个epoch训练完后的最新准确率list
            global_test_loss.append(epoch_test_loss)

            if epoch_index % 50 == 0:
                torch.save(classifier,f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-on-clean-{self.opt.dataset}-epoch-{epoch_index:04d}.pkl')

        torch.save(classifier,f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-on-clean-{self.opt.dataset}-finished.pkl')

        accuracy_png_name = f'standard trained classifier {self.opt.model} accuracy on clean {self.opt.dataset}'
        SaveAccuracyCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_acc,global_test_acc,accuracy_png_name)
        
        # accuracy_txt=open(f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}.txt', "w")    
        # accuracy_txt_content = f"{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}-trainset = {global_train_acc[-1]} \n{exp_result_dir}/standard-trained-classifier-{self.opt.model}-accuracy-on-clean-{self.opt.dataset}-testset = {global_test_acc[-1] }"
        # accuracy_txt.write(str(accuracy_txt_content))

        loss_png_name = f'standard trained classifier {self.opt.model} loss on clean{self.opt.dataset}'
        SaveLossCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_loss,global_test_loss,loss_png_name)
        
        # loss_txt=open(f'{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}.txt', "w")    
        # loss_txt_content = f"{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}-trainset = {global_train_loss[-1]} \n{exp_result_dir}/standard-trained-classifier-{self.opt.model}-loss-on-clean-{self.opt.dataset}-testset = {global_test_loss[-1] }"
        # loss_txt.write(str(loss_txt_content))

        return self.classify_model

    def evaluate(self,classify_model,test_dataloader,exp_result_dir):       
        classify_loss= self.classify_loss
        
        if torch.cuda.is_available():
            classify_model.cuda()
            classify_loss.cuda()
        test_accuracy, test_loss = EvaluateAccuracy(classify_model,classify_loss,test_dataloader)
        return test_accuracy, test_loss

    def avtrain(self,exp_result_dir,train_dataloader,test_dataloader):
        print('adversarial train classifier')
        epochs = self.opt.epochs
        learn_rate = self.opt.lr
        
        optimizer = self.classify_optimizer
        classifier = self.classify_model
        classify_loss = self.classify_loss

        global_train_acc = []
        global_test_acc = []
        global_train_loss = []
        global_test_loss = []

        trainset_total_num = len(train_dataloader.dataset)
        print("train set total num:",trainset_total_num)     #   cifar10 500000
       
        if torch.cuda.is_available():
            classify_loss.cuda()
            classifier.cuda()

        for epoch_index in range(epochs):
            adjust_learning_rate(learn_rate,optimizer,epoch_index)       
            epoch_correct_num = 0
            epoch_total_loss = 0
            for batch_index, (images, labels) in enumerate(train_dataloader):
                imgs = images.cuda()
                labs = labels.cuda()
                optimizer.zero_grad()
                output = classifier(imgs)
                loss = classify_loss(output,labs)
                loss.backward()
                optimizer.step()
                print("[Epoch %d/%d] [Batch %d/%d] [Classify loss: %f] " % (epoch_index, epochs, batch_index, len(train_dataloader), loss.item()))

                #--------在当前训练集batch上的准确率-------------
                _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index
                batch_same_num = (predicted_label_index == labs).sum().item()     #   当前batch的正确数目
                epoch_correct_num += batch_same_num                                     #   加上本轮之前所有batch的正确数目总和
                epoch_total_loss += loss

            #--------当前epoch分类模型在当前epoch上的准确率-------------            
            epoch_train_accuarcy = epoch_correct_num / trainset_total_num
            print(f'Classifier accuary on the {epoch_index} epoch training examples:{epoch_train_accuarcy*100:.4f}%' )   
            global_train_acc.append(epoch_train_accuarcy)   #   每个epoch训练完后的最新准确率list
            
            if epoch_train_accuarcy > 0.9:
                print(f'epoch index = {epoch_index} epoch_train_accuarcy = {epoch_train_accuarcy}%')

            #--------当前epoch分类模型在当前epoch上的损失-------------                            
            epoch_train_loss = epoch_total_loss / len(train_dataloader)

            print(f'Classifier loss on the {epoch_index} epoch training examples:{epoch_train_loss:.4f}' )   
            global_train_loss.append(epoch_train_loss)

            #--------当前epoch分类模型在测试集整体上的准确率------------- 
            epoch_test_accuracy, epoch_test_loss = EvaluateAccuracy(classifier,classify_loss,test_dataloader)
            print(f'{epoch_index} epoch Classifier accuary on the adversarial testing examples:{epoch_test_accuracy*100:.4f}%' )   
            global_test_acc.append(epoch_test_accuracy)   #   每个epoch训练完后的最新准确率list
            global_test_loss.append(epoch_test_loss)

            if epoch_index % 50 == 0:
                torch.save(classifier,f'{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-on-augmented-{self.opt.dataset}-epoch-{epoch_index:04d}.pkl')

        torch.save(classifier,f'{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-on-augmented-{self.opt.dataset}-finished.pkl')

        accuracy_png_name = f'adversarial trained classifier {self.opt.model} accuracy on adv {self.opt.dataset}'
        SaveAccuracyCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_acc,global_test_acc,accuracy_png_name)

        # accuracy_txt=open(f'{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-accuracy-on-adv-{self.opt.dataset}.txt', "w")    
        # accuracy_txt_content = f"{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-accuracy-on-adv-{self.opt.dataset}-trainset = {global_train_acc[-1]} \n{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-accuracy-on-adv-{self.opt.dataset}-testset = {global_test_acc[-1] }"
        # accuracy_txt.write(str(accuracy_txt_content))

        loss_png_name = f'adversarial trained classifier {self.opt.model} loss on adv {self.opt.dataset}'
        SaveLossCurve(self.opt.model,self.opt.dataset,exp_result_dir,global_train_loss,global_test_loss,loss_png_name)
        # loss_txt=open(f'{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-loss-on-adv-{self.opt.dataset}.txt', "w")    
        # loss_txt_content = f"{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-loss-on-adv-{self.opt.dataset}-trainset = {global_train_loss[-1]} \n{exp_result_dir}/adversarial-trained-classifier-{self.opt.model}-loss-on-adv-{self.opt.dataset}-testset = {global_test_loss[-1] }"
        # loss_txt.write(str(loss_txt_content))

        return self.classify_model        

    def __getmodel__(self) -> "torchvision.models":

        model_name = self._args.model
        torchvisionmodel_dict = ['resnet34','resnet50','vgg19','densenet169','alexnet','inception_v3']
        if model_name == [].__dict__
        torchvision.models.vgg
        model = self.__gettorchvisionmodel__()
        
        return model

    def __gettorchvisionmodel__(self) ->"torchvision.models":
        model_name = self._args.model
        classes_number = self._args.n_classes
        pretrain_flag = self._args.pretrained_on_imagenet

        torchvisionmodel =  torchvision.models.__dict__[model_name](pretrained=pretrain_flag)
        
        last_name = list(torchvisionmodel._modules.keys())[-1]
        last_module = torchvisionmodel._modules[last_name]
        print('last_name:',last_name)
        print('last_module:',last_module)

        if isinstance(last_module, torch.nn.Linear):
            n_features = last_module.in_features
            torchvisionmodel._modules[last_name] = torch.nn.Linear(n_features, classes_number)
        
        elif isinstance(last_module, torch.nn.Sequential):
            seq_last_name = list(torchvisionmodel._modules.keys())[-1]
            seq_last_module = torchvisionmodel._modules[seq_last_name]
            print('seq_last_name:',seq_last_name)
            print('seq_last_module:',seq_last_module)
            n_features = seq_last_module.in_features
            last_module._modules[seq_last_name] = torch.nn.Linear(n_features, classes_number)
    
        last = list(torchvisionmodel.named_modules())[-1][1]
        print('torchvisionmodel.last:',last)

        return torchvisionmodel

