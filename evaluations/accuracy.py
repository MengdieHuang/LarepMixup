"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import torch
import torch.utils.data

def EvaluateAccuracy(classifier, classify_loss, test_dataloader:torch.utils.data.DataLoader, cla_model_name):
    classifier.eval()     #   eval mode
    #   计算准确率
    testset_total_num = len(test_dataloader.dataset)
    # print("flag B test set total_num:",testset_total_num)                                                       #   test set total_num: 10000 样本总数
    # print("flag B test_dataloader.len:",len(test_dataloader))                                                   #   test_dataloader.len: 40 batch数目（batchsize256）

    # print("flag B test_dataloader.dataset[0][0][0]:",test_dataloader.dataset[0][0][0])                                #   归一化的
    # print("flag B test_dataloader.dataset[0][1]:",test_dataloader.dataset[0][1])                                  #   
    # print("flag B test_dataloader.dataset[1][1]:",test_dataloader.dataset[1][1])                                  #   
    # print("flag B test_dataloader.dataset[2][1]:",test_dataloader.dataset[2][1])                                  #   

    epoch_correct_num = 0
    epoch_total_loss = 0
    
    # print("test_dataloader.dataset.data:",test_dataloader.dataset.data[:2])

    for batch_index,(images, labels) in enumerate(test_dataloader):
        # if batch_index == 0:
        #     print("flag B images[0][0]:",images[0][0])                                                                       #     images.shape: torch.Size([256, 3, 32, 32])
        #     print("flag B labels[0]:",labels[0])                                                                         #   labels.shape: torch.Size([256])

        imgs = images.cuda()
        labs = labels.cuda()
        
        # if batch_index == 0:
        #     print("flag B imgs[0][0]:",images[0][0])                                                                        #   归一化的 但和test_dataloader.dataset[0][0]不一样
        #     print("flag B labs[0]:",labs[0])                                                                             #   labs.shape: torch.Size([256])

        
        with torch.no_grad():

            #   output = classifier(imgs)
            #--------maggie 20220722---------
            # print("images[0].shape:",images[0].shape)
            if images[0].shape == (3,256,256):                          #   表明是 ImageNetMixed 10
                output = classifier(imgs, imagenetmixed10=True)
            else:
                output = classifier(imgs)
            #--------------------------------

            if cla_model_name == 'inception_v3':
                output, aux = classifier(imgs)
            elif cla_model_name == 'googlenet':
                if images.size(-1) == 256:  #   只有imagenet搭配googlenet时是返回一个值
                    output = classifier(imgs)
                else:
                    output, aux1, aux2 = classifier(imgs)
            else:
                output = classifier(imgs)         
            
            
            # print("output:",output)                                     #   output: tensor([[-3.9918e+00, -4.0301e+00,  6.1573e+00,  ...,  1.5459e+00
            # print("output.shape:",output.shape)                         #   output.shape: torch.Size([256, 10])
            # softmax_output = torch.nn.functional.softmax(output, dim = 1)
            # print("softmax_output:",softmax_output)                     #   softmax_output: tensor([[2.6576e-05, 2.5577e-05, 6.7951e-01,  ..., 6.7526e-03, 4.7566e-05,,
            # print("softmax_output.shape:",softmax_output.shape)         #   softmax_output.shape: torch.Size([256, 10])              
            # raise Exception("maggie error 20210906")

            loss = classify_loss(output,labs)
            _, predicted_label_index = torch.max(output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
            
            # loss = classify_loss(softmax_output,labs)
            # _, predicted_label_index = torch.max(softmax_output.data, 1)    #torch.max()这个函数返回的是两个值，第一个值是具体的value，第二个值是value所在的index   
                        
            
            # print("predicted_label_index:",predicted_label_index)                                                           #   predicted_label_index: tensor([1, 4, 0, 6, 0, 0, 7, 8, 0, 3, 3, 0, 7, 4, 9, 3], device='cuda:0')
            # print("labs:",labs)                                                                                             #   labs: tensor([1, 6, 8, 2, 8, 8, 5, 0, 1, 1, 9, 0, 7, 4, 1, 2], device='cuda:0')
            batch_same_num = (predicted_label_index == labs).sum().item()
            epoch_correct_num += batch_same_num
            epoch_total_loss += loss

    #------当前epoch分类模型在测试集整体上的准确率--------------------(测试集只用1轮)
    test_accuracy = epoch_correct_num / testset_total_num
    test_loss = epoch_total_loss / len(test_dataloader)                                                         #   除以batch num
    # print("测试样本总数：",testset_total_num)
    # print("预测正确总数：",epoch_correct_num)
    # print("预测总损失：",epoch_total_loss)
    return test_accuracy,test_loss 
    