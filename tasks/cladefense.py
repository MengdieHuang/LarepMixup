

def ClaAdvTrain(exp_result_dir,target_classifier,aug_train_dataloader,adv_test_dataloader):
    print('adversarial training classifier')

    av_target_classifier = target_classifier
    av_target_classifier.avtrain(exp_result_dir,aug_train_dataloader,adv_test_dataloader)
