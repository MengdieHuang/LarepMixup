from attacks.advattack import AdvAttackClassifier
import torch 

def ClaAttack(args,exp_result_dir,target_classifier):
    print("Flag: attacking trained classifier !")

    attack_classifier = AdvAttackClassifier(args,target_classifier)
    torch.save(attack_classifier.art_estimator_model,f'{exp_result_dir}/attack-classifier-artmodel-{args.model}-on-{args.dataset}.pkl')

    # adv_dataloader = attack_classifier.generate()
    aug_train_dataloader = attack_classifier.generate(exp_result_dir)
    attack_classifier.evaluate(exp_result_dir)

    return attack_classifier,aug_train_dataloader