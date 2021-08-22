"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import torch
from clamodels.classifier import Classifier


def ClaTrain(args,exp_result_dir):
    print("Flag: standard training classifier !")

    # luachering training classifier task
    target_classifier = Classifier(args)
    target_classifier.train(exp_result_dir) 

    # target_classifier.classify_model 

    print("Training *%s* classifier finished !" % args.model)

    target_classifier.evaluate(exp_result_dir)
    print("Testing *%s* classifier finished !" % args.model)

    return target_classifier