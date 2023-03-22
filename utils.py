
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim



classes = ['salamander','turtle','fox','domestic cat','bear','beetle','butterfly','ape','monkey','lizard','snake','spider','grouse','parrot','crab','dog','wolf']

superclass = ['bird','amphibian','reptile','arthropod','mammal']
classes_2 = [['dummy6','gallinaceon'],['dummy8'],['serpent','chelonian reptiles','saurian'],['arachnid','crustacean','insect'],['primate','carnivore']]

classes_2_list = ['dummy6','gallinaceon','dummy8','serpent','chelonian reptiles','saurian','arachnid','crustacean','insect','primate','carnivore']

subclasses = [[['parrot'],['grouse']],
             [['salamander']],
             [['snake'],['turtle'],['lizard']],
             [['spider'],['crab'],['beetle','butterfly']],
             [['ape','monkey'],['dog','wolf','fox','domestic cat','bear']]]




def get_hier_labels(ground_truth):            
    for s,t in enumerate(subclasses):
        for u,v in enumerate(t):
            for m,n in enumerate(v):
                if ground_truth == n:
                    global head1_gt
                    head1_gt = s
                    global head2_gt 
                    head2_gt = u
                    global head3_gt 
                    head3_gt = m
    
    return head1_gt, head2_gt, head3_gt

def get_level1(gt_head1):
    superclass_name = superclass[gt_head1]
    head_1 = superclass.index(superclass_name)
    return head_1


def get_level2(gt_head1,gt_head2):
    class_name = classes_2[gt_head1][gt_head2]
    head_2 = classes_2_list.index(class_name)
    return head_2
