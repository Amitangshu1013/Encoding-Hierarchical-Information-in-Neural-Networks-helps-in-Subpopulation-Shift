#Author Amitangshu Mukherjee

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.optim as optim
from Hierarchical_model import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.4717, 0.4499, 0.3837],std=[0.2600, 0.2516, 0.2575])
])



# Load Model

PATH = './trial_5_BREEDS_17_best_Hierarchical_ResNet_18.pth'

net = Hierarchical_resnet18()

checkpoint = torch.load(PATH)

net.load_state_dict(checkpoint['model'])


net.eval();

class_list = ['salamander','turtle','fox','domestic cat','bear','beetle','butterfly','ape','monkey','lizard','snake','spider','grouse','parrot','crab','dog','wolf']

superclass = ['bird','amphibian','reptile','arthropod','mammal']
classes_2 = [['dummy6','gallinaceon'],['dummy8'],['serpent','chelonian reptiles','saurian'],['arachnid','crustacean','insect'],['primate','carnivore']]

subclasses = [[['parrot'],['grouse']],
             [['salamander']],
             [['snake'],['turtle'],['lizard']],
             [['spider'],['crab'],['beetle','butterfly']],
             [['ape','monkey'],['dog','wolf','fox','domestic cat','bear']]]


def get_hier_groundtruth(ground_truth):
    for x,y in enumerate(subclasses):
        for i,j in enumerate(y):
            for a,b in enumerate(j):
                if ground_truth == b:
                    global head1_gt 
                    head1_gt = x
                    global head2_gt 
                    head2_gt = i
                    global head3_gt
                    head3_gt = a
                    
    return head1_gt, head2_gt, head3_gt


# Add paths to seen source validation set and unseen target set


val_path = "/local/a/mukher44/Work/Datasets/BREEDS_datasets/Living_17/Regular_unbalanced/Source_Val/"

target_path = "/local/a/mukher44/Work/Datasets/BREEDS_datasets/Living_17/Regular_unbalanced/Target/"


val_imageNet = torchvision.datasets.ImageFolder(val_path,transform)

valloader = torch.utils.data.DataLoader(val_imageNet, batch_size=100, shuffle=True, num_workers=4)


target_imageNet = torchvision.datasets.ImageFolder(target_path,transform)

targetloader = torch.utils.data.DataLoader(target_imageNet, batch_size=100, shuffle=True, num_workers=4)



correct_1 = 0
correct_2 = 0
correct_3 = 0
total = 0
cat_1 = 0
cat_2 = 0
cat_3 = 0





with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs1, outputs2, outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(0,valloader.batch_size):
            preds = class_list[predicted[j]]
            groundtruth = class_list[labels[j]]
            gt_1, gt_2, gt_3 = get_hier_groundtruth(groundtruth)
            pred_1, pred_2, pred_3 = get_hier_groundtruth(preds)
            
            #Head_1
            if (pred_1 == gt_1):
                correct_1 = correct_1 + 1
            if (pred_1 != gt_1):
                cat_1 = cat_1 + 6
            
            #Head_2
            if (pred_1 == gt_1) and (pred_2 == gt_2):
                correct_2 = correct_2 + 1
            if (pred_1 == gt_1) and (pred_2 != gt_2):
                cat_2 = cat_2 + 4
                
            #Head_3    
            if (pred_1 == gt_1) and (pred_2 == gt_2) and (pred_3 == gt_3):
                correct_3 = correct_3 + 1    
            if (pred_1 == gt_1) and (pred_2 == gt_2) and (pred_3 != gt_3):
                cat_3 = cat_3 + 2
            
            
        total += labels.size(0)        

        
        

total_cat = cat_1 + cat_2 + cat_3

print("\n")
print("For  Source Set......")




print('Accuracy of the network (head_1) on the  test images: %.2f %%' % (
    100 * correct_1 / total))

print('Accuracy of the network (head_2) on the test images: %.2f %%' % (
    100 * correct_2 / total))


print('Accuracy of the network (head_3) on the  test images: %.2f %%' % (
    100 * correct_3 / total))



print(total_cat/total)




correct_1 = 0
correct_2 = 0
correct_3 = 0
total = 0
cat_1 = 0
cat_2 = 0
cat_3 = 0





with torch.no_grad():
    for data in targetloader:
        images, labels = data
        outputs1, outputs2, outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(0,targetloader.batch_size):
            preds = class_list[predicted[j]]
            groundtruth = class_list[labels[j]]
            gt_1, gt_2, gt_3 = get_hier_groundtruth(groundtruth)
            pred_1, pred_2, pred_3 = get_hier_groundtruth(preds)
            
            #Head_1
            if (pred_1 == gt_1):
                correct_1 = correct_1 + 1
            if (pred_1 != gt_1):
                cat_1 = cat_1 + 6
            
            #Head_2
            if (pred_1 == gt_1) and (pred_2 == gt_2):
                correct_2 = correct_2 + 1
            if (pred_1 == gt_1) and (pred_2 != gt_2):
                cat_2 = cat_2 + 4
                
            #Head_3    
            if (pred_1 == gt_1) and (pred_2 == gt_2) and (pred_3 == gt_3):
                correct_3 = correct_3 + 1    
            if (pred_1 == gt_1) and (pred_2 == gt_2) and (pred_3 != gt_3):
                cat_3 = cat_3 + 2
            
            
        total += labels.size(0)        

        
        

total_cat = cat_1 + cat_2 + cat_3

print("\n")
print("For  Target Set......")


print('Accuracy of the network (head_1) on the  test images: %.2f %%' % (
    100 * correct_1 / total))

print('Accuracy of the network (head_2) on the test images: %.2f %%' % (
    100 * correct_2 / total))


print('Accuracy of the network (head_3) on the  test images: %.2f %%' % (
    100 * correct_3 / total))


print(total_cat/total)

