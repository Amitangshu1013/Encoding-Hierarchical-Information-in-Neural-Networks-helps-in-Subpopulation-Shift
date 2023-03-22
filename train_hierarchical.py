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

# Transforms for Augmentation. The exact same augmentation techniques have been used in the BREEDS paper (Santurkar et al.)

IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
 

transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec']),
        transforms.Normalize(mean=[0.4717, 0.4499, 0.3837],std=[0.2600, 0.2516, 0.2575])
    ])

transform_test = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(mean=[0.4717, 0.4499, 0.3837],std=[0.2600, 0.2516, 0.2575])
])

# These are the hardcoded system paths for the data. Download and store the data as per given by the robustness packagae and then proceed with the correct data path.
# Path for Training Data

train_path = "/local/a/mukher44/Work/Datasets/BREEDS_datasets/Living_17/Regular_unbalanced/Source/"


train_imageNet = torchvision.datasets.ImageFolder(train_path,transform_train)

train_loader = torch.utils.data.DataLoader(train_imageNet, batch_size=128, shuffle=True, num_workers=8)

# Path for validation on the seen to seen distrubtion. The models are saved using the performance of the seen validation set. During training only this test path is used.

test_path = "/local/a/mukher44/Work/Datasets/BREEDS_datasets/Living_17/Regular_unbalanced/Source_Val/"

#val_path = "/local/a/mukher44/Work/Datasets/BREEDS_datasets/Living_17/Regular_unbalanced/Target/"

val_imageNet = torchvision.datasets.ImageFolder(test_path,transform_test)

val_loader = torch.utils.data.DataLoader(val_imageNet, batch_size=128, shuffle=True, num_workers=8)


classes = ['salamander','turtle','fox','domestic cat','bear','beetle','butterfly','ape','monkey','lizard','snake','spider','grouse','parrot','crab','dog','wolf']

superclass = ['bird','amphibian','reptile','arthropod','mammal']
classes_2 = [['dummy6','gallinaceon'],['dummy8'],['serpent','chelonian reptiles','saurian'],['arachnid','crustacean','insect'],['primate','carnivore']]

classes_2_list = ['dummy6','gallinaceon','dummy8','serpent','chelonian reptiles','saurian','arachnid','crustacean','insect','primate','carnivore']

subclasses = [[['parrot'],['grouse']],
             [['salamander']],
             [['snake'],['turtle'],['lizard']],
             [['spider'],['crab'],['beetle','butterfly']],
             [['ape','monkey'],['dog','wolf','fox','domestic cat','bear']]]





net = Hierarchical_resnet18()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device);

criterion_1 = nn.CrossEntropyLoss(reduction="none")
criterion_2 = nn.CrossEntropyLoss(reduction="none")
criterion_3 = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.SGD(cl.parameters(), lr = 0.1, momentum=0.9, weight_decay= 1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 150 , gamma=0.1)
net.to(device);

def validate(model, device, set_loader, criterion):
    model.eval()
    corrects = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in set_loader:
            inputs, label = inputs.to(device), labels.to(device)
            output, output2, output3 = model(inputs)
            val_loss += (criterion(output3, label).mean()).item()
            val_loss = val_loss/len(set_loader)
            preds = output3.max(1, keepdim=True)[1]
            corrects += preds.eq(label.view_as(preds)).sum().item()
        acc = corrects / float(len(set_loader.dataset))
    return val_loss, acc


for epoch in range(450):
    net.to(device);
    net.train()
    count = 0
    running_loss = 0.0
    running_loss_2 = 0.0
    running_loss_3 = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        head_1_labels = []
        head_2_labels = []
        # This block converts the one-hot encoded labels into hierarchical labels using functions from utils
        # These hierarchical labels are used to train the Hierarchical-18 model 
        for j in range(0,len(labels)):
            groundtruth = classes[labels[j]]
            h1, h2, h3 = get_hier_labels(groundtruth)
            head_1_real = get_level1(h1)
            head_2_real = get_level2(h1,h2)
            head_1_labels.append(head_1_real)
            head_2_labels.append(head_2_real)
        labels1 = torch.tensor(head_1_labels, dtype = torch.long)
        labels2 = torch.tensor(head_2_labels, dtype = torch.long)
        inputs, labels1, labels2, labels3 = inputs.to(device), labels1.to(device), labels2.to(device), labels.to(device) 

        # zero the parameter gradients
        optimizer.zero_grad()

        # This marks the start of the conditional training block
        # forward + backward + optimize 
        # forward pass for all 3 heads

       
        head_1, head_2, head_3 = net(inputs)

        # calculate individual loss for head 1
        loss1 = criterion_1(head_1, labels1)
        loss1 = loss1.mean()
        
        loss1.backward()

        optimizer.step()

        optimizer.zero_grad()

        head_1, head_2, head_3 = net(inputs)
        loss2 = criterion_2(head_2, labels2)

        # make predictions
        _, head_1_predicted = torch.max(head_1, 1)

        # check predictions (validity matrix) for head 2 
        assign_1 = (head_1_predicted == labels1)

        # convert bool to int 
        res_1 = assign_1.type(torch.int8)

        # calculate conditional loss for head 2
        final_loss_2 = loss2 * res_1
        final_loss_2 = final_loss_2.mean()
        final_loss_2.backward()

        optimizer.step()

        # zero the parameter gradients
        optimizer.zero_grad()

        head_1, head_2, head_3 = net(inputs)


        # calculate individual loss
        loss3 = criterion_3(head_3, labels3)

        # make predictions
        _, head_1_predicted = torch.max(head_1, 1)
        _, head_2_predicted = torch.max(head_2, 1)


        # calculate conditional loss
        # check predictions and calculate validity matrix for head 3
        assign_1 = (head_1_predicted == labels1)
        assign_2 = (head_2_predicted == labels2)


        # convert bool to int 
        res_1 = assign_1.type(torch.int8)
        res_2 = assign_2.type(torch.int8)

        # Validity for head 3
        prod_1 = res_1 * res_2


        # calculate final losses for head 3
        final_loss_3 = loss3 * prod_1
        final_loss_3 = final_loss_3.mean()

        # backward and optimize
        final_loss_3.backward()
        optimizer.step()

        L1 = loss1
        L2 = final_loss_2
        L3 = final_loss_3
        


        # print statistics
        running_loss += L1.item()
        running_loss_2 += L2.item()
        running_loss_3 += L3.item()



    

    print('Training-----Epoch {}: Loss_1={:.4f}'.format(epoch, running_loss/len(train_loader)))
    print('Training-----Epoch {}: Loss_2={:.4f}'.format(epoch, running_loss_2/len(train_loader)))
    print('Training-----Epoch {}: Loss_3={:.4f}'.format(epoch, running_loss_3/len(train_loader)))

    val_loss, val_acc3 = validate(net, device, val_loader, criterion_3)
    print('Validation------Epoch {}: Accuracy_Head3={:.3f}'.format(epoch, val_acc3))


    if val_acc3 > best_acc:
        checkpoint = {'model': net.state_dict(),
                      'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, 'trial_5_BREEDS_17_best_Hierarchical_ResNet_18.pth')
        best_acc = val_acc3

    

    scheduler.step()


print('Finished Training')
