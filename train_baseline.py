import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

import torchvision.models as models

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


val_imageNet = torchvision.datasets.ImageFolder(test_path,transform_test)

val_loader = torch.utils.data.DataLoader(val_imageNet, batch_size=128, shuffle=True, num_workers=8)

# For ResNet-18
cl = models.resnet18()
# For ResNet-34
#cl = models.resnet34()
cl.fc = nn.Linear(512,17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




cl.to(device);


# The standard softmax loss is used for training. All hyperparameters are used as described in the BREEDS paper (Santurkar et al.)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(cl.parameters(), lr = 0.1, momentum=0.9, weight_decay= 1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 150, gamma=0.1)




def save_model(model, filename):
    torch.save(model.state_dict(),filename)



def validate(model, device, set_loader, criterion):
    model.eval()
    corrects = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in set_loader:
            inputs, label = inputs.to(device), labels.to(device)
            output = model(inputs)
            val_loss += criterion(output, label).item()
            val_loss = val_loss/len(set_loader)
            preds = output.max(1, keepdim=True)[1]
            corrects += preds.eq(label.view_as(preds)).sum().item()
        acc = corrects / float(len(set_loader.dataset))
    return val_loss, acc





best_acc = 0.0
for epoch in range(450):
    cl.train()
    count = 0
    for inputs, label in train_loader:
        inputs, label = inputs.to(device), label.to(device)
        optimizer.zero_grad()
        output = cl(inputs)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        count += len(inputs)

    tr_loss, tr_acc = validate(cl, device, train_loader, loss_fn)
    print('Training-----Epoch {}: Loss={:.4f}, Accuracy={:.3f}'.format(epoch, tr_loss, tr_acc))

    val_loss, val_acc = validate(cl, device, val_loader, loss_fn)
    print('Validation------Epoch {}: Loss={:.4f}, Accuracy={:.3f}'.format(epoch, val_loss, val_acc))

    if val_acc > best_acc:
        checkpoint = {'model': cl.state_dict(),
                      'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, 'trial_5_trained_ResNet_18_baseline_best_model.pth')
        best_acc = val_acc

    scheduler.step()


print('Finished Training')