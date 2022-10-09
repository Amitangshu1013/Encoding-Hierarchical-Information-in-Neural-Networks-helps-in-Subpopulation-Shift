# Encoding Hierarchical Information in Neural Networks helps in Subpopulation Shift
This repo covers an implementation of the work titled as "Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift". The implementation is in PyTorch. 


# Taxonomical Hierarchy

An example hierarchical representation of a custom subset of ImageNet. The classes for the classification task are at the intermediate level, denoted by "Class". The constituent subpopulations of each class are particular classes from the ImageNet dataset and are marked at the leaf level as "Subpopulations". The labels for these are not shown to the network. The letter 'S' denotes `Seen' distribution and 'U' denotes 'Unseen' shifted distributions. One-hot labels are provided at each level of the tree. The colored arrows indicate the graphical distance from one leaf node to the other. This shows that mispredicting a Felidae as a Canis (two graph traversals) is less catastrophic than predicting the same as an Salamander (four graph traversals). For illustration we provide the names of one set of subpopulations for each class.


<p align="center">
  <img src="https://github.com/Amitangshu1013/Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift/blob/main/Block.png" width="800">
</p>


# Conditional Training

Practical implementation of conditional training for a batch of images. The validity mask serves to ensure that the blocks corresponding to a particular level are trained only on the instances that are correctly classified at the previous level.  Instead of blocking representations by multiplying with zeros, we implement conditional training via multiplication of losses with the corresponding validity masks, resulting in the same outcome. Validity masks $V_{l_1-l_2}$ represent the propagation of correctly classified instances from level $l_1$ to $l_2$, and contain a 1 where the instance was correctly classified by all levels between $l_1$ and $l_2$ and 0 otherwise. They can be built from the composition of several validity masks. For instance, as shown in the figure, the validity mask for propagation from level 1 to level 3 is calculated by multiplying the validity mask from level 1 to level 2 with the validity mask from level 2 to level 3.

<p align="center">
  <img src="https://github.com/Amitangshu1013/Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift/blob/main/valmask-1.png" width="800">
</p>

The following scripts are required for conditional training:
1. Hierarchical_model.py contains the modified three headed ResNet-18 model.
2. Functions required to convert one-hot encoded labels into hierarchical labels can be found in utils.py
3. To train a multi-headed architecture with the proposed algorithm use train_hierarchical.py
4. To train a baseline architecture with standard supervised training use train_baseline.py
5. For Hyper-parameters, details of subpopulation shift benchmarks and training details please follow the [BREEDS Paper](https://openreview.net/pdf?id=mQPBmvyAuk)


# Conditional Evaluation
