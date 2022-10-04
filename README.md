# Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift
This repo covers an implementation of the work titled as "Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift". The implementation is in PyTorch. 


# Taxonomical Hierarchy

An example hierarchical representation of a custom subset of ImageNet. The classes for the classification task are at the intermediate level, denoted by `class'. The constituent subpopulations of each class are particular classes from the ImageNet dataset and are marked at the leaf level as `subpopulations'. The labels for these are not shown to the network. The letter `S' denotes `Seen' distribution and `U' denotes 'Unseen' shifted distributions. One-hot labels are provided at each level of the tree. The colored arrows indicate the graphical distance from one leaf node to the other. This shows that mispredicting a Felidae as a Canis (two graph traversals) is less catastrophic than predicting the same as an Salamander (four graph traversals). For illustration we provide the names of one set of subpopulations for each class.


<p align="center">
  <img src="https://github.com/Amitangshu1013/Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift/blob/main/Block.png" width="800">
</p>





# Conditional Training

# Conditional Evaluation
