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

All models are evaluated on accuracy (standard evaluation) and catastrophic co-efficient (conditional evaluation). Please refer to Section III.D for more details on the catastrophic co-efficient. This metric is used to capture the severity of mispredictions of the trained models in the presence and absence of the shift. The lower this quantity is for a model, the better it is. This quantity is computed by aggregrating the Least Common Amcester Distance (LCA) bewteen the correct and mispredicted nodes given a Hierarchy. 

The following files are required for conditional evaluation:
1. The script conditional_evaluation_baseline.py is used to evaluate the flat baseline models. 
2. The script conditional_evaluation.py is used to evaluate the multi-headed hierarchy trained models.

The output of each of these scripts are accuracy and catastrophic co-efficients on seen and unseen (subpopulation shifted) target sets.

# Note

1. The datasets and pre-trained models will be shared upon request. Please look into the Appendix of the ICLR submission for classes used for dataset creaton for Custom Datasets and LIVING-17 A. For hierarchical labels please refer to utils.py, the labels shown in the ICLR submission is not used in the final version of the accepted paper in IEEE Transactions on Artificial Intelligence.
2. Training and testing files are uploaded only for the LIVING-17 A dataset. In the paper we have shown results on four other datasets. The training and evaluation scripts for these datasets will be uploaded soon.
3. The Poster presented at the [Ninth FGVC9 Wokrshop](https://sites.google.com/view/fgvc9), CVPR 2022 is attached here [Poster](https://github.com/Amitangshu1013/Encoding-Hierarchical-Information-in-Neural-Networks-helps-in-Subpopulation-Shift/blob/main/CVRP_2022_FGVC9.pdf).
4. The paper and reviews previously submitted to in ICLR 2022. The paper and reviews from ICLR 2022 are attached [here](https://openreview.net/pdf?id=hJk11f5yfy). 
5. The follwing points have been rectified as per comments from ICLR 2022.
6. Comparisons with Branch CNN paper (related work) has been added in this submission (TAI). 
7. All relevant lierature has been added as per comments. 
8. The submission to ICLR had results on three datasets (LIVING 17 and custom datasets). This submission has results for two more larger subpopulation shift benchmarks. The previous results have also been improved. All models have been trained on 5 random seeds and mean numbers have been reported. Standard deviation results can be provided upon request. 

## Reference
```
@article{mukherjee2023encoding,
  title={Encoding Hierarchical Information in Neural Networks Helps in Subpopulation Shift},
  author={Mukherjee, Amitangshu and Garg, Isha and Roy, Kaushik},
  journal={IEEE Transactions on Artificial Intelligence},
  year={2023},
  publisher={IEEE}
}
```
