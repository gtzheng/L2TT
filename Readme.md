
## Introduction
This is the code for the SDM 2023 Paper: *Learning to Learn Task Transformations for Improved Few-Shot Classification*.
## Datasets
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view)

- [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view)

Change the values of the two parameters in `config.py` such that they point to the correct datasets.

## Training Script
Metric-based meta-learning:
```
python main.py --head ProtoNet --network ProtoNet --dataset CIFAR_FS --temp 20.0 --step 3 --save-path ./l2tt_experiments --gpu 0 
```
- head could be R2D2, SVM (i.e., MetaOptNet in the paper), ProtoNet
- network could be ResNet, ProtoNet (i.e., CNN64 in the paper)
- step is the maximum policy length (corresponding to L in the paper)
- temp is the sampling temperature (corresponding to $\epsilon$ in the paper)
- dataset could be miniImageNet or CIFAR-FS


Gradient-based meta-learning
```
python train_maml.py --network ProtoNet --dataset CIFAR_FS --temp 20 --step 4 --save-path ./l2tt_experiments --gpu 0
```
- network could be ResNet, ProtoNet (i.e., CNN64 in the paper)
- MAML is supported

## Citation
Please consider citing this paper if you find the code helpful.
```
@inproceedings{zhengSDM23learning,
  title={Learning to Learn Task Transformations for Improved Few-Shot Classification},
  author={Zheng, Guangtao and Suo, Qiuling and Huai, Mengdi and Zhang, Aidong},
  booktitle={SIAM International Conference on Data Mining (SDM)},
  year={2023}
}
```