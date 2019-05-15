# Experiments May 15, 2019 by Tristan
10 epochs per session

| Architecture | Augmented | Val_acc | Val_loss | Test AUROC | Run time |
| ----- | ----- | --- | --- | --- | --- |
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | True | 0.8271 | 0.3864 | 0.9434 | 1:15:28
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | False | 0.8661 | 0.3136 | - |  0:57:10 | 
DenseNet (2x 64 HU) | True | 0.7448 | 0.51459 | - | 1:16:52
ResNet50 (Empty weights) | True  | No convergence |
ResNet50 (ImageNet) | True | 0.8304| 0.3985 | - | 3:02:15

