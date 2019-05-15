# Experiments May 15, 2019 by Tristan
10 epochs per session

| Architecture | Augmented | Val_acc | Val_loss | Test AUROC | Run time |
| ----- | ----- | --- | --- | --- | --- |
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | True | 0.8271 | 0.3864 | 0.9434 | 1:15:28
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | False | 0.8661 | 0.3136 | - |  0:57:10 | 
DenseNet (2x 64 HU) | True | 0.7448 | 0.51459 | - | 1:16:52
ResNet50 (Empty weights) | True  | No convergence after 1 epoch |
ResNet50 (ImageNet), 5 epochs | True | 0.8304| 0.3985 | - | 3:02:15

## Possible improvements
ConvNet without augmentation seems to perform better on the validation set. I'll test if the performance is better on the test set.

Grand Challenge uses AUC. We should be using AUC as well, as we can use this metric to optimize on.

ResNet without weights does not converge, with ImageNet weights it does converge. Unfortunately, it takes a long time to run and it stops after a certain time.
