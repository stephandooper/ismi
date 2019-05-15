# Experiments May 15, 2019 by Tristan
10 epochs per session

| Architecture | Augmented | Val_acc | Val_loss | Test AUROC | Run time |
| ----- | ----- | --- | --- | --- | --- |
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | True | 0.8271 | 0.3864 | 0.9434 | 1:15:28
[ConvNet](https://arxiv.org/pdf/1902.06543.pdf) | False | 0.8661 | 0.3136 | - |  0:57:10 | 
DenseNet (2x 64 HU) | True | 0.7448 | 0.51459 | - | 1:16:52
ResNet50 (Empty weights) | True  | No convergence after 1 epoch |
ResNet50 (ImageNet), 5 epochs | True | 0.8304| 0.3985 | - | 3:02:15
## ConvNet
> The architecture of such CNN classifiers consisted of 9layers of strided convolutions with 32, 64, 64, 128, 128,256, 256, 512 and 512 3x3 filters, stride of 2 in the evenlayers, BN and LRA; followed by global average pooling;50% dropout; a dense layer with 512 units, BN and LRA;and a linear dense layer with either 2 or 9 units depend-ing on the classification task, followed by a softmax.  Weapplied L2 regularization with a factor of 1×10−6.We minimized the cross-entropy loss using stochasticgradient descent with Adam optimization and 64-sampleclass-balanced mini-batch, decreasing the learning rate bya factor of 10 starting from 1×10−2every time the valida-tion loss stopped improving for 4 consecutive epochs until1×10−5.  Finally, we selected the weights correspondingto the model with the lowest validation loss during train-ing. - [D. Tellez et al.](https://arxiv.org/pdf/1902.06543.pdf)

## Possible improvements
ConvNet without augmentation seems to perform better on the validation set. I'll test if the performance is better on the test set.

Grand Challenge uses AUC. We should be using AUC as well, as we can use this metric to optimize on.

ResNet without weights does not converge, with ImageNet weights it does converge. Unfortunately, it takes a long time to run and it stops after a certain time.
