# Meeting 2019-04-19
**What is the challenge?**
Many people work on Camelyon, Patch Camelyon is less popular. Finding large clusters of tumour cells is not difficult. Finding isolated tumour cells can be difficult.
Pathologists are good at identifying isolated tumour cells, however they don’t have time to do this.
There are treatment consequences if the outcome of the research by the pathologist is different. If we find tumorous cells, then the treatment would be different. Adding neural networks can speed up the process of the research, resulting in faster and better treatment.

**Staining**
Camelyon16 has two different labs; Camelyon17 has five different labs. Our dataset has differences between labs, as the scanners in Utrecht and Nijmegen give different results. In current research, we try to make the network robust for the variations by doing stain augmentation, by re-staining the images strongly.
Blurring; sharpening; contrast enhancement. Images are often slightly blurred due to the microscope that introduces blur. Try to make incremental steps to find out what works. Different augmentation approaches can have different effects on different networks.

**CapsNet, U-Net and other networks**
If you have a feeling that it can work, then try it! Jeroen will have a look at the paper.
Pre-training seems to help. Try to re-implement what people did. Try to tinker with values and layers. Try to first start with an implementation that seems to work for others. 

**Best Camelyon16 challenge**
See the Google paper! Ratio of 1:4 (tumour to normal) to get the best results, which is one of the variables that you should play with.

**Centre 32x32 pixels**
A positive label indicates that the centre 32x32px region of a patch contains at least one pixel of tumour tissue. Tumour tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behaviour when applied to a whole-slide image.

**Hyperparameter optimization**
Three methods: (1) Hyperparameter optimization pipeline (Tpot), (2) Range of possibly working hyperparameters, (3) Try every value itself. Jeroen advises a combination of 2 and 3.

**Sacred**
You can make a centralized database (MongoDB). Whenever you run an experiment, the hyperparameters are uploaded to the database, including results. Klaus will take a look at it.

**Final Score vs Scientific Contribution**
You’re not improving the world. Trying new ways for us to experiment, where you can run into obstacles, and thus learn from, then it would be nice, as you would learn much. Try to do the basis first, as this can be difficult as well.

Next meeting: Tuesday April 30, 10:30 - 11:30. Goal: present first results
Tuesday May 7: 13:00 - 14:00 
Tuesday May 14: 12:00 - 13:00 (Skype)
Tuesday May 21: 13:00 - 14:00 

