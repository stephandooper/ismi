# ReCNN parameters
ReCNN is a DenseNet architecture (based on [Huang et al.](https://arxiv.org/pdf/1608.06993.pdf)). It adds rotation equivariance to the architecture, hence the name.

## Terminology

The conv group (essentially the type of rotation equivariance) parameter has a confusing naming scheme.

| Code               | Paper  | Meaning            
|----|--|---------------------|
| C4 | P4 | 4 rotations |
| D4 | P4M | 8 transformations (4 rotations * mirroring) |




## Parameters (in the code)

| Parameter               | Description                                            | Default             | Comment |
|-------------------------|--------------------------------------------------------|---------------------|--------|
| depth                   | Total number of layers in the DenseNet                 |          40         |    If -**None**, computed from nb_dense_block and nb_layers_per_block, this is preferable   |
| nb_dense_block          | Dense blocks added to the end                          |          3          |    Veeling seems to be using **5** dense blocks throughout    |
| nb_filter               | Initial number of filters (in the dense block?)        |   2 * growth_rate   |  **12**    |
| growth_rate (k)         | How many filters are added to the number per block     |          12         |  24, 12 for C4, 8 for D4 |
| nb_layers_per_block     | How many layers within each dense block                | Computed from depth |    Veeling uses **1** layer per dense block    |
| bottleneck              | If True, bottleneck blocks will be added               |        False        |    Nothing mentioned about this by Veeling; looks like **True**   |
| reduction               | Factor within transition blocks                        |         0.0         |    Nothing mentioned; use **0.33**  |
| dropout_rate            | Dropout (where?)                                       |         0.0         |    Nothing mentioned, so keep it **0.0**    |
| weight_decay            | Weight decay (where?)                                  |         1e-4        |    Notthing mentioned, so keep the **default**   |
| subsample_initial_block | True for ImageNet, False for CIFAR                     |        False        |    Nothing mentioned, seems to be size-related, so **False** for us as we're closer to CIFAR than ImageNet    |
| include_top             | Fully-connected layer at the top?                      |         True        |    Nothing mentioned; looks more like **False**  |
| classes                 | Number of classes                                      |          10         |   Can be ignored, fixed to **1**     |
| activation              | Activation function at the final FC layer              |       softmax       |   Veeling seems to be using a **sigmoid** activation according to the paper     |
| use_gcnn                | Whether to use rotation-invariant CNN layers           |       False         |    If this is switched off, we end up with a normal DenseNet architecture    |
| conv_group              | What kind of transformations to use with RE conv       |         C4          |    Veeling: **D4** somewhat better ( + ~2 % points gain)     |
