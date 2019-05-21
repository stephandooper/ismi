# DenseNet parameters

## Terminology

The conv group parameter has a confusing naming scheme.

| Code               | Paper  | Meaning            
|----|--|---------------------|
| C4 | P4 | 4 rotations |
| D4 | P4M | 8 transformations (4 rotations * mirroring) |




## Parameters (in the code)

| Parameter               | Description                                            | Default             | Effect |
|-------------------------|--------------------------------------------------------|---------------------|--------|
| depth                   | Total number of layers in the DenseNet                 |          40         |        |
| nb_dense_block          | Dense blocks added to the end                          |          3          |        |
| nb_filter               | Initial number of filters (in the dense block?)        |   2 * growth_rate   |        |
| growth_rate             | How many filters are added per to the number per block |          12         |        |
| nb_layers_per_block     | How many layers within each dense block                | Computed from depth |        |
| bottleneck              | If True, bottleneck blocks will be added               |        False        |        |
| reduction               | Factor within transition blocks                        |         0.0         |        |
| dropout_rate            | Dropout (where?)                                       |         0.0         |        |
| weight_decay            | Weight decay (where?)                                  |         1e-4        |        |
| subsample_initial_block | True for ImageNet, False for CIFAR                     |        False        |        |
| include_top             | Fully-connected layer at the top?                      |         True        |        |
| weights                 | None or Imagenet (but fails for < 1000 classes)        |         None        |        |
| classes                 | Number of classes                                      |          10         |        |
| activation              | Activation function at the final FC layer              |       softmax       |        |
| use_gcnn                | Whether to use rotation-invariant CNN layers           |       False         |        |
| conv_group              | What kind of transformations to use with RE conv       |         C4          |        |
