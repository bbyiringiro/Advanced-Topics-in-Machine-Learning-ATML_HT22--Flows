# Variational Inference with Normalizing Flows 

**ATML Group 9 - HT22**

Repository containing code for the reproducibility challenge set as an exam in the ATML HT22 course. A reproduction of the results of the [original paper](https://arxiv.org/abs/1505.05770) by Danilo Jimenez Rezende and Shakir Mohamed. 

The code for the Sylvester flows in `flows.py` is adapted from https://github.com/riannevdberg/sylvester-flows.

Examples of how to train and evaluate the models are found in the Jupyter notebooks `train_MNIST` and `train_CIFAR`.


|![Model architecture](images/arch/architecture.png)
|:--:| 
| *Model architecture* |




|![Comparasion Flows Impact at differnt flow Length](images/comparison_g.png)
|:--:| 
| *Effect of the flow-length on MNIST* |

| ![Visualising the impact of applying Normilising Flows](images/figure_1/full.png)
|:--:| 
| *Effect of planar and radial normalising flows on two standard distributions* |

| ![Flows ability to fit any complex distributions](images/figure_3/figure_3.svg)
|:--:| 
|*Approximating complex 2D distributions* |

| ![reconstructions of images through the flows](images/recon/recon.png)
|:--:| 
| *Original input $`{x}`$ and the reconstructed $`\hat{x}`$* |


