# Toward Explainable Artificial Intelligence for Regression Models: A methodological perspective

![Alt text](./figures/figure_1_overview.png)

## About
Welcome to xai-regression. We think of explainable artificial intelligence (XAI) as crucial for an informed application of machine learning methods in practice. Although XAI techniques have reached significant popularity for classifiers, little attention has been devoted to XAI for regression models so far. With our IEEE SPM publication [*Toward Explainable Artificial Intelligence for Regression Models: A methodological perspective*](https://ieeexplore.ieee.org/document/9810062) (open pre-print available [here](https://arxiv.org/abs/2112.11407)) and the corresponding [website](xai-regression.org) we aim to address this gap and provide an overview of the latest developments of the field.

This repository contains implementations of the restructuring approach presented in the above mentioned paper.

If you are using the code, please cite it as:
```sh
@ARTICLE{Letzgus_XAIR_2022,
  author={Letzgus, Simon and Wagner, Patrick and Lederer, Jonas and Samek, Wojciech and Müller, Klaus-Robert and Montavon, Grégoire},
  journal={IEEE Signal Processing Magazine}, 
  title={Toward Explainable Artificial Intelligence for Regression Models: A methodological perspective}, 
  year={2022},
  volume={39},
  number={4},
  pages={40-58},
  doi={10.1109/MSP.2022.3153277}}
```


## Getting Started

The [restructuring_implementation.ipynb](./restructuring_implementation.ipynb) contains functions implementing the proposed *flooding-rule* (**find_a_ref()**) for latent offset selection and the introduced restructuring of ANN top-layers (**restructure_model()**) alongside the respective parts of the paper. 




### Prerequisites

The required libraries are listed in the requirements.txt-file.

## Usage
The functions can be used to explain your model output relative to custom-specific reference values using existing propagation-based XAI techniques, such as LRP (pytorch-implementation available in the [Zennit](https://github.com/chr5tphr/zennit) library). 


## Contact
simon.letzgus@tu-berlin.de 

Machine Learning Group, Technische Universität Berlin, Straße des 17. Juni 135, Berlin, 10623, Germany.
