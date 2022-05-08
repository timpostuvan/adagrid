# AdaGrid: Adaptive Grid Search for Link Prediction Training Objective

## Overview

This directory contains code necessary to run the AdaGrid algorithm.
AdaGrid can be viewed as an effective automated algorithm for designing machine learning training objectives. Here, AdaGrid is used to improve the performance of graph neural networks (GNNs) for the link prediction task. See our [paper](https://arxiv.org/pdf/2203.16162.pdf) for details on the algorithm.


## Requirements

The code requires among others PyTorch, NumPy, SciPy, sklearn, and NetworkX. You can install all the required packages using the following command:

```
pip install -r requirements.txt
```


To guarantee that you have the right package versions, you can use [Docker](https://docs.docker.com/) to easily set up a virtual environment. See the Docker subsection below for more info.


### Docker

If you do not have [Docker](https://docs.docker.com/) installed, you will need to do so. (Just click on the preceding link, the installation is pretty painless).  

You can run AdaGrid inside a [Docker](https://docs.docker.com/) image. After cloning the project, build and run the image as follows:

```
docker build -t adagrid .
docker run -it adagrid
```

You can also run the above build with GPUs:

```
docker run --gpus=all -it adagrid
```


## Running the code

To replicate and analyze the experiments that are included in the [paper](https://arxiv.org/pdf/2203.16162.pdf), we provide code in subdirectories `uniform-negative-sampling` and `community-ratio-based-negative-sampling`. All the following commands have to be executed from the corresponding subdirectory.


### Uniform negative sampling experiment

To reproduce the uniform negative sampling experiment on Cora dataset using a GNN model with 2 layers and hidden dimension 64, run:

```
python adagrid_uniform_negative_sampling.py --dataset cora --num_layers 2 --hidden_dim 64
```

We also provide an example that runs the experiment and analyzes the results:

```
bash example_uniform_negative_sampling.sh
```


### Community ratio-based negative sampling experiment

To reproduce the community ratio-based negative sampling experiment on Cora dataset using a GNN model with 2 layers and hidden dimension 64, run:

```
python adagrid_community_ratio_based_negative_sampling.py --dataset cora --num_layers 2 --hidden_dim 64
```
    
We also provide an example that runs the experiment and analyzes the results:

```
bash example_community_ratio_based_negative_sampling.sh
```


## Citation

If you make use of this code or the AdaGrid algorithm in your work, please cite the following paper:

```
@article{postuvan2022adagrid,
    title={AdaGrid: Adaptive Grid Search for Link Prediction Training Objective},
    author={Po{\v{s}}tuvan, Tim and You, Jiaxuan and Banaei, Mohammadreza and Lebret, R{\'e}mi and Leskovec, Jure},
    journal={arXiv preprint arXiv:2203.16162},
    year={2022}
}
```