# Code for "Tab-semiSL: Tabular Data-Driven Semi-Supervised Learning to Identify Factors Associated with Immune-Related Adverse Events"

Authors: Anonymous

This directory contains implementations of Tab-semiSL framework using MNIST dataset.

## Command inputs:

-   iterations: Number of experiments iterations
-   label_no: Amount of labeled data to be used
-   model_name: Supervised model name (e.g., xgboost or catboost)
-   p_m: Perturbation probability for masked
-   alpha: Hyper-parameter to control the mask loss
-   beta: Hyper-parameter to control the contrastive loss
-   gamma: Hyper-parameter to control the unsupervised loss
-   label_data_rate: Ratio of labeled data
-   datasets_name: Selected datasets name

Note that hyper-parameters should be optimized for different datasets.

## Example
First, you need to create a conda environment.

```shell
$ conda create -n Tab-semiSL python=3.11
$ conda activate Tab-semiSL
```

Then, ensure that the path leads to the Tab-semiSL root directory and install requirements,

```shell
$ cd Tab-semiSL-master
$ pip install -r requirements.txt
```
run main.py

```shell
$ python main.py
```

you also can change hyper-parameters. For example:
```shell
$ python main.py --iterations 10 --label_no 1000 --model_name xgboost
--p_m 0.5 --alpha 2.0 --beta 0.1 --gamma 0.1 --label_data_rate 0.1 --datasets_name MNIST 
```

