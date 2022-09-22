# Is Complexity Required for Neural Network Pruning? A Case Study on Global Magnitude Pruning
This GitHub repository is the official repository for the paper "Is Complexity Required for Neural Network Pruning? A Case Study on Global Magnitude Pruning". 

## Set Up
0. Clone this repository.
1. Using `Python 3.6.9`, create a virtual environment `venv` with  `python -m venv myenv` and run `source myenv/bin/activate`.
2. Install requirements with `pip install -r requirements.txt` for `venv`.
3. Create a folder which has LabelSmoothing.py, prune.py (or prune_withMT.py), model_list.py, and the base model. 

## Training with Global MP and Global MP with MT
To run the global magnitude pruning without minimum threshold (MT), run the prune.py file. To run the global magnitude pruning with MT, run the prune_withMT.py file. 

Note - you should change the base model's location and the dataset's location in the the prune.py and prune_withMT.py files before running them. 

To run the prune.py file, run the command-
```
python3 prune.py
```
To run the prune_withMT.py file, run the command-
```
python3 prune_withMT.py
```

### Dense Model:

This model is the base model that we used for our ResNet-50 on ImageNet experiments.

| Architecture | Parameters | Sparsity (%) | Top-1 Acc (%) | Model Links |
| ------------ | :--------: | :----------: | :-----------: | :---------: |
| Resnet-50        | 25.50M  | 0.00        | 77.04         | [Base Model](https://drive.google.com/file/d/1I7dxZD87-Ftav-BvIxqCWCWGWqYZFVK2/view?usp=sharing) |