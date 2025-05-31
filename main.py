import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from torch import tensor


path = Path('titanic')
if not path.exists():
    import zipfile,kaggle
    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f'{path}.zip').extractall(path)

path = Path('titanic')
df = pd.read_csv(path/'train.csv')
df.fillna(df.mode().iloc[0], inplace=True)
df = pd.get_dummies(df, columns=["Pclass", "Sex", "Embarked"])
df['LogFare'] = np.log(df['Fare']+1)
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare', 'Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
dep_cols = ['Survived']
df[indep_cols] = df[indep_cols].astype(float)



predictors = tensor(df[indep_cols].values, dtype=torch.float)
targets = tensor(df[dep_cols].values, dtype=torch.float)  # Ensure targets are float

vals,indices = predictors.max(dim=0)
predictors = predictors / vals

from typing import List, Tuple

def initialize_coeffs(layer_sizes):
    layers: List[torch.Tensor] = []
    constants: List[torch.Tensor] = []
    for i in range(len(layer_sizes)-1):
        layers.append((torch.rand(layer_sizes[i], layer_sizes[i+1])-0.3) / layer_sizes[i+1] * 4)
        constants.append(torch.rand(1)[0]*0.1)
    for l in layers+constants: l.requires_grad_()
    return layers, constants

def calc_preds(coeffs: Tuple[List[torch.Tensor]], predictors: List[torch.Tensor]):
        layers, constants = coeffs
        n = len(layers)
        res = predictors
        for i in range(n):
            res = res@layers[i] + constants[i]
            if i != n-1: res = torch.relu(res)
        return torch.sigmoid(res)

def calc_loss(coeffs: Tuple[List[torch.Tensor]], predictors: List[torch.Tensor], targets: List[torch.Tensor]): return torch.abs(calc_preds(coeffs, predictors) -targets).mean()

def update_coeffs(coeffs: Tuple[List[torch.Tensor]], lr: int):
    layers, constants = coeffs
    with torch.no_grad(): 
        for layer in layers + constants:
            layer.sub_(layer.grad * lr)
            layer.grad.zero_()

def one_epoch(coeffs: Tuple[List[torch.Tensor]], predictors: List[torch.Tensor], targets: List[torch.Tensor], lr: int):
    loss = calc_loss(coeffs, predictors, targets)
    loss.backward()
    update_coeffs(coeffs, lr)
    print(f"{loss:.3f}", end="; ")

def train_model(predictors, targets, layer_sizes, epochs=10, lr=0.1):
    coeffs = initialize_coeffs(layer_sizes)
    for i in range(epochs):
        one_epoch(coeffs, predictors, targets, lr)
    return coeffs

train_model(predictors, targets, [len(indep_cols), 10,  1], epochs=20, lr=1)