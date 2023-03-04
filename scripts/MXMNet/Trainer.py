from MXMNet import *
from Global_MP import *
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
import pickle
import os.path as osp
from math import pi as PI
from ase.io import read
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, radius
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from torch_scatter import scatter
import argparse
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from pytorch_lightning import LightningModule, Trainer

text_file = open("../../data/outliers.txt", "r")
outliers = text_file.readlines()

atoms_list=[]
names=[]
xyz_molecules = list(filter(
    lambda x: x.endswith('xyz'),
    (map(lambda x: os.path.join('../../data/tmQMg_xyz/', x),
    os.listdir("../../data/tmQMg_xyz/")))
))
targets = pd.read_csv('../../data/tmQMg_properties_and_targets.csv', index_col="id")

names=[]
for xyz in tqdm(xyz_molecules):
    name=xyz[-10:-4]
    names.append(name)
print(names)
    
#molecules=np.setdiff1d(names, outliers)
molecules=names
data_molecules = []
hls=[]
for name in tqdm(molecules):
    try:
        # set the atomic numbers, positions, and cell
        atoms = read('../../data/tmQMg_xyz/'+name+'.xyz')
        #name = xyz[-10:-4]
        dp=torch.load('../../data/tmQMg_xyz/'+name+'.pt')
        atom = torch.Tensor(atoms.get_atomic_numbers())
        positions = torch.Tensor(atoms.get_positions())
        natoms = positions.shape[0]

        # put the minimum data in torch geometric data object
        
    
        data = Data(
        pos=positions,
        z=atom,
      #  atomic_numbers=atoms,
        natoms=natoms,
        node_attr=dp.x,
        edge_index=dp.edge_index,
        edge_attr=dp.edge_attr,
        graph_attr=dp.graph_attr,
        g_id=dp.id
        )
           # natoms=natoms,

        # calculate energy
        data.y = targets.loc[name].iso_Polarizability
        data_molecules.append(data)
        ys.append(targets.loc[name].iso_Polarizability)
    except:
        pass
from torch_geometric.data import DataLoader
train_dataset, testval_dataset = train_test_split(data_molecules, test_size=0.2)
test_dataset, val_dataset = train_test_split(testval_dataset, test_size=0.5)
train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

import argparse
config = Config(dim=128, n_layer=3, cutoff=10.0)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model=MXMNet(config).to(device)

from warmup_scheduler import GradualWarmupScheduler

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler)

import wandb
wandb.init(project="NBO_MXMNet")

ema = EMA(model, decay=0.999)

print('===================================================================================')
print('                                Start training:')
print('===================================================================================')

train_loss_epoch = []
val_loss_epoch = []

for epoch in range(300):
    loss_all = 0
    step = 0
    model.train()

    for data in tqdm(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = F.l1_loss(output, data.y)
        loss_all += loss.item()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1000, norm_type=2)
        optimizer.step()
        
        curr_epoch = epoch + float(step) / (len(train_dataset) / 32)
        scheduler_warmup.step(curr_epoch)

        ema(model)
        step += 1

    train_loss = loss_all/len(train_loader.dataset)
    
    error=0
    model.eval()
    ema.assign(model)
    for data in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        loss=F.l1_loss(output, data.y)
        error += loss.item()
        
    val_loss = error/len(test_loader)
    ema.resume(model)
    try:
        wandb.log(
         {
            "train/MAE": train_loss,
            "valid/MAE": val_loss 
          } ) 
    except:
        pass
    train_loss_epoch.append(train_loss)
    
    val_loss_epoch.append(val_loss)

    
    print('Epoch: {:03d}, Train MAE: {:.7f}, Validation MAE: {:.7f} '.format(epoch+1, train_loss, val_loss))

print('===================================================================================')


