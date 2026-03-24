import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset


def weights_init(m):
    """
    Initialise weights of the networks
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.1)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


def KL_dist(x1, x2, eps=1e-6):
    """
    Calculate the KL divergence between two univariate Gaussians
    """
    logli = ((x2[1] + eps)/(x1[1] + eps)).log() + \
    (x1[1] + (x1[0] - x2[0]).pow(2)).div(x2[1].mul(2.0) + eps) - 0.5
    nll = (logli.sum(1).mean())
    return nll


def TripletLoss(anchor, positive, negative, margin=0.2, loss='BCE'):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    if loss == 'BCE':
        dist = nn.BCELoss()
    elif loss == 'MSE':
        dist = nn.MSELoss()

    distance_positive = dist(positive, anchor)
    distance_negative = dist(negative, anchor)
    losses = F.relu(distance_positive - distance_negative + margin)

    return losses.mean()



def get_loader(x, batch_size, training=True, eps=1e-2):
    """
    Load data from file
    input args
        data (dict): data dictionary
        batch_size (int): batch size
        training (bool): training or testing
        eps (float, optional): epsilon value

    return
        dataloader (DataLoader): data loader

    """

    data_bin = np.where(x > eps, 1, 0)
    data_troch = torch.FloatTensor(x)
    data_bin_troch = torch.FloatTensor(data_bin)
    tensor_data = TensorDataset(data_troch, data_bin_troch)

    # Create dataloader.
    if training:
        dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        dataloader = DataLoader(tensor_data, batch_size=batch_size, shuffle=False, drop_last=False)

    print("Dataloader for augmentation created!")
    return dataloader


def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)


