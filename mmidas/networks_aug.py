import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F



class RNA_augmenter(nn.Module):
    """
        Class for the neural network module for augmenting the RNA-seq data.
    """

    def __init__(self, noise_dim, latent_dim, input_dim, fc_dim, p_drop, momentum, affine, eps):
        """
            class instantiation for the Augmenter network.

            noise_dim: dimension of the noise variable.
            latent_dim: dimension of the continuous (state) latent variable.
            input_dim: input dimension (size of the input layer).
            n_dim: dimension of the hidden layer.
            p_drop: dropout probability.
            momentum: a hyperparameter for batch normalization that updates its running statistics.
            device: computing device, either 'cpu' or 'cuda'.
        """
        super().__init__()

        self.eps = eps
        self.dp = nn.Dropout(p_drop)
        self.noise_dim = noise_dim

        # Define layers for the Augmenter network
        self.noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.bnz = nn.BatchNorm1d(self.noise.out_features)

        # Fully connected layers and their batch normalizations
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=eps, momentum=momentum, affine=affine)
        # self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        # self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features, eps=eps, momentum=momentum, affine=False)
        self.fc3 = nn.Linear(1000, fc_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc4 = nn.Linear(fc_dim, fc_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc5 = nn.Linear(fc_dim, fc_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc5n = nn.Linear(fc_dim + noise_dim, fc_dim // 5)
        self.batch_fc5n = nn.BatchNorm1d(num_features=self.fc5n.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc_mu = nn.Linear(fc_dim // 5, latent_dim)
        self.fc_sigma = nn.Linear(fc_dim // 5, latent_dim)
        self.batch_fc_mu = nn.BatchNorm1d(num_features=self.fc_mu.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc6 = nn.Linear(latent_dim, fc_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc7 = nn.Linear(self.fc6.out_features, fc_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features, eps=eps, momentum=momentum, affine=affine)
        # self.fc8 = nn.Linear(fc_dim, fc_dim)
        # self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features, eps=eps, momentum=momentum, affine=False)
        self.fc9 = nn.Linear(fc_dim, fc_dim)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc10 = nn.Linear(fc_dim, 1000)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc11 = nn.Linear(fc_dim, input_dim)

    def forward(self, x, noise, scale=1.):
        # Apply noise and batch normalization
        z = scale * torch.randn(x.shape[0], self.noise_dim, device=x.device)
        z = F.elu(self.bnz(self.noise(z)))
        # Apply dropout, fully connected layers, and batch normalization with ReLU activation
        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        # x = F.relu(self.batch_fc2(self.fc2(x)))
        # x = F.relu(self.batch_fc3(self.fc3(x)))
        x = F.relu(self.batch_fc4(self.fc4(x)))
        
        # If noise is enabled, concatenate with z and pass through another layer
        if noise:
            x = torch.cat((x, z), dim=1)
            x = F.relu(self.batch_fc5n(self.fc5n(x)))
        else:
            x = F.relu(self.batch_fc5(self.fc5(x)))

        # Compute mean (mu) and standard deviation (sigma) for reparameterization trick
        mu = self.batch_fc_mu(self.fc_mu(x))
        sigma = torch.sigmoid(self.fc_sigma(x))
        s = self.reparam_trick(mu, sigma)

        # Pass through remaining fully connected layers with batch normalization and ReLU activation
        x = F.relu(self.batch_fc6(self.fc6(s)))
        x = F.relu(self.batch_fc7(self.fc7(x)))
        # x = F.relu(self.batch_fc8(self.fc8(x)))
        x = F.relu(self.batch_fc9(self.fc9(x)))
        # x = F.relu(self.batch_fc10(self.fc10(x)))
        
        return s, F.relu(self.fc11(x))
        
    
    def reparam_trick(self, mu, log_sigma):
        """
        Generate samples from a normal distribution for reparametrization trick.

        input args
            mu: mean of the Gaussian distribution for
                q(s|z,x) = N(mu, sigma^2*I).
            log_sigma: log of variance of the Gaussian distribution for
                       q(s|z,x) = N(mu, sigma^2*I).

        return
            a sample from Gaussian distribution N(mu, sigma^2*I).
        """
        std = log_sigma.exp().sqrt()
        eps = torch.rand_like(std).to(mu.device)
        return eps.mul(std).add(mu)


# Define the Discriminator class inheriting from nn.Module
class RNA_discriminator(nn.Module):
    """
        Class for the neural network module for discriminating the latent variables, used in the 
        augmented VAE network.
    """

    def __init__(self, input_dim, fc_dim, p_drop, momentum, affine, eps):
        super().__init__()

        self.dp = nn.Dropout(p_drop)

        # Fully connected layers and their batch normalizations
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features, eps=eps, momentum=momentum, affine=affine)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features, eps=eps, momentum=momentum, affine=affine)
        self.disc = nn.Linear(self.fc2.out_features, 1, 1)

    def forward(self, x):
        # Apply dropout, fully connected layers, and batch normalization with ReLU activation
        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        # Compute discriminator output with sigmoid activation
        output = torch.sigmoid(self.disc(x))
        return x, output
