import numpy as np
import torch
from torch.nn import functional as F
import math
from torch.distributions.normal import Normal

from torch.utils.data import random_split,Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ----- Eval Criteria -----

def logit_normal_observation_likelihood(x, mus):
    logits = torch.log(x / (1-x))
    log_norm_lik = torch.sum(Normal(mus, 1.).log_prob(logits) - torch.log(x * (1 - x)), axis=1)
    return log_norm_lik

# Computes the log-likelihood estimator for one batch of samples
# This is similar to ELBO, but it isn't summed
def compute_log_likelihood(x, model, binary):

    # Approximate mu and log-variance of initial posterior density q0(z0|x)
    mu, log_var, _ = model.encoder(x)
    stddev = torch.exp(log_var/2)

    # Reparameterise to sample z0 from posterior q0(z0|x)
    z_o = model.reparameterize(mu, log_var)

    # Pass z0 through the flows to get zk
    z_k, log_det_sum = model.flow(z_o)

    # Decode x_hat, ie estimate mu of likelihood p(x|zk)
    x_hat = model.decoder(z_k)
    
    # Calculate q0(z0) ie likelihood of sampling z0 from the initial posterior
    log_q0_zo = torch.sum(Normal(mu, stddev).log_prob(z_o), axis=1)

    # Calculate p(zk)
    log_p_zk = torch.sum(Normal(0., 1.).log_prob(z_k), axis=-1)

    # Calculate qk(zk) ie the likelihood of sampling zk from the final (post-flow) density
    log_qk_zk = log_q0_zo - log_det_sum 

    if binary:
        CE = torch.sum(F.binary_cross_entropy(x_hat, x, reduction='none'), axis=-1)
    else:
        CE = logit_normal_observation_likelihood(x, x_hat)

    # final log likelihood
    return -CE + log_p_zk - log_qk_zk

# Importance sample to estimate the average -log p(x) in the dataset
def estimate_marginal_likelihood(num_samples, data_loader, binary, model, device):
    
    estimator = .0
    for x, _ in data_loader:
        batch_size = x.shape[0]
        x = x.flatten(1).to(device)

        s = []
        for _ in range(num_samples):
          log_likelihood = compute_log_likelihood(x, model, binary).double()
          s.append(log_likelihood.detach().cpu().numpy()) 

        estimator += np.sum(np.mean(s, 0))

    return -(estimator / len(data_loader.dataset))

# ----- End Eval Criteria -----

# ----- Datasets -----

class BinaryTransform():
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold).type(x.type())

def BinaryMNIST(batch_size=100):
    train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),BinaryTransform()]))

    test_dataset = datasets.MNIST('./data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),BinaryTransform()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
class RangeTransform():
    def __init__(self, eps=0.0001, max_val = 1):
        self.eps = eps
        self.range = 1 - eps * 2
        self.max_val = max_val

    def __call__(self, x):
        return (self.eps + self.range * (x / self.max_val)).type(torch.float)
        
def CIFAR10(batch_size=100):
    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),RangeTransform(),\
                       transforms.RandomCrop(8)]))

    test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),RangeTransform(),\
                          transforms.RandomCrop(8)]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
 
def SVHN(batch_size=100):
    train_dataset = datasets.SVHN('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),RangeTransform()]))

    test_dataset = datasets.SVHN('./data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),RangeTransform()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
 
def FashionMNIST(batch_size=100):
    train_dataset = datasets.FashionMNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()]))

    test_dataset = datasets.FashionMNIST('./data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ----- End Datasets -----