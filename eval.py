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

# takes one sample and computes the log-likelihood estimator
# this is the same as the forward and loss from above, but it isn't  summed
# TODO: use the forward function from above
def compute_log_likelihood(x, model):
    mu, log_var, _ = model.encoder(x)
    # Reparameterise to sample z_o
    z_o = model.reparameterize(mu, log_var)

    # pass zo through the flows to get zk
    z_k, log_det_sum = model.flow(z_o)

    #Generative Model
    #decode x_hat
    recon = model.decoder(z_k) # temp, just to test the normal vae

    stddev = torch.exp(log_var/2)
    # logq0_zo = torch.sum(-0.5 * torch.log(torch.tensor(2 * math.pi)) - log_var/2 - 0.5 * ((z_o - mu) / stddev) ** 2, axis=1)
    logq0_zo = torch.sum(Normal(mu, torch.exp((0.5 * log_var))).log_prob(z_o), axis=1)
          
    logp_zk = torch.sum(-0.5 * (torch.log(torch.tensor(2 * math.pi)) + z_k ** 2), axis=1)
    # logp_zk = torch.sum(Normal(0., 1.).log_prob(z_k)

    logqk_zk = logq0_zo - log_det_sum 
    return -torch.sum(F.binary_cross_entropy(recon, x, reduction='none'), axis=-1)+logp_zk-logqk_zk

# importance sample to estimate the average -log p(x) in the dataset
def estimate_marginal_likelihood(num_samples, data_loader, model, D, device):
    estimator = .0

    for x, _ in data_loader:
        x = x.to(device)
        x = x.view(-1, D)
        s = torch.zeros(x.shape[0]).to(device).double()
        for _ in range(num_samples):
          log_likelihood = compute_log_likelihood(x, model).double()
          s += torch.exp(log_likelihood) 

        estimator += torch.sum(torch.log(s/num_samples)).item()
        torch.cuda.empty_cache()
    return -(estimator/len(data_loader.dataset))

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
    def __init__(self, eps=0.0001, max_val = 255):
        self.eps = eps
        self.range = 1 - eps * 2
        self.max_val = max_val

    def __call__(self, x):
        return (self.eps + self.range * (x / max_val)).type(torch.float)
        
def CIFAR10(batch_size=100):
    train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),RangeTransform()]))

    test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),RangeTransform()]))

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