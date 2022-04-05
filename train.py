import numpy as np
import torch
from torch.nn import functional as F
import math
from torch.distributions.normal import Normal

from torch.utils.data import random_split,Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import save_model, load_model
from models import NormalisingFlowModelVAE
from eval import estimate_marginal_likelihood
import torch.optim as optim

def logit_normal_observation_likelihood(x, mus):
    logits = torch.log(x / (1-x))
    log_norm_lik = torch.sum(Normal(mus, 1.).log_prob(logits) / (x * (1 - x)), axis=1)
    return torch.exp(log_norm_lik)

# Annealed version ELBO with where Bt = min(1, 0.01 + t / 10000)
def annealed_ELBO(x, recon, log_p_zo, log_p_zk, log_det_sum, binary, beta_t=1.):

    if binary:
        CE = F.binary_cross_entropy(recon, x, reduction='sum')
    else:
        CE = torch.sum(logit_normal_observation_likelihood(x, recon))
    log_p_x_zk = (torch.sum(log_p_zk, -1) - CE)

    F_bt = torch.sum(log_p_zo, -1) - beta_t * log_p_x_zk
    # happens if k=0
    if type(log_det_sum) != float:
       F_bt -= torch.sum(log_det_sum.view(-1))
    return F_bt
  
def train_epoch(model, optimizer, tr_loader, binary, epoch_num, steps, max_steps, device, print_progress=False):

    model.train()
    total_loss = 0.
    for batch_idx, (x, _) in enumerate(tr_loader):
        if steps > max_steps - 1:
            return total_loss, -1

        # convert x to shape [batch x input_size]
        x = x.flatten(1).to(device)

        optimizer.zero_grad()
        recon, logp_zo, logp_zk, log_det_sum = model(x)
        
        anneling_beta_t = min(1., 0.01 + steps / 10000)
        loss = annealed_ELBO(x, recon, logp_zo, logp_zk, log_det_sum, binary, beta_t=anneling_beta_t)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        steps += 1

    total_loss /= len(tr_loader.dataset)
    if print_progress:
        print(f"==> Epoch: {epoch_num} Average loss: {(total_loss):.2f}")

    return total_loss, steps
    
def test(model, testing_loader, device, print_progress=False):

    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i, (x, _) in enumerate(testing_loader):
            x = x.flatten(1).to(device)
            recon, logp_zo, logp_zk, log_det_sum = model(x)
            loss = annealed_ELBO(x, recon, logp_zo, logp_zk, log_det_sum )
            total_loss += loss.detach().item()

    total_loss /= len(testing_loader.dataset)

    if print_progress:
      print(f"==> Test loss: {total_loss:.2f}")

    return total_loss
    
def train_and_test(name, tr_loader, test_loader, settings, device, print_freq=25):

    batch_size, optim_lr, rms_prop_momentum, num_training_steps,\
    imp_samples, D, encoder_hidden_dims, decoder_hidden_dims, latent_size,\
    maxout_window_size, non_linearity, optim_type, flow_type, num_flow_blocks,\
    binary = settings

    load_check = load_model(name, device)
    if load_check == False:
        model = NormalisingFlowModelVAE(dim_input = D,
                  e_hidden_dims = encoder_hidden_dims,
                  d_hidden_dims = decoder_hidden_dims,
                  flow_layers_num=num_flow_blocks,
                  non_linearity=non_linearity,
                  latent_size=latent_size,
                  maxout_window_size = maxout_window_size,
                  flow_type=flow_type,
                  ).to(device)

        model.train()
        if optim_type == "Adam":
          optimizer = optim.Adam(model.parameters())
        else:
          optimizer = optim.RMSprop(model.parameters(), lr=optim_lr, momentum=rms_prop_momentum)
        steps = 0
        train_losses = []
        save_model(name, model, optimizer, steps, train_losses, settings)
    else:
        model, optimizer, steps, train_losses, settings = load_check

        batch_size, optim_lr, rms_prop_momentum, num_training_steps,\
        imp_samples, D, encoder_hidden_dims, decoder_hidden_dims, latent_size,\
        maxout_window_size, non_linearity, optim_type, flow_type, num_flow_blocks, binary = settings

        print(f"Loaded saved model. Restarting training from step {steps}.")

    epoch = 0

    print("Training: ")
    while steps != -1:

      train_loss, steps = train_epoch(model, optimizer, tr_loader, binary, epoch,\
                                      steps, num_training_steps, device)
      train_losses.append(train_loss)
      epoch += 1
      if (epoch % print_freq == 0):
          print(f"==> {steps} steps; train loss: {train_loss:.2f}")
          #test_loss = test(model, test_loader, device)
          #print(f"\t\t test loss: {test_loss:.2f}")
          save_model(name, model, optimizer, steps, train_losses, settings)

    test_loss = test(model, test_loader, device)
    print(f"Final test loss: {test_loss:.2f}")
    marg_log_lik = estimate_marginal_likelihood(imp_samples, test_loader, model, device)
    print(f"Marginal log likelihood: {marg_log_lik:.2f}")

    save_model(name, model, optimizer, steps, train_losses, settings)

    return marg_log_lik, test_loss, train_losses, model