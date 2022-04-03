import torch
import torch.nn as nn
from flows import *

from typing import List

class FlowModule(nn.Module):
  def __init__(self, dim_input:int, num_layers: int, flow_type:str ='Planar'):
    super(FlowModule, self).__init__()
    
    flow_block_class = None
    try:
      flow_block_class = globals()[flow_type]
    except (KeyError):
      raise ModuleNotFoundError(f"Error: No '{flow_type}' flow type was not found in the scope, import it to make sure is in scope")

    if(flow_type=='Planar'):
      Planar=flow_block_class
      self.flows = nn.ModuleList([Planar(dim_input,) for _ in range(num_layers)])
    elif (flow_type=='Radial'):
      Radial=flow_block_class
      self.flows = nn.ModuleList([Radial(dim_input,) for _ in range(num_layers)])
    else:
      raise NotImplementedError

  def forward(self, zo):
    log_det_sum = 0.
    zk=zo
    for flow_block in self.flows:
      zk, log_det = flow_block(zk)
      log_det_sum +=log_det

    return zk, log_det_sum


class Encoder(nn.Module):
    def __init__(self, dim_input: int,hidden_dims: List[int], latent_size: int =40, maxout_window_size: int =4):
        super(Encoder, self).__init__()
        assert(len(hidden_dims) >= 1)

        self.maxout_window_size=maxout_window_size
        self.latent_size = latent_size
        hidden_dims = [dim_input] + hidden_dims
        
        self.layers = []

        first = True;
        
        for i in range(len(hidden_dims) - 1):
          if first:
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.MaxPool1d(kernel_size=self.maxout_window_size, stride=self.maxout_window_size))
            first = False
          else:
            self.layers.append(nn.Linear(hidden_dims[i]//(self.maxout_window_size), hidden_dims[i+1]))
            self.layers.append(nn.MaxPool1d(kernel_size=self.maxout_window_size, stride=self.maxout_window_size))


        self.seqNet = nn.Sequential(*self.layers)

        self.fc_mean = nn.Linear(hidden_dims[-1]//(self.maxout_window_size), self.latent_size)
        self.fc_var = nn.Linear(hidden_dims[-1]//(self.maxout_window_size), self.latent_size)

    def forward(self, x):
        z = self.seqNet(x)
        
        return self.fc_mean(z), self.fc_var(z)


class Decoder(nn.Module):
    def __init__(self, dim_input: int, out_dim: int, hidden_dims: List[int], maxout_window_size=4):
        super(Decoder, self).__init__()

        assert(len(hidden_dims) >= 1)

        self.maxout_window_size=maxout_window_size
        
        hidden_dims = [dim_input] + hidden_dims

        
        self.layers = []
        first = True;

        for i in range(len(hidden_dims) - 1):
          if first:
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.MaxPool1d(kernel_size=maxout_window_size, stride=self.maxout_window_size))
            first = False
          else:
            self.layers.append(nn.Linear(hidden_dims[i]//(self.maxout_window_size), hidden_dims[i+1]))
            self.layers.append(nn.MaxPool1d(kernel_size=self.maxout_window_size, stride=self.maxout_window_size))

        
        self.layers.append(nn.Linear(hidden_dims[-1]//self.maxout_window_size, out_dim))
        
        self.seqNet = nn.Sequential(*self.layers)

        
    def forward(self, x):
        return torch.sigmoid(self.seqNet(x))


class NormalisingFlowModel(nn.Module):
  def __init__(self,
               dim_input: int,
               e_hidden_dims: List[int],
               d_hidden_dims:List[int],
               flow_layers_num: int,
               flow_type: str,
               act=torch.tanh, latent_size =40, maxout_window_size=4):
      super(NormalisingFlowModel, self).__init__()

      self.encoder = Encoder(dim_input, e_hidden_dims, latent_size = latent_size, maxout_window_size=maxout_window_size)
      self.flow = FlowModule(latent_size, num_layers=flow_layers_num, flow_type=flow_type)
      self.decoder = Decoder(latent_size,dim_input, d_hidden_dims)

  def reparameterize(self, mu, logvar):
      # returns a random sample from the approximate posterior q(z|x) 
      eps = torch.randn_like(mu)
      return mu + eps * torch.exp(0.5*logvar) 
    

  # Returns:
  # z: sample from the latent space
  # recon: reconstruction of the input
  # log_q0_z0: log(q_0(z_0)) term, the log probability of sampling z_0 from the q_0 initial distribution
  # log_p_zk: log(p(z_k)) term, the log probability of sampling z_k from the distribution p
  # log_det_jacobians: log(partial f/ partial z) terms for the flows
  def forward(self, z):

    #Inference Network
    #Encode
    mu, log_var = self.encoder(z)
    # Reparameterise to sample z_o
    z_o = self.reparameterize(mu, log_var)

    # pass zo through the flows to get zk
    z_k, log_det_sum = self.flow(z_o)

    #Generative Model
    #decode x_hat
    recon = self.decoder(z_k) # temp, just to test the normal vae


    stddev = torch.exp(log_var/2)
    logp_zo = torch.sum(-0.5 * torch.log(torch.tensor(2 * math.pi)) - log_var/2 - 0.5 * ((z_o - mu) / stddev) ** 2, axis=1)
    # logp_zo = Normal(mu, torch.exp((0.5 * log_var))).log_prob(z_o)
    
    logp_zk = torch.sum(-0.5 * (torch.log(torch.tensor(2 * math.pi)) + z_k ** 2), axis=1)
    # logp_zk = torch.sum(Normal(0., 1.).log_prob(z_k)


    return recon, logp_zo, logp_zk, log_det_sum

