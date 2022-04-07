import torch
import torch.nn as nn
from flows import *
from torch.autograd import Variable
from torch.distributions.normal import Normal
import math

from typing import List


class Encoder(nn.Module):
    def __init__(self, dim_input: int, hidden_dims: List[int],
                non_linearity: str = "MaxOut",
                latent_size: int =40, maxout_window_size: int =4):
        super(Encoder, self).__init__()
        assert(len(hidden_dims) >= 1)

        self.maxout_window_size=maxout_window_size
        self.latent_size = latent_size
        hidden_dims = [dim_input] + hidden_dims
        
        self.layers = []
        if non_linearity == "MaxOut":
          first = True
          for i in range(len(hidden_dims) - 1):
            if first:
              self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
              self.layers.append(nn.MaxPool1d(kernel_size=self.maxout_window_size, stride=self.maxout_window_size))
              first = False
            else:
              self.layers.append(nn.Linear(hidden_dims[i]//(self.maxout_window_size), hidden_dims[i+1]))
              self.layers.append(nn.MaxPool1d(kernel_size=self.maxout_window_size, stride=self.maxout_window_size))

          self.fc_mean = nn.Linear(hidden_dims[-1]//(self.maxout_window_size), self.latent_size)
          self.fc_var = nn.Linear(hidden_dims[-1]//(self.maxout_window_size), self.latent_size)

        elif non_linearity == "ReLU":
            for i in range(len(hidden_dims) - 1):
              self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
              self.layers.append(nn.ReLU())

            self.fc_mean = nn.Linear(hidden_dims[-1], self.latent_size)
            self.fc_var = nn.Linear(hidden_dims[-1], self.latent_size)
        else:
          raise ModuleNotFoundError(f"Error: {non_linearity} is not a valid non linearity layer.")

        self.seqNet = nn.Sequential(*self.layers)

    def forward(self, x):
        h = self.seqNet(x)
        
        return self.fc_mean(h), self.fc_var(h), h


class Decoder(nn.Module):
    def __init__(self, dim_input: int, out_dim: int, hidden_dims: List[int],
                non_linearity: str = "MaxOut", maxout_window_size=4):
        super(Decoder, self).__init__()

        assert(len(hidden_dims) >= 1)

        self.maxout_window_size=maxout_window_size
        
        hidden_dims = [dim_input] + hidden_dims

        self.layers = []
        if non_linearity == "MaxOut":
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

        elif non_linearity == "ReLU":
          for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
          self.layers.append(nn.Linear(hidden_dims[-1], out_dim))

        else:
          raise ModuleNotFoundError(f"Error: {non_linearity} is not a valid non linearity layer.")
        
        self.seqNet = nn.Sequential(*self.layers)

    def forward(self, x):
        return torch.sigmoid(self.seqNet(x))

class FlowModule(nn.Module):
  def __init__(self, dim_input:int, num_layers: int,
              encoder_out_dim = 40, flow_type:str ='Planar'):
    super(FlowModule, self).__init__()
    
    flow_block_class = None
    self.flow_type = flow_type
    self.num_flows = num_layers
    self.z_size =dim_input;

    try:
      flow_block_class = globals()[flow_type]
    except (KeyError):
      raise ModuleNotFoundError(f"Error: No '{flow_type}' flow type was not found in the scope, import it to make sure is in scope")

    if(flow_type=='Planar'):
      Planar=flow_block_class
      self.flows = nn.ModuleList([Planar(dim_input,) for _ in range(num_layers)])
    elif (flow_type=='PlanarV2'):
      self.q_z_nn_output_dim = encoder_out_dim;
      PlanarV2=flow_block_class
      self.flows = nn.ModuleList([PlanarV2() for _ in range(num_layers)])



      # Amortized flow parameters
      self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
      self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
      self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)







    elif (flow_type=='Radial'):
      Radial=flow_block_class
      self.flows = nn.ModuleList([Radial(dim_input,) for _ in range(num_layers)])
    elif(flow_type=="TriangularSylvester"):

      assert(encoder_out_dim!=None)
      
      self.flows = nn.ModuleList([TriangularSylvester(dim_input,) for _ in range(num_layers)])

      # the following for TriangularSylvester is adapted from the orginal paper publication from:
      #https://github.com/riannevdberg/sylvester-flows
      # self.flip_idx
      self.q_z_nn_output_dim = encoder_out_dim;


       # permuting indices corresponding to Q=P (permutation matrix) for every other flow
      flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
      self.register_buffer('flip_idx', flip_idx)

      # Masks needed for triangular r1 and r2.
      triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
      triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
      diag_idx = torch.arange(0, self.z_size).long()

      self.register_buffer('triu_mask', Variable(triu_mask))
      self.triu_mask.requires_grad = False
      self.register_buffer('diag_idx', diag_idx)

      # Amortized flow parameters
      # Diagonal elements of r1 * r2 have to satisfy -1 < r1 * r2 for flow to be invertible
      self.diag_activation = nn.Tanh()

      self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)

      self.amor_diag1 = nn.Sequential(
          nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
          self.diag_activation
      )
      self.amor_diag2 = nn.Sequential(
          nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size),
          self.diag_activation
      )

      self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)



    else:
      raise NotImplementedError

  def sylvgetMortisedParameters(self, h):
    batch_size =h.size(0) # TASK
    h = h.view(-1, self.q_z_nn_output_dim)

    # Amortized r1, r2, q, b for all flows


    full_d = self.amor_d(h)
    diag1 = self.amor_diag1(h)
    diag2 = self.amor_diag2(h)

    
    full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
    diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
    diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)

    r1 = full_d * self.triu_mask
    r2 = full_d.transpose(2, 1) * self.triu_mask

    r1[:, self.diag_idx, self.diag_idx, :] = diag1
    r2[:, self.diag_idx, self.diag_idx, :] = diag2

    b = self.amor_b(h)

    # Resize flow parameters to divide over K flows
    b = b.resize(batch_size, 1, self.z_size, self.num_flows)

    return r1, r2, b


  def getPlanarParameters(self, h):
    batch_size = h.size(0)


    h = h.view(-1, self.q_z_nn_output_dim)
    # return amortized u an w for all flows
    u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
    w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
    b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)

    return u, w, b




  def forward(self, zo, h=None):
    log_det_sum = 0.
    zk=zo
    if(self.flow_type == "TriangularSylvester"):

      #get amortise  r1, r2, b for all flows
      r1, r2, b = self.sylvgetMortisedParameters(h)


      for k, flow_block in enumerate(self.flows):
          if k % 2 == 1:
              permute_z = self.flip_idx
          else:
              permute_z = None

          zk, log_det_sum = flow_block(zk, r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z, sum_ldj=True)

          log_det_sum += log_det_sum
    elif (self.flow_type =="PlanarV2"):
      u, w, b = self.getPlanarParameters(h)
      for k in range(self.num_flows):
            u, w, b = self.getPlanarParameters(h)
            zk, log_det_jacobian = self.flows[k](zk, u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            log_det_sum += log_det_jacobian
    else:
      for flow_block in self.flows:
        zk, log_det = flow_block(zk)
        log_det_sum +=log_det


    return zk, log_det_sum


class NormalisingFlowModelVAE(nn.Module):
  def __init__(self,
               dim_input: int,
               e_hidden_dims: List[int],
               d_hidden_dims:List[int],
               flow_layers_num: int,
               flow_type: str,
               non_linearity: str = "MaxOut",
               latent_size=40, maxout_window_size=4, encoder_out_dim=100):
    super().__init__()
    self.flow_type = flow_type
    self.encoder = Encoder(dim_input, e_hidden_dims,
                          non_linearity = non_linearity,
                          latent_size = latent_size,
                          maxout_window_size = maxout_window_size)
    self.flow = FlowModule(latent_size, num_layers = flow_layers_num,
                          flow_type = flow_type,
                          encoder_out_dim = encoder_out_dim)
    self.decoder = Decoder(latent_size, dim_input, d_hidden_dims,
                          non_linearity = non_linearity,
                          maxout_window_size = maxout_window_size)

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
    mu, log_var, h = self.encoder(z)
    # Reparameterise to sample z_o

    z_o = self.reparameterize(mu, log_var)

    # pass zo through the flows to get zk

    if(self.flow_type == "TriangularSylvester" or self.flow_type=="PlanarV2"):
      z_k, log_det_sum = self.flow(z_o, h=h)
    else:
      z_k, log_det_sum = self.flow(z_o)

    #Generative Model
    #decode x_hat
    recon = self.decoder(z_k) # temp, just to test the normal vae


    stddev = torch.exp(log_var/2)
    logq0_zo = torch.sum(Normal(mu, torch.exp((0.5 * log_var))).log_prob(z_o), axis=1)

    logp_zk = torch.sum(Normal(0., 1.).log_prob(z_k), axis=1)

    return recon, logq0_zo, logp_zk, log_det_sum


