import torch
import torch.nn as nn
EPS_L_SAFE = 1e-18
def m(x):
  return -1 + torch.log(EPS_L_SAFE + 1 + torch.exp(x))
class FlowBlockAbstract(nn.Module):
  def __init__(self):
    super(FlowBlockAbstract, self).__init__()
    pass

  def forward(self, z):
    raise NotImplementedError

class Planar(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(Planar, self).__init__()
    self.dim_num=dim_num
    

    self.h = act
    self.h_derivative = lambda x: 1 - (torch.tanh(x) ** 2)

    self.u = torch.nn.Parameter(torch.randn(dim_num))
    self.w = torch.nn.Parameter(torch.randn(dim_num))
    self.b = torch.nn.Parameter(torch.randn(1))

  def forward(self, z):
    # constraining u to insure invertibility
    u = self.u + ( m(self.w@self.u) - (self.w@self.u))*self.w/(EPS_L_SAFE + torch.norm(self.w) ** 2)
    #Planar transforamtion
    w_z_b=(z@self.w + self.b).unsqueeze(1) 
    f_z = z + u*self.h(w_z_b)
    
    #log_det_jacobian
    psi = self.h_derivative(w_z_b) * self.w 
    log_det = torch.log(EPS_L_SAFE + torch.abs(1+ psi@u))
    
    return f_z, log_det
    

    
    
    
class Radial(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh, zo_=None):
    super(Radial, self).__init__()
    self.dim_num=dim_num

    
    
    if zo_ is not None:
        self.zo = torch.nn.Parameter(zo_)
    else:
        self.zo = torch.nn.Parameter(torch.randn(dim_num))
    self.alpha_log = torch.nn.Parameter(torch.randn(1)) # since alpha has to be positive R+
    self.beta = torch.nn.Parameter(torch.randn(1))
    
    
    self.h = lambda alpha,r: 1/(EPS_L_SAFE + alpha+r)
    self.h_derivative = lambda alpha,r: -1/(EPS_L_SAFE + r+alpha) ** 2
    

  def forward(self, z):
    z_diff = z-self.zo
    r = torch.linalg.vector_norm(z_diff, dim=-1, keepdim=True)
    # alpha = torch.exp(self.alpha_log)
    # alpha = self.alpha_log
    alpha = torch.abs(self.alpha_log)

    # constraining B to insure invertibility
    beta = -alpha + (1+m(self.beta))

    h_a_r = self.h(alpha, r)
    h_a_r_deriv = self.h_derivative(alpha, r)

    f_z = z + beta*h_a_r*z_diff
    
    log_det = (self.dim_num - 1) * torch.log( EPS_L_SAFE + 1 + beta * h_a_r) + torch.log( EPS_L_SAFE + 1 + (beta * h_a_r) + beta *h_a_r_deriv* r )


    return f_z, log_det
    

class InVerseAutoRegressive(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(InVerseAutoRegressive, self).__init__()

  def forward(self, z):
    raise NotImplementedError  
    
class GlowFlow(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(GlowFlow, self).__init__()

  def forward(self, z):
    raise NotImplementedError
    
    
class SylvesterFlow(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(SylvesterFlow, self).__init__()

  def forward(self, z):
    raise NotImplementedError
     
    
class RealNVP(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(RealNVP, self).__init__()

  def forward(self, z):
    raise NotImplementedError
    
    
