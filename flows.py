import torch
import torch.nn as nn
from torch.autograd import Variable
import math
EPS_L_SAFE = 1e-18
def m(x):
  return -1 + torch.log(EPS_L_SAFE + 1 + torch.exp(x))
class FlowBlockAbstract(nn.Module):
  def __init__(self):
    super().__init__()
    pass

  def forward(self, z):
    raise NotImplementedError

class Planar(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super().__init__()
    self.dim_num=dim_num
    

    self.h = act
    self.h_derivative = lambda x: 1 - (torch.tanh(x) ** 2)

    self.u = torch.nn.Parameter(torch.randn(dim_num))
    self.w = torch.nn.Parameter(torch.randn(dim_num))
    self.b = torch.nn.Parameter(torch.randn(1))

  def forward(self, z):
    # constraining u to insure invertibility
    w_dot_u = self.w@self.u
    u = self.u + ( m(w_dot_u) - (w_dot_u))*self.w/(EPS_L_SAFE + torch.norm(self.w) ** 2)

    #Planar transforamtion
    w_z_b=(z@self.w + self.b).unsqueeze(-1)
    f_z = z + u * self.h(w_z_b)
    psi = self.h_derivative(w_z_b) * self.w 
    log_det = torch.log(EPS_L_SAFE + torch.abs(1+ psi@u))
    return f_z, log_det
    

    
    
    
class Radial(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh, zo_=None):
    super().__init__()
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
    





class TriangularSylvester(nn.Module):
    """
    Adapted from official implementation at https://github.com/riannevdberg/sylvester-flows
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size, act=torch.tanh):

        super().__init__()

        self.z_size = z_size
        self.h = act

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)



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
     
    
class RealNVP(FlowBlockAbstract):
  def __init__(self, dim_num, act=torch.tanh):
    super(RealNVP, self).__init__()

  def forward(self, z):
    raise NotImplementedError
    
    
