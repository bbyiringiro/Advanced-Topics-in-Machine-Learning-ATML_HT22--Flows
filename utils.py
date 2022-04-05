import os.path
import torch

from models import NormalisingFlowModelVAE
import torch.optim as optim

def save_model(name, model, optimizer, steps, train_losses, settings):
    path = f"drive/MyDrive/ATML_HT22/saved_models/{name}" 

    torch.save({
      'steps': steps,
      'train_losses': train_losses,
      'model': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'settings': settings
    }, path)

def load_model(name, device):
    path = f"drive/MyDrive/ATML_HT22/saved_models/{name}"

    if not os.path.isfile(path):
        return False

    checkpoint = torch.load(path)

    batch_size, optim_lr, rms_prop_momentum, num_training_steps,\
    imp_samples, D, encoder_hidden_dims, decoder_hidden_dims, latent_size,\
    maxout_window_size, non_linearity, optim_type, flow_type, num_flow_blocks,\
    binary = checkpoint['settings']

    model = NormalisingFlowModelVAE(dim_input = D,
                  e_hidden_dims = encoder_hidden_dims,
                  d_hidden_dims = decoder_hidden_dims,
                  flow_layers_num=num_flow_blocks,
                  non_linearity=non_linearity,
                  latent_size=latent_size,
                  maxout_window_size = maxout_window_size,
                  flow_type=flow_type,
                  ).to(device)
    model.load_state_dict(checkpoint['model'])
    model.train()

    if optim_type == 'Adam':
      optimizer = optim.Adam(model.parameters())
    else:
      optimizer = optim.RMSprop(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return model, optimizer, checkpoint['steps'], checkpoint['train_losses'], checkpoint['settings']