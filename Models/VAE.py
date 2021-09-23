# inspired from https://github.com/pytorch/examples/blob/master/vae/main.py
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
#import matplotlib.pyplot as plt
from time import sleep
import numpy as np




""" 
torch.manual_seed(999)

device = torch.device("cpu")
 """
#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):

    """
    FUCK
    """



    def __init__(self,num_feats,latent_dims,hidden_dims):
        super(VAE, self).__init__()
        self.input_dim = num_feats
        self.latent_dims = latent_dims

        #BUILD ENCODER
        modules = []
        dim_in = self.input_dim

        for l_size in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(dim_in,l_size),
                nn.ReLU()))

            dim_in = l_size

        self.encoder    = nn.Sequential(*modules)
        self.for_mu     = nn.Linear(hidden_dims[-1],self.latent_dims)
        self.for_logvar = nn.Linear(hidden_dims[-1],self.latent_dims)


        #BUILD DECODER 
        modules = []
        hidden_dims.reverse()
        dim_in = self.latent_dims

        for l_size in hidden_dims:
            modules.append(nn.Sequential(
                nn.Linear(dim_in,l_size),
                nn.ReLU()))

            dim_in = l_size

        self.decoder    = nn.Sequential(*modules)
        self.recon     = nn.Linear(hidden_dims[0],self.input_dim)
       
            

            


    def encode(self, x):
   
        return 
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        h5 = F.relu(self.fc32(h4))
        return torch.sigmoid(self.fc4(h5))

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    

    def forward(self, x):
        self.mu, self.logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparam(self.mu, self.logvar)
        return self.decode(z), self.mu, self.logvar


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss(self,recon_x, x):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1,self.input_dim), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        KLD /= x.view(-1, self.input_dim).data.shape[0] * self.input_dim
        return BCE + KLD


# """ model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#  """

if __name__ == "__main__":
    

    model = VAE(10,2,[2,3,5,6])
    print(model.encoder)
    print(model.decoder)


    print(torch.sum.__doc__)
   
   

    



  

          
