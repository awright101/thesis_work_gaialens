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

class BETA_VAE(nn.Module):

    """
    Docstring for VAE
    
    """



    def __init__(self,num_feats,latent_dims,hidden_dims,beta=2):
        super(BETA_VAE, self).__init__()
        self.input_dim = num_feats
        self.latent_dims = latent_dims
        self.beta = beta
        self.mu = None
        self.logvar = None

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
        self.recon     = nn.Linear(hidden_dims[-1],self.input_dim)
       
            

            


    def encode(self, x):
        mu = self.for_mu(self.encoder(x.view(-1, self.input_dim)))
        logvar = self.for_logvar(self.encoder(x))
        return mu,logvar
    
    def decode(self, z):
        
        temp = self.decoder(z)
        temp = self.recon(temp)


        return torch.sigmoid(temp)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    

    def forward(self, x):
        self.mu, self.logvar = self.encode(x)
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
        return BCE + self.beta*KLD


# """ model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#  """

if __name__ == "__main__":
    print()
    print()
    print()
    x = torch.FloatTensor([i for i in range(10)])
    model = BETA_VAE(10,2,[200,2,3,6],beta=2)

    reconx,_,_ = model(x)
    print(model.loss(reconx,x))
    


   

    



  

          
