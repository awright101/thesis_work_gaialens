# inspired from https://github.com/pytorch/examples/blob/master/vae/main.py
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


#parser = argparse.ArgumentParser(description='VAE MNIST Example')
#parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    #help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    #help='number of epochs to train (default: 10)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
                    #help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
                    #help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    #help='how many batches to wait before logging training status')
#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()

# batch_size = 128
# epochs     = 10

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
    def __init__(self,num_feats,latent_dims):
        super(VAE, self).__init__()
        self.input_dim = num_feats
        self.fc1 = nn.Linear(self.input_dim, 400)
        self.fc11 = nn.Linear(400, 300)
        self.fc12 = nn.Linear(300, 200)
        self.fc21 = nn.Linear(200, latent_dims)
        self.fc22 = nn.Linear(200, latent_dims)
        self.fc3 = nn.Linear(latent_dims, 200)
        self.fc31 = nn.Linear(200, 300)
        self.fc32 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(400, self.input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc11(h1))
        h3 = F.relu(self.fc12(h2))
        return self.fc21(h3), self.fc22(h3)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc31(h3))
        h5 = F.relu(self.fc32(h4))
        return torch.sigmoid(self.fc4(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


# """ model = VAE().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
#  """

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1,232), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.view(-1, 232).data.shape[0] * 232
   
    
    return BCE + KLD


    



  

          
