if __name__ == "__main__":

    from Models.simple_VAE import VAE,loss_function
    from sklearn.preprocessing import MinMaxScaler
    import argparse
    import torch
    import torch.utils.data
    from torch import nn, optim
    from torch.nn import functional as F
    #from torchvision import datasets, transforms
    #from torchvision.utils import save_image
    import matplotlib.pyplot as plt
    from time import sleep
    import numpy as np




    darray = np.loadtxt('data/wine2.csv',delimiter=',')
    scaler = MinMaxScaler()
    scaler.fit(darray)
    dout = scaler.transform(darray)

    #Sending scaled data to device
    device = torch.device("cpu")
    dtensor = torch.FloatTensor(dout)
    


    model = VAE(num_feats = 14,latent_dims=2).to(device)

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    loss_store = []

    for epoch in range(2500): 
        drecon,mu,log_var =  model(dtensor)
        optimizer.zero_grad()
        loss = loss_function(drecon, dtensor, mu, log_var)
        loss.backward()
        optimizer.step()


        loss_store.append(loss.item())

        # if epoch %100 == 0:
       
       
        #     print(loss.item())

    #print(model.reparam(mu,log_var).size())
    recon_numpy =scaler.inverse_transform(drecon.detach().numpy())
    og_numpy    =scaler.inverse_transform(dtensor.detach().numpy())
    print(np.mean(np.mean(og_numpy-recon_numpy,axis=0)))

    
    #plt.plot(loss_store)
    #plt.show()