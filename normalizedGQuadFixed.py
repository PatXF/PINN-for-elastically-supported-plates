import torch
import torch.nn as nn
import numpy as np
from utils import NFFNetwork, getLossQ 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## parameters
a = 1.0                                    ## length (m)
b = 1.0                                    ## width (m)
E = 200e9                                  ## Young's modulus (Pa)
nu = 0.3                                   ## Poisson's ratio
h = 0.01                                   ## thickness (m)
q = -10000                                 ## load (N/m^2)
D = (E * (h ** 3)) / (12 * (1 - nu ** 2))  ## flexural rigidity
A = a * b                                  ## area
w0 = ((A ** 2) * q) / (8 * D)              
Rho = a / b
E0 = w0 * A * q


def trainDRM():
    model = NFFNetwork().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    epochs = 5000
    
    N_q = 70

    pts_1d, weights_1d = np.polynomial.legendre.leggauss(N_q)

    x_grid, y_grid = np.meshgrid(pts_1d, pts_1d, indexing='ij')
    W_grid = np.outer(weights_1d, weights_1d)

    x_c = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_c = torch.tensor(y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    W_c = torch.tensor(W_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    x_c.requires_grad_(True)
    y_c.requires_grad_(True)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = getLossQ(model, x_c, y_c, W_c, nu, Rho)
        
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss.item() * E0:.6f}") 

    return model


if __name__ ==  "__main__":
    model = trainDRM()
    torch.save(model.state_dict(), "modelNFFQuad.pth")
    x = torch.tensor([[0.0]], device=device)
    y = torch.tensor([[0.0]], device=device)
    
    model.eval() 
    print(model(x, y)[0].item() * w0)