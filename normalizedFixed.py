import torch
import torch.nn as nn
from utils import NFFNetwork, getLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

## parameters
a = 1.0                                    ## length (m)
b = 2.0                                    ## width (m)
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    epochs = 5000
    numCollocation = 5000

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        x_c = (2 * torch.rand(numCollocation, 1, device=device) - 1).requires_grad_(True)
        y_c = (2 * torch.rand(numCollocation, 1, device=device) - 1).requires_grad_(True)

        loss = getLoss(model, x_c, y_c, nu, Rho)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss * E0}")

    return model


if __name__ ==  "__main__":
    model = trainDRM()
    torch.save(model.state_dict(), "modelNSS.pth")
    x = torch.tensor([[0.0]], device=device)
    y = torch.tensor([[0.0]], device=device)
    print(model(x, y)[0].item() * w0)