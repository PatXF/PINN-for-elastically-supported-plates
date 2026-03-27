import torch
import torch.nn as nn
from utils import FixedNetwork, getDerivatives

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


def getLoss(model, x, y):
    w = model(x, y)

    _, _, w_xx, w_yy, w_xy = getDerivatives(w, x, y)

    U1 = torch.mean(0.5 * D * ((w_xx + w_yy) ** 2))
    U2 = torch.mean(0.5 * D * (2 * (1 - nu) * (w_xx * w_yy - w_xy ** 2)))

    # strain energy
    U = U1 - U2

    # external work
    V = torch.mean(q * w)

    S = U - V
    
    return S * A


def trainDRM():
    model = FixedNetwork(a=a, b=b).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    epochs = 5000
    numCollocation = 5000

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        x_c = (torch.rand(numCollocation, 1, device=device) * a).requires_grad_(True)
        y_c = (torch.rand(numCollocation, 1, device=device) * b).requires_grad_(True)

        loss = getLoss(model, x_c, y_c)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} Loss: {loss}")

    return model


if __name__ ==  "__main__":
    model = trainDRM()
    torch.save(model.state_dict(), "modelFixed.pth")
    x = torch.tensor([[0.5]], device=device)
    y = torch.tensor([[1.0]], device=device)
    print(model(x, y)[0])