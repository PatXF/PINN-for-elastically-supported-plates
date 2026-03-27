import torch
import torch.nn as nn
import numpy as np
from utils import generalNetworkAnalysis, getLossElastic

torch.cuda.manual_seed(42)

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


def train():
    model = generalNetworkAnalysis().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    epochs = 25000
    
    N_q = 70

    pts_1d, weights_1d = np.polynomial.legendre.leggauss(N_q)

    x_grid, y_grid = np.meshgrid(pts_1d, pts_1d, indexing='ij')
    W_grid = np.outer(weights_1d, weights_1d)

    x_c = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_c = torch.tensor(y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    W_c = torch.tensor(W_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    
    W_b = torch.tensor(weights_1d, dtype=torch.float32, device=device).unsqueeze(1)

    x_c.requires_grad_(True)
    y_c.requires_grad_(True)

    x_b = [
        torch.tensor(pts_1d, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
        torch.ones((N_q, 1), dtype=torch.float32, device=device).requires_grad_(True),
        torch.tensor(pts_1d, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
        -torch.ones((N_q, 1), dtype=torch.float32, device=device).requires_grad_(True),
    ]

    y_b = [
        -torch.ones((N_q, 1), dtype=torch.float32, device=device).requires_grad_(True),
        torch.tensor(pts_1d, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
        torch.ones((N_q, 1), dtype=torch.float32, device=device).requires_grad_(True),
        torch.tensor(pts_1d, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
    ]

    model.train()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, s1, s2 = getLossElastic(model, x_c, y_c, x_b, y_b, W_c, W_b, nu, Rho, [epochs / 25]*4, [0]*4)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5000 == 0 or epoch == epochs - 1:
            print(f"\t\tEpoch {epoch:>5d} | S: {loss.item() * E0:.6f} | S1 (plate): {s1.item() * E0:.6f} | S2 (spring): {s2.item() * E0:.6f} | lr: {scheduler.get_last_lr()[0]:.2e}")

    return model


if __name__ ==  "__main__":
    ## 1st analysis: the rotational spring stiffness is 0 to
    ## understand the required Kappa to get the simply supported
    ## condition.
    model = train()
    torch.save(model.state_dict(), "modelGeneralAnalysis.pth")

    x = torch.tensor([[0.0]], device=device)
    y = torch.tensor([[0.0]], device=device)
        
    model.eval() 
    print(f"\t\tCenter deflection: {model(x, y)[0].item() * w0 * 1000:.3f}")