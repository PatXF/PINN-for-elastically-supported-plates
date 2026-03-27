import torch
import torch.nn as nn

class SSNetwork(nn.Module):
    def __init__(self, a, b, numLayers = 4, units = 50):
        super().__init__()
        self.a = a
        self.b = b
        layers = [nn.Linear(2, units), nn.Tanh()]

        for _ in range(numLayers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(units, 1))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        wPred = self.net(inputs) * (x * (self.a - x) * y * (self.b - y))
        return wPred
    

class FixedNetwork(nn.Module):
    def __init__(self, a, b, numLayers = 4, units = 50):
        super().__init__()
        self.a = a
        self.b = b
        layers = [nn.Linear(2, units), nn.Tanh()]

        for _ in range(numLayers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(units, 1))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        wPred = self.net(inputs) * ((x ** 2) * ((self.a - x) ** 2) * (y ** 2) * ((self.b - y) ** 2))
        return wPred
    

class NSSNetwork(nn.Module):
    def __init__(self, numLayers = 4, units = 50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]

        for _ in range(numLayers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(units, 1))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        wHat = self.net(inputs) * ((1 - x) * (1 + x) * (1 - y) * (1 + y))
        return wHat
    

class NFFNetwork(nn.Module):
    def __init__(self, numLayers = 4, units = 50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]

        for _ in range(numLayers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(units, 1))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        wHat = self.net(inputs) * (((1 + x) ** 2) * ((1 - x) ** 2) * ((1 + y) ** 2) * ((1 - y) ** 2))
        return wHat
    

class generalNetworkAnalysis(nn.Module):
    def __init__(self, numLayers = 4, units = 50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers.append(nn.Linear(units, units))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(units, 1))

        self.net = nn.Sequential(*layers)

    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim = 1)
        return self.net(inputs)
    

def getDerivatives(w, x, y):
    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]

    return w_x, w_y, w_xx, w_yy, w_xy


def getLoss(model, x, y, nu, Rho):
    w = model(x, y)

    _, _, w_xx, w_yy, w_xy = getDerivatives(w, x, y)

    U1 = torch.mean(((w_xx / Rho + w_yy * Rho) ** 2))
    U2 = torch.mean((2 * (1 - nu) * (w_xx * w_yy - w_xy ** 2)))

    # strain energy
    U = U1 - U2

    # external work
    V = torch.mean(w)

    S = U - V
    
    return S


def getLossQ(model, x, y, W, nu, Rho):
    w = model(x, y)

    _, _, w_xx, w_yy, w_xy = getDerivatives(w, x, y)

    uDensity1 = (w_xx / Rho + w_yy * Rho) ** 2
    uDensity2 = 2 * (1 - nu) * (w_xx * w_yy - w_xy ** 2)

    strainEnergyDensity = uDensity1 - uDensity2

    workDensity = w

    energyDensity = strainEnergyDensity - workDensity

    S = torch.sum(W * energyDensity) / 4.0
    
    return S


def getLossElastic(model, x_c, y_c, x_b, y_b, W_c, W_b, nu, Rho, kS, kT):
    ## kS and kT are the list of vertical and rotational
    ## normalised spring coefficients.
    ## assumes x_b and y_b are lists of boundary points
    w_c = model(x_c, y_c)

    _, _, w_xx, w_yy, w_xy = getDerivatives(w_c, x_c, y_c)

    uDensity1 = (w_xx / Rho + w_yy * Rho) ** 2
    uDensity2 = 2 * (1 - nu) * (w_xx * w_yy - w_xy ** 2)

    strainEnergyDensity = uDensity1 - uDensity2

    workDensity = w_c

    verticalSpringDensity = 0
    rotationalSpringDensity = 0

    for i in range(4):
        w_b = model(x_b[i], y_b[i])
        w_bx, w_by, _, _, _ = getDerivatives(w_b, x_b[i], y_b[i])

        verticalSpringDensity += (kS[i] * (w_b ** 2))

        rotationalSpringDensity += (kT[i] * (w_by ** 2) if i % 2 == 0 else kT[i] * (w_bx ** 2))

    S1 = torch.sum(W_c * (strainEnergyDensity - workDensity)) / 4.0 
    S2 = torch.sum(W_b * (verticalSpringDensity + rotationalSpringDensity)) / 2.0

    S = S1 + S2

    return S, S1, S2