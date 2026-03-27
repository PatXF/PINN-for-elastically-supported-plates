import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# ── Network definitions (same as utils.py) ──────────────────────────────────

import torch.nn as nn

class SSNetwork(nn.Module):
    def __init__(self, a=1.0, b=2.0, numLayers=4, units=50):
        super().__init__()
        self.a, self.b = a, b
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers += [nn.Linear(units, units), nn.Tanh()]
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        wPred = self.net(torch.cat([x, y], dim=1))
        return wPred * (x * (self.a - x) * y * (self.b - y))


class FixedNetwork(nn.Module):
    def __init__(self, a=1.0, b=2.0, numLayers=4, units=50):
        super().__init__()
        self.a, self.b = a, b
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers += [nn.Linear(units, units), nn.Tanh()]
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        wPred = self.net(torch.cat([x, y], dim=1))
        return wPred * ((x**2) * ((self.a - x)**2) * (y**2) * ((self.b - y)**2))


class NSSNetwork(nn.Module):
    def __init__(self, numLayers=4, units=50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers += [nn.Linear(units, units), nn.Tanh()]
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        wHat = self.net(torch.cat([x, y], dim=1))
        return wHat * ((1 - x) * (1 + x) * (1 - y) * (1 + y))


class NFFNetwork(nn.Module):
    def __init__(self, numLayers=4, units=50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers += [nn.Linear(units, units), nn.Tanh()]
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        wHat = self.net(torch.cat([x, y], dim=1))
        return wHat * (((1 + x)**2) * ((1 - x)**2) * ((1 + y)**2) * ((1 - y)**2))


class GeneralNetworkAnalysis(nn.Module):
    def __init__(self, numLayers=4, units=50):
        super().__init__()
        layers = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(numLayers):
            layers += [nn.Linear(units, units), nn.Tanh()]
        layers.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


# ── Helpers ──────────────────────────────────────────────────────────────────

def getDerivatives(w, x, y):
    w_x  = torch.autograd.grad(w,   x, grad_outputs=torch.ones_like(w),   create_graph=True)[0]
    w_y  = torch.autograd.grad(w,   y, grad_outputs=torch.ones_like(w),   create_graph=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_xy = torch.autograd.grad(w_x, y, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    return w_x, w_y, w_xx, w_yy, w_xy


def detect_network(name: str):
    """Infer model class and domain from the filename stem."""
    n = name.upper()
    if   "NFF"     in n: return NFFNetwork(),              "normalized"
    elif "NSS"     in n: return NSSNetwork(),              "normalized"
    elif "GENERAL" in n: return GeneralNetworkAnalysis(),  "normalized"
    elif "FIX"     in n: return FixedNetwork(),            "physical"
    elif "SS"      in n: return SSNetwork(),               "physical"
    else:
        raise ValueError(
            f"Cannot infer network type from '{name}'.\n"
            "Expected one of: NSS, NFF, General, SS, Fixed in the filename."
        )


def to_physical(xi, eta, a, b, domain):
    """Map model coordinates → physical (m)."""
    if domain == "normalized":          # xi,eta ∈ [-1,1]
        x_phys = (xi + 1) / 2 * a
        y_phys = (eta + 1) / 2 * b
    else:                               # xi=x, eta=y ∈ [0,a]×[0,b]
        x_phys = xi
        y_phys = eta
    return x_phys, y_phys


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate(model_path: str):
    # ── Physical parameters ──────────────────────────────────────────────────
    a   = 1.0
    b   = 1.0
    E   = 200e9
    nu  = 0.3
    h   = 0.01
    q   = -10000.0
    D   = (E * h**3) / (12 * (1 - nu**2))
    A   = a * b
    w0  = (A**2 * q) / (8 * D)
    Rho = a / b
    E0  = w0 * A * q

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device : {device}")
    print(f"Reference deflection w0 = {w0*1e3:.4f} mm\n")

    # ── Load model ───────────────────────────────────────────────────────────
    stem   = os.path.splitext(os.path.basename(model_path))[0]
    model, domain = detect_network(stem)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"Model  : {stem}  |  Domain : {domain}")

    # ── Coordinate helpers ────────────────────────────────────────────────────
    # Model coords for boundary & interior
    if domain == "normalized":
        xi_min, xi_max   = -1.0,  1.0
        eta_min, eta_max = -1.0,  1.0
    else:
        xi_min, xi_max   =  0.0,  a
        eta_min, eta_max =  0.0,  b

    def mk_tensor(val, req_grad=False):
        t = torch.tensor([[val]], dtype=torch.float32, device=device)
        if req_grad:
            t.requires_grad_(True)
        return t

    # ── Evaluation grid ───────────────────────────────────────────────────────
    N = 200
    xi_vals  = np.linspace(xi_min,  xi_max,  N)
    eta_vals = np.linspace(eta_min, eta_max, N)
    XI, ETA  = np.meshgrid(xi_vals, eta_vals, indexing="ij")

    XI_t  = torch.tensor(XI.flatten(),  dtype=torch.float32, device=device).unsqueeze(1)
    ETA_t = torch.tensor(ETA.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        W_hat = model(XI_t, ETA_t).cpu().numpy().reshape(N, N)

    W_phys_mm = W_hat * w0 * 1e3           # dimensioned deflection in mm
    X_phys, Y_phys = to_physical(XI, ETA, a, b, domain)

    # ── 1. Mean deflection at boundary ────────────────────────────────────────
    idx_lo_xi  = 0
    idx_hi_xi  = N - 1
    idx_lo_eta = 0
    idx_hi_eta = N - 1

    boundary_values = np.concatenate([
        W_hat[idx_lo_xi,  :],       # xi = xi_min  (x=0)
        W_hat[idx_hi_xi,  :],       # xi = xi_max  (x=a)
        W_hat[:,  idx_lo_eta],      # eta = eta_min (y=0)
        W_hat[:,  idx_hi_eta],      # eta = eta_max (y=b)
    ])
    mean_bdy_raw  = float(np.mean(boundary_values))
    mean_bdy_mm   = mean_bdy_raw * w0 * 1e3

    # ── 2. Total energy (Gauss quadrature, 70-point) ──────────────────────────
    N_q = 70
    pts_1d, weights_1d = np.polynomial.legendre.leggauss(N_q)
    xg, yg = np.meshgrid(pts_1d, pts_1d, indexing="ij")
    Wg     = np.outer(weights_1d, weights_1d)

    xg_t = torch.tensor(xg.flatten(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    yg_t = torch.tensor(yg.flatten(), dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True)
    Wg_t = torch.tensor(Wg.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    w_g = model(xg_t, yg_t)
    _, _, w_xx, w_yy, w_xy = getDerivatives(w_g, xg_t, yg_t)

    u1_density      = (w_xx / Rho + w_yy * Rho) ** 2
    u2_density      = 2 * (1 - nu) * (w_xx * w_yy - w_xy**2)
    strain_energy   = torch.sum(Wg_t * (u1_density - u2_density)) / 4.0
    external_work   = torch.sum(Wg_t * w_g) / 4.0
    total_energy    = strain_energy - external_work

    strain_energy_J = strain_energy.item() * E0
    ext_work_J      = external_work.item() * E0
    total_energy_J  = total_energy.item()  * E0

    # ── 3. Deflections at specific points ─────────────────────────────────────
    # Points in physical coords (x, y); converted to model coords as needed
    specific_points_phys = [
        ("Centre          (a/2, b/2)", a/2,   b/2  ),
        ("Quarter (a/4, b/4)",         a/4,   b/4  ),
        ("Quarter (3a/4, b/4)",        3*a/4, b/4  ),
        ("Quarter (a/4, 3b/4)",        a/4,   3*b/4),
        ("Quarter (3a/4, 3b/4)",       3*a/4, 3*b/4),
        ("(a/3, b/3)",                 a/3,   b/3  ),
        ("(2a/3, 2b/3)",               2*a/3, 2*b/3),
    ]

    print("=" * 62)
    print("  EVALUATION RESULTS")
    print("=" * 62)
    print(f"\n  Model file : {model_path}")
    print(f"  Network    : {type(model).__name__}  |  Domain : {domain}")

    print(f"\n{'─'*62}")
    print("  BOUNDARY DEFLECTIONS")
    print(f"{'─'*62}")
    print(f"  Mean (normalised) : {mean_bdy_raw:.6e}")
    print(f"  Mean (mm)         : {mean_bdy_mm:.6e} mm")
    print(f"  Max  (mm)         : {float(np.max(np.abs(boundary_values))) * w0 * 1e3:.6e} mm")

    print(f"\n{'─'*62}")
    print("  ENERGY (dimensional)")
    print(f"{'─'*62}")
    print(f"  Strain energy  : {strain_energy_J:.6f} J")
    print(f"  External work  : {ext_work_J:.6f} J")
    print(f"  Total energy   : {total_energy_J:.6f} J")

    print(f"\n{'─'*62}")
    print("  DEFLECTIONS AT SPECIFIC POINTS")
    print(f"{'─'*62}")
    print(f"  {'Point':<35} {'w (mm)':>12}  {'w/w0':>10}")
    print(f"  {'-'*55}")

    point_results = []
    with torch.no_grad():
        for label, xp, yp in specific_points_phys:
            if domain == "normalized":
                xi_p  = 2 * xp / a - 1
                eta_p = 2 * yp / b - 1
            else:
                xi_p, eta_p = xp, yp
            xt = torch.tensor([[xi_p]],  dtype=torch.float32, device=device)
            yt = torch.tensor([[eta_p]], dtype=torch.float32, device=device)
            w_hat = model(xt, yt).item()
            w_mm  = w_hat * w0 * 1e3
            print(f"  {label:<35} {w_mm:>12.6f}  {w_hat:>10.6f}")
            point_results.append((label, w_mm, w_hat))

    print("=" * 62)

    # ── 4. Plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Model Evaluation — {stem}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # 4a. Filled contour (physical coords)
    ax1  = fig.add_subplot(gs[0, 0])
    cp   = ax1.contourf(X_phys, Y_phys, W_phys_mm, levels=50, cmap="RdBu_r")
    fig.colorbar(cp, ax=ax1, label="w  [mm]")
    ax1.set_xlabel("x  [m]")
    ax1.set_ylabel("y  [m]")
    ax1.set_title("Deflection field")
    ax1.set_aspect("equal")

    # 4b. Contour lines
    ax2 = fig.add_subplot(gs[0, 1])
    cs  = ax2.contour(X_phys, Y_phys, W_phys_mm, levels=15, cmap="RdBu_r")
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.3f")
    ax2.set_xlabel("x  [m]")
    ax2.set_ylabel("y  [m]")
    ax2.set_title("Contour lines")
    ax2.set_aspect("equal")

    # Mark the specific evaluation points
    for label, w_mm, _ in point_results:
        parts = label.split("(")[1].rstrip(")")
        cx = eval(parts.split(",")[0].strip().replace("a", str(a)).replace("b", str(b)))
        cy = eval(parts.split(",")[1].strip().replace("a", str(a)).replace("b", str(b)))
        ax2.plot(cx, cy, "k.", ms=5)

    # 4c. Cross-section along x=a/2
    ax3 = fig.add_subplot(gs[0, 2])
    mid_xi = N // 2
    ax3.plot(Y_phys[mid_xi, :], W_phys_mm[mid_xi, :], "b-", lw=1.8, label=f"x = a/2")
    mid_eta = N // 2
    ax3.plot(X_phys[:, mid_eta], W_phys_mm[:, mid_eta], "r--", lw=1.8, label=f"y = b/2")
    ax3.axhline(0, color="k", lw=0.6, ls=":")
    ax3.set_xlabel("Coordinate  [m]")
    ax3.set_ylabel("w  [mm]")
    ax3.set_title("Centre-line sections")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    out_path = f"{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved → {out_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter model filename (e.g. modelNSSQuad.pth): ").strip()

    if not path.endswith(".pth"):
        path += ".pth"

    if not os.path.isfile(path):
        print(f"[ERROR] File not found: '{path}'")
        sys.exit(1)

    evaluate(path)