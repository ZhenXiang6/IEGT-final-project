"""
OTT Streaming Platform — Enhanced two-dimensional dynamic strategic model.

Upgrades vs. the report baseline (all grounded in the verified literature dossier):
  * Retention NETWORK EXTERNALITY:   rho(P,C,s) = rho0 + dP + dC + kappa_rho * s   (capped)
      -> direct network effect / lock-in (Katz-Shapiro 1985; Klemperer 1987; Farrell-Klemperer 2007)
  * Acquisition DIFFUSION / WOM:     alpha(P,C,s) = a0 + dP~ + dC~ + kappa_alpha * s
      -> Bass (1969) hazard p + q*F:  inflow alpha(s)*(1-s) = (p + q*s)(1-s)   (Peres et al. 2010; Godes-Mayzlin 2004)
  * MARKET-SCALE cost:               C_A(s) = c_A (1 + lambda_A s),  C_R(s) = c_R (1 + lambda_R s),  lambda_A > lambda_R
      -> acquisition cost rises faster with penetration (Min, Zhang, Kim & Srivastava 2016, JMR)
  * Scale-sensitive TVOD revenue:    R(s,T) = p_T (phi0 + phi1 s);  SVOD flat R(s,S)=p_S  (Carroni-Paolini 2020)

State s in [0,1]; action (P,C) in {T,S}x{A,R}; transition s' = rho*s + alpha*(1-s);
Bellman V(s)=max_a {pi + beta V(s')}; solved by value iteration on a 401-point grid
(contraction => unique fixed point: Bellman 1957; Blackwell 1965; Stokey-Lucas-Prescott 1989).

Headline calibration: kappa_rho=0.08, kappa_alpha=0.20, lambda_A=1.0, lambda_R=0.3.
  delta_S=+0.105 (Iyengar et al. 2011); lambda_A,lambda_R from report cost-design (Min et al. 2016);
  kappa_alpha ~ Bass imitation q (typical 0.3-0.5; Sultan et al. 1990); beta=0.95.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt

# ===================== Parameters =====================
BETA = 0.95
p_S, p_T = 15.0, 5.0
phi0, phi1 = 0.5, 0.5
rho_0, delta_S, delta_T, delta_R, delta_A = 0.65, 0.105, 0.0, 0.10, 0.0
kappa_rho, rho_max = 0.08, 0.98
alpha_0, tilde_S, tilde_T, tilde_A, tilde_R = 0.10, 0.0, 0.05, 0.15, 0.0
kappa_alpha = 0.20
c_A, c_R, lambda_A, lambda_R = 1.2, 0.8, 1.0, 0.3

ACTIONS = [("T", "A"), ("T", "R"), ("S", "A"), ("S", "R")]
ACTION_LABELS = {("T", "A"): "Trans-Acq", ("T", "R"): "Trans-Ret",
                 ("S", "A"): "Sub-Acq", ("S", "R"): "Sub-Ret"}
ACTION_COLORS = {("T", "A"): "#E63946", ("T", "R"): "#F4A261",
                 ("S", "A"): "#2A9D8F", ("S", "R"): "#264653"}
BLUE = "#003EA8"


# ===================== Primitives =====================
def revenue(s, P):
    return p_S if P == "S" else p_T * (phi0 + phi1 * s)

def cost(s, C, lamA=lambda_A, lamR=lambda_R):
    return c_A * (1 + lamA * s) if C == "A" else c_R * (1 + lamR * s)

def rho(s, P, C, kr=kappa_rho):
    dP = delta_S if P == "S" else delta_T
    dC = delta_R if C == "R" else delta_A
    return min(rho_0 + dP + dC + kr * s, rho_max)

def alpha(s, P, C, ka=kappa_alpha):
    tP = tilde_S if P == "S" else tilde_T
    tC = tilde_A if C == "A" else tilde_R
    return alpha_0 + tP + tC + ka * s

def transition(s, a, kr=kappa_rho, ka=kappa_alpha):
    P, C = a
    return float(np.clip(rho(s, P, C, kr) * s + alpha(s, P, C, ka) * (1 - s), 0.0, 1.0))

def profit(s, a, lamA=lambda_A, lamR=lambda_R):
    P, C = a
    return revenue(s, P) * s - cost(s, C, lamA, lamR)


# ===================== Value iteration =====================
def value_iteration(n_grid=401, tol=1e-8, max_iter=4000,
                    kr=kappa_rho, ka=kappa_alpha, lamA=lambda_A, lamR=lambda_R):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)
    it = 0
    for it in range(max_iter):
        Vn = np.zeros(n_grid)
        pol = np.zeros(n_grid, dtype=int)
        for i, s in enumerate(s_grid):
            best, barg = -1e18, 0
            for ai, a in enumerate(ACTIONS):
                q = profit(s, a, lamA, lamR) + BETA * np.interp(transition(s, a, kr, ka), s_grid, V)
                if q > best:
                    best, barg = q, ai
            Vn[i], pol[i] = best, barg
        diff = np.max(np.abs(Vn - V))
        V = Vn
        if diff < tol:
            break
    return s_grid, V, pol, it + 1


def simulate(s0, T, s_grid, pol, kr=kappa_rho, ka=kappa_alpha, lamA=lambda_A, lamR=lambda_R):
    s, path, apath, pis = s0, [s0], [], []
    for _ in range(T):
        a = ACTIONS[pol[int(np.argmin(np.abs(s_grid - s)))]]
        apath.append(a); pis.append(profit(s, a, lamA, lamR))
        s = transition(s, a, kr, ka); path.append(s)
    return path, apath, pis


def policy_regimes(s_grid, pol):
    out, cur, start = [], pol[0], 0
    for i in range(1, len(s_grid)):
        if pol[i] != cur:
            out.append((s_grid[start], s_grid[i - 1], ACTIONS[cur])); cur, start = pol[i], i
    out.append((s_grid[start], s_grid[-1], ACTIONS[cur]))
    return out


def fixed_points(s_grid, pol, kr=kappa_rho, ka=kappa_alpha):
    f = np.array([transition(s, ACTIONS[pol[i]], kr, ka) for i, s in enumerate(s_grid)])
    d = f - s_grid
    res = []
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            sfp = s_grid[i] - d[i] * (s_grid[i + 1] - s_grid[i]) / (d[i + 1] - d[i])
            idx = int(np.argmin(np.abs(s_grid - sfp))); i0, i1 = max(0, idx - 4), min(len(s_grid) - 1, idx + 4)
            slope = (f[i1] - f[i0]) / (s_grid[i1] - s_grid[i0] + 1e-12)
            res.append((float(sfp), "stable" if slope < 1 else "unstable"))
    return f, res


# ===================== Solve =====================
print("=" * 70)
print(f"ENHANCED MODEL  (kappa_rho={kappa_rho}, kappa_alpha={kappa_alpha}, lambda_A={lambda_A}, lambda_R={lambda_R})")
print("=" * 70)
s_grid, V, pol, iters = value_iteration()
print(f"converged in {iters} iters")
print("\nOptimal policy a*(s):")
for lo, hi, a in policy_regimes(s_grid, pol):
    print(f"  s in [{lo:.3f},{hi:.3f}]  ->  {ACTION_LABELS[a]}")
f_enh, fps_enh = fixed_points(s_grid, pol)
print("\nFixed points (under optimal policy):")
for sfp, st in fps_enh:
    print(f"  s* = {sfp:.4f}  ({st})")

print("\nFirm asymmetry (25-yr discounted profit):")
for s0, name in [(0.05, "Newcomer (YouTube/Apple)"), (0.30, "Mid (Disney+)"), (0.65, "Incumbent (Netflix)")]:
    path, _, pis = simulate(s0, 25, s_grid, pol)
    disc = sum(BETA ** t * pi for t, pi in enumerate(pis))
    print(f"  {name:26s}: s0={s0:.2f}  s5={path[5]:.3f}  s25={path[-1]:.3f}  disc={disc:.2f}")

# contrast: no-externality constant-cost baseline (report-style)
sg0, V0, pol0, _ = value_iteration(kr=0.0, ka=0.0, lamA=0.0, lamR=0.0)
_, fps0 = fixed_points(sg0, pol0, kr=0.0, ka=0.0)
print("\nNo-externality baseline:")
for lo, hi, a in policy_regimes(sg0, pol0):
    print(f"  s in [{lo:.3f},{hi:.3f}]  ->  {ACTION_LABELS[a]}")
print(f"  steady state s* = {fps0[0][0]:.3f}")

# bifurcation over kappa_rho
print("\nBifurcation over kappa_rho:")
kappas = np.round(np.arange(0.0, 0.301, 0.02), 2)
bif = []
for kr in kappas:
    sgk, Vk, polk, _ = value_iteration(kr=kr)
    _, fpk = fixed_points(sgk, polk, kr=kr)
    for sfp, st in fpk:
        bif.append((kr, sfp, st))
    print(f"  kappa_rho={kr:.2f}: " + ", ".join(f"{x:.3f}({s[0]})" for x, s in fpk))

# ===================== Deck figures =====================
# Fig 1: value function + simulated paths
fig, ax = plt.subplots(figsize=(7.6, 4.8))
cmap = plt.cm.viridis
for j, s0 in enumerate(np.linspace(0.02, 0.98, 25)):
    p, _, _ = simulate(s0, 60, s_grid, pol)
    ax.plot(range(len(p)), p, color=cmap(j / 24), lw=1.1, alpha=0.75)
for sfp, st in fps_enh:
    ax.axhline(sfp, color=BLUE if st == "stable" else "#E63946", lw=1.5, ls="-" if st == "stable" else "--")
    ax.text(61, sfp, f"s*={sfp:.3f}", va="center", fontsize=10, color=BLUE)
ax.set_xlabel("time t (periods)"); ax.set_ylabel("market share s_t"); ax.set_ylim(0, 1); ax.set_xlim(0, 70)
ax.set_title("Simulated paths from 25 starts -> unique steady state", fontweight="bold"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("./fig_value_paths.png", dpi=160, bbox_inches="tight"); plt.close()

# Fig 2: phase diagram
fig, ax = plt.subplots(figsize=(6.4, 5.2))
prev, seg, plotted = pol[0], 0, set()
for i in range(1, len(s_grid) + 1):
    if i == len(s_grid) or pol[i] != prev:
        a = ACTIONS[prev]; lab = ACTION_LABELS[a]
        ax.plot(s_grid[seg:i], f_enh[seg:i], color=ACTION_COLORS[a], lw=2.6, label=lab if lab not in plotted else "_nolegend_")
        plotted.add(lab)
        if i < len(s_grid): prev, seg = pol[i], i
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="45-degree")
for sfp, st in fps_enh:
    ax.scatter(sfp, sfp, s=150, zorder=9, color=BLUE if st == "stable" else "#E63946")
    ax.annotate(f"s*={sfp:.3f}", (sfp, sfp), (sfp - 0.33, sfp + 0.05), fontsize=10, color=BLUE)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("s_t"); ax.set_ylabel("s_{t+1}")
ax.set_title("Phase diagram f(s) under optimal policy", fontweight="bold")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("./fig_phase.png", dpi=160, bbox_inches="tight"); plt.close()

# Fig 3: bifurcation over kappa_rho
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for kr, sfp, st in bif:
    ax.scatter(kr, sfp, s=26, color=BLUE if st == "stable" else "#E63946")
ax.scatter([], [], color=BLUE, label="stable"); ax.scatter([], [], color="#E63946", label="unstable")
ax.axvline(kappa_rho, color="gray", ls=":", lw=1.2)
ax.text(kappa_rho + 0.004, 0.08, f"baseline kappa_rho={kappa_rho}", fontsize=8, color="gray")
ax.set_xlabel("kappa_rho  (retention network externality)"); ax.set_ylabel("steady-state s*")
ax.set_title("Stronger retention network effect -> concentration", fontweight="bold")
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02)
plt.tight_layout(); plt.savefig("./fig_bifurcation.png", dpi=160, bbox_inches="tight"); plt.close()

# Fig 4a: transition building blocks
ss = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.plot(ss, [alpha(s, "S", "A") * (1 - s) for s in ss], color="#2A9D8F", lw=2.6, label=r"acquisition inflow $\alpha(s)(1-s)$")
ax.plot(ss, [alpha(s, "S", "A") for s in ss], color="#2A9D8F", lw=1.4, ls=":", label=r"acquisition rate $\alpha(s)=\alpha_0+\kappa_\alpha s$")
ax.plot(ss, [rho(s, "S", "R") * s for s in ss], color=BLUE, lw=2.6, label=r"retention inflow $\rho(s)\,s$")
ax.set_xlabel("market share s"); ax.set_title("Transition building blocks: retention vs acquisition inflow", fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("./fig_buildingblocks.png", dpi=160, bbox_inches="tight"); plt.close()

# Fig 4b: market-scale cost
fig, ax = plt.subplots(figsize=(6.8, 4.6))
ax.plot(ss, [cost(s, "A") for s in ss], color="#E63946", lw=2.6, label=r"$C_A(s)=c_A(1+\lambda_A s)$")
ax.plot(ss, [cost(s, "R") for s in ss], color="#F4A261", lw=2.6, label=r"$C_R(s)=c_R(1+\lambda_R s)$")
ax.set_xlabel("market share s"); ax.set_ylabel("per-period marketing cost")
ax.set_title("Market-scale cost (Min et al. 2016: lambda_A > lambda_R)", fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("./fig_cost.png", dpi=160, bbox_inches="tight"); plt.close()

print("\nSaved: fig_value_paths.png, fig_phase.png, fig_bifurcation.png, fig_buildingblocks.png, fig_cost.png")
