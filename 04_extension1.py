"""
OTT Streaming Platform — Extension 1: Multiple Steady States & Path Dependence
Section 7: Network externality on retention ρ(P,C,s) = ρ₀ + δ_P + δ_C + κ·s

Transition (consistent with 03_solve_model.py baseline):
  s' = ρ(P,C,s,κ)·s + α_bar(P,C)·(1-s)²
  α_bar(P,C) is constant; the (1-s) market-saturation factor is embedded in α,
  and the transition multiplies by another (1-s) for the available non-user pool.
  This matches the baseline and produces the non-linear f(s) needed for bifurcation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ========== Parameters ==========
BETA  = 0.95
p_S   = 15.0
p_T   = 5.0
phi0  = 0.5
phi1  = 0.5
c_A   = 1.2
c_R   = 0.8

rho_0   = 0.65
delta_S = 0.105
delta_T = 0.0
delta_R = 0.10
delta_A = 0.0

alpha_0 = 0.10
tilde_S = 0.0
tilde_T = 0.05
tilde_A = 0.15
tilde_R = 0.0

ACTIONS = [('T','A'), ('T','R'), ('S','A'), ('S','R')]
ACTION_LABELS = {
    ('T','A'): 'Trans-Acq',
    ('T','R'): 'Trans-Ret',
    ('S','A'): 'Sub-Acq',
    ('S','R'): 'Sub-Ret',
}
ACTION_COLORS = {
    ('T','A'): '#E63946',
    ('T','R'): '#F4A261',
    ('S','A'): '#2A9D8F',
    ('S','R'): '#264653',
}

# ========== Model primitives ==========

def revenue(s, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s)

def cost(C):
    return c_A if C == 'A' else c_R

def rho_net(P, C, s, kappa):
    """Retention rate with network externality κ·s."""
    dP = delta_S if P == 'S' else delta_T
    dC = delta_R if C == 'R' else delta_A
    return min(rho_0 + dP + dC + kappa * s, 1.0)

def alpha_val(s, P, C):
    """Acquisition rate — α_bar(P,C)·(1-s), consistent with baseline model."""
    tP = tilde_S if P == 'S' else tilde_T
    tC = tilde_A if C == 'A' else tilde_R
    return (alpha_0 + tP + tC) * (1 - s)

def transition(s, action, kappa):
    """s' = ρ(s,κ)·s + α_bar·(1-s)²"""
    P, C = action
    return rho_net(P, C, s, kappa) * s + alpha_val(s, P, C) * (1 - s)

def profit(s, action):
    P, C = action
    return revenue(s, P) * s - cost(C)

# ========== Value Iteration ==========

def value_iteration(kappa, n_grid=201, tol=1e-7, max_iter=2000):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)

    for _ in range(max_iter):
        V_new = np.zeros(n_grid)
        policy = np.zeros(n_grid, dtype=int)
        for i, s in enumerate(s_grid):
            Q = []
            for a in ACTIONS:
                sp = np.clip(transition(s, a, kappa), 0, 1)
                Q.append(profit(s, a) + BETA * np.interp(sp, s_grid, V))
            V_new[i] = max(Q)
            policy[i] = int(np.argmax(Q))
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    return s_grid, V, policy

# ========== Fixed-point utilities ==========

def transition_curve(s_grid, policy, kappa):
    """f(s) = s' under optimal policy for each s on grid."""
    return np.array([
        np.clip(transition(s, ACTIONS[policy[i]], kappa), 0, 1)
        for i, s in enumerate(s_grid)
    ])

def find_fixed_points(s_grid, f):
    """Find s* where f(s*) = s* via sign-change detection."""
    diff = f - s_grid
    fps = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] <= 0:
            # Linear interpolation
            denom = diff[i+1] - diff[i]
            if abs(denom) > 1e-12:
                s_fp = s_grid[i] - diff[i] * (s_grid[i+1] - s_grid[i]) / denom
                fps.append(float(s_fp))
    return fps

def is_stable(s_fp, s_grid, f):
    """Stable fixed point iff local slope of f < 1."""
    idx = np.argmin(np.abs(s_grid - s_fp))
    i0, i1 = max(0, idx - 4), min(len(s_grid) - 1, idx + 4)
    slope = (f[i1] - f[i0]) / (s_grid[i1] - s_grid[i0] + 1e-12)
    return slope < 1.0

# ========== Simulate from a starting point ==========

def simulate(s0, T, s_grid, policy, kappa):
    path = [s0]
    s = s0
    for _ in range(T):
        i = np.argmin(np.abs(s_grid - s))
        s = np.clip(transition(s, ACTIONS[policy[i]], kappa), 0, 1)
        path.append(s)
    return path

# ========== Run bifurcation sweep ==========

kappa_vals = np.round(np.arange(0.0, 0.51, 0.02), 2)

print("=" * 60)
print("Extension 1: Multiple Steady States & Path Dependence")
print("=" * 60)
print(f"\nSweeping κ ∈ [0, 0.5] ({len(kappa_vals)} values) ...")

bifurc_data = []   # list of (kappa, fp, stable)
cached = {}        # kappa -> (s_grid, V, policy, f)

for kappa in kappa_vals:
    s_grid, V, policy = value_iteration(kappa)
    f = transition_curve(s_grid, policy, kappa)
    fps = find_fixed_points(s_grid, f)
    for fp in fps:
        stable = is_stable(fp, s_grid, f)
        bifurc_data.append((kappa, fp, stable))
    cached[kappa] = (s_grid, V, policy, f)

    fp_str = ', '.join(f'{x:.3f}' for x in fps)
    print(f"  κ = {kappa:.2f}: {len(fps)} fixed point(s) → [{fp_str}]")

# ========== Phase diagram at κ = 0.18 (bifurcation zone) ==========

kappa_hi = 0.18
s_grid, V, policy, f_hi = cached[kappa_hi]
fps_hi = find_fixed_points(s_grid, f_hi)
stab_hi = [is_stable(fp, s_grid, f_hi) for fp in fps_hi]

print(f"\nκ = {kappa_hi}: fixed points = {[round(x,3) for x in fps_hi]}")
print(f"  stability = {['stable' if st else 'unstable' for st in stab_hi]}")

# Simulate from many starting points
n_starts = 21
start_points = np.linspace(0.02, 0.98, n_starts)
T_sim = 50
sim_paths = [simulate(s0, T_sim, s_grid, policy, kappa_hi) for s0 in start_points]

# ========== Plot ==========

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Plot 1: Bifurcation Diagram ---
ax = axes[0]
stab_color   = '#264653'
unstab_color = '#E63946'

for kappa, fp, stable in bifurc_data:
    ax.scatter(kappa, fp,
               color=stab_color if stable else unstab_color,
               s=25, zorder=5)

# Legend proxies
ax.scatter([], [], color=stab_color,   s=40, label='Stable')
ax.scatter([], [], color=unstab_color, s=40, label='Unstable')

ax.axvline(x=0.18, color='gray', linestyle='--', alpha=0.6, linewidth=1)
ax.text(0.19, 0.05, 'κ = 0.18', fontsize=9, color='gray')
ax.set_xlabel('κ (network effect on retention)', fontsize=11)
ax.set_ylabel('Steady-state s*', fontsize=11)
ax.set_title('Bifurcation Diagram', fontsize=12, fontweight='bold')
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: Phase Diagram at κ = 0.3 ---
ax = axes[1]

# Color the f(s) curve by policy regime
prev_p = policy[0]
seg_start = 0
plotted_labels = set()

for i in range(1, len(s_grid) + 1):
    if i == len(s_grid) or policy[i] != prev_p:
        a = ACTIONS[prev_p]
        lbl = ACTION_LABELS[a]
        seg_s = s_grid[seg_start:i]
        seg_f = f_hi[seg_start:i]
        ax.plot(seg_s, seg_f,
                color=ACTION_COLORS[a], linewidth=2.5,
                label=lbl if lbl not in plotted_labels else '_nolegend_')
        plotted_labels.add(lbl)
        if i < len(s_grid):
            prev_p = policy[i]
            seg_start = i

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='s_{t+1} = s_t')

fp_marker_colors = [stab_color, unstab_color, stab_color]
for idx, (fp, stable) in enumerate(zip(fps_hi, stab_hi)):
    mc = stab_color if stable else unstab_color
    ax.scatter(fp, fp, color=mc, s=150, zorder=10)
    ax.annotate(f'{fp:.3f}', xy=(fp, fp),
                xytext=(fp + 0.03, fp - 0.08),
                fontsize=9, color=mc, fontweight='bold')

ax.set_xlabel('s_t', fontsize=11)
ax.set_ylabel('s_{t+1}', fontsize=11)
ax.set_title(f'Phase Diagram at κ = {kappa_hi}', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)

# --- Plot 3: Simulated Paths from Many Starting Points (path dependence) ---
ax = axes[2]
cmap = plt.cm.coolwarm
for j, (s0, path) in enumerate(zip(start_points, sim_paths)):
    color = cmap(j / (n_starts - 1))
    ax.plot(range(T_sim + 1), path, color=color, linewidth=1, alpha=0.7)

# Draw steady-state lines
for fp, stable in zip(fps_hi, stab_hi):
    ls = '-' if stable else '--'
    ax.axhline(fp, color=stab_color if stable else unstab_color,
               linestyle=ls, linewidth=1.5, alpha=0.8,
               label=f's* = {fp:.3f} ({"stable" if stable else "unstable"})')

ax.set_xlabel('Time t', fontsize=11)
ax.set_ylabel('Market share s_t', fontsize=11)
ax.set_title(f'Simulated Paths at κ = {kappa_hi}\n(Path Dependence)', fontsize=12, fontweight='bold')
ax.set_xlim(0, T_sim)
ax.set_ylim(0, 1)
ax.legend(fontsize=8, loc='center right')
ax.grid(True, alpha=0.3)

plt.suptitle(
    'Extension 1: Multiple Steady States & Path Dependence\n'
    r'$\rho(P,C,s) = \rho_0 + \delta_P + \delta_C + \kappa s$',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/extension1_multi_steady.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")
plt.show()
