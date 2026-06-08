"""
OTT Streaming Platform — 延伸三: Cost Design Deepening
Replace flat costs with scale/saturation-dependent versions (Min et al. 2016):
  c_A(s) = 1.2 * (1 + s)        acquisition more expensive as market saturates
  c_R(s) = 0.8 * (1 + 0.3 * s)  retention scales with customer base

At s=1: c_A doubles to 2.40; c_R rises to 1.04.

Key findings:
  (1) Main threshold moves: 0.541 → 0.489 (↓ 0.052)
  (2) Discrete steady state: 0.505 → 0.490 (↓ 0.015)
  (3) Continuous control more robust: steady state ↓ only 0.006
  (4) Dimension separation: cost change only shifts A/R boundary; T/S boundary nearly fixed
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ========== Parameters (identical to baseline 03) ==========
BETA  = 0.95
p_S   = 15.0
p_T   = 5.0
phi0  = 0.5
phi1  = 0.5

rho_0   = 0.65
delta_S = 0.105
delta_T = 0.0
delta_R = 0.10

alpha_0 = 0.10
tilde_S = 0.0
tilde_T = 0.05
tilde_A = 0.15

GAMMA = 0.5
N_EFFORT = 101
effort_grid = np.linspace(0, 1, N_EFFORT)
PRICES = ['S', 'T']

ACTIONS = [('T','A'), ('T','R'), ('S','A'), ('S','R')]
ACTION_LABELS = {
    ('T','A'): 'Trans-Acq', ('T','R'): 'Trans-Ret',
    ('S','A'): 'Sub-Acq',   ('S','R'): 'Sub-Ret',
}
ACTION_COLORS = {
    ('T','A'): '#E63946', ('T','R'): '#F4A261',
    ('S','A'): '#2A9D8F', ('S','R'): '#264653',
}

# ========== Cost functions ==========
c_A_FLAT = 1.2
c_R_FLAT = 0.8

def c_A_sdep(s):
    return 1.2 * (1 + s)           # 1.20 at s=0; 2.40 at s=1

def c_R_sdep(s):
    return 0.8 * (1 + 0.3 * s)    # 0.80 at s=0; 1.04 at s=1

# ========== Primitives ==========
def revenue(s, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s)

def rho(P, C):
    return rho_0 + (delta_S if P == 'S' else delta_T) + (delta_R if C == 'R' else 0.0)

def alpha(P, C):
    return alpha_0 + (tilde_S if P == 'S' else tilde_T) + (tilde_A if C == 'A' else 0.0)

def transition(s, action):
    P, C = action
    return rho(P, C) * s + alpha(P, C) * (1 - s)

def profit_flat(s, action):
    P, C = action
    return revenue(s, P) * s - (c_A_FLAT if C == 'A' else c_R_FLAT)

def profit_sdep(s, action):
    P, C = action
    return revenue(s, P) * s - (c_A_sdep(s) if C == 'A' else c_R_sdep(s))

# ========== Value Iteration — Discrete ==========
def value_iteration_disc(profit_fn, label, n_grid=201, tol=1e-7, max_iter=2000):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)
    for it in range(max_iter):
        V_new   = np.zeros(n_grid)
        policy  = np.zeros(n_grid, dtype=int)
        for i, s in enumerate(s_grid):
            Q = [profit_fn(s, a) + BETA * np.interp(
                     np.clip(transition(s, a), 0, 1), s_grid, V)
                 for a in ACTIONS]
            V_new[i]  = max(Q)
            policy[i] = int(np.argmax(Q))
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            print(f"  [{label}] converged after {it+1} iterations (diff={diff:.2e})")
            break
    return s_grid, V, policy

# ========== Value Iteration — Continuous Effort with s-dep costs ==========
def value_iteration_cont_sdep(gamma=GAMMA, n_grid=201, tol=1e-7, max_iter=2000):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)

    # Precompute transitions (costs don't affect transitions)
    sp_all = np.zeros((n_grid, 2, N_EFFORT))
    pi_all = np.zeros((n_grid, 2, N_EFFORT))
    for i, s in enumerate(s_grid):
        for pi_idx, P in enumerate(PRICES):
            for j, e in enumerate(effort_grid):
                rho_e   = rho_0 + (delta_S if P=='S' else delta_T) + delta_R * (e ** gamma)
                alpha_e = alpha_0 + (tilde_S if P=='S' else tilde_T) + tilde_A * ((1-e)**gamma)
                sp_all[i, pi_idx, j] = np.clip(rho_e*s + alpha_e*(1-s), 0, 1)
                cost_e = (1-e) * c_A_sdep(s) + e * c_R_sdep(s)
                pi_all[i, pi_idx, j] = revenue(s, P) * s - cost_e

    policy_P = np.zeros(n_grid, dtype=int)
    policy_e = np.zeros(n_grid, dtype=int)

    for it in range(max_iter):
        V_new     = np.zeros(n_grid)
        pol_P_new = np.zeros(n_grid, dtype=int)
        pol_e_new = np.zeros(n_grid, dtype=int)
        for i in range(n_grid):
            best_q = -np.inf
            best_pi = best_ej = 0
            for pi_idx in range(2):
                V_sp = np.interp(sp_all[i, pi_idx, :], s_grid, V)
                Q    = pi_all[i, pi_idx, :] + BETA * V_sp
                idx  = int(np.argmax(Q))
                if Q[idx] > best_q:
                    best_q = Q[idx]; best_pi = pi_idx; best_ej = idx
            V_new[i]     = best_q
            pol_P_new[i] = best_pi
            pol_e_new[i] = best_ej
        diff = np.max(np.abs(V_new - V))
        V = V_new; policy_P = pol_P_new; policy_e = pol_e_new
        if diff < tol:
            print(f"  [cont-sdep] converged after {it+1} iterations (diff={diff:.2e})")
            break
    return s_grid, V, policy_P, policy_e

# ========== Helpers ==========
def find_thresholds(s_grid, policy):
    thresholds = []
    for i in range(1, len(policy)):
        if policy[i] != policy[i-1]:
            thresholds.append((s_grid[i], ACTIONS[policy[i-1]], ACTIONS[policy[i]]))
    return thresholds

def simulate_disc(s_grid, policy, s0, T=100):
    s = s0
    path = [s]
    for _ in range(T):
        idx = np.argmin(np.abs(s_grid - s))
        a   = ACTIONS[policy[idx]]
        s   = float(np.clip(transition(s, a), 0, 1))
        path.append(s)
    return np.array(path)

def simulate_cont(s_grid, policy_P, policy_e, s0, T=100, gamma=GAMMA):
    s = s0
    path = [s]
    for _ in range(T):
        idx  = np.argmin(np.abs(s_grid - s))
        P    = PRICES[policy_P[idx]]
        e    = effort_grid[policy_e[idx]]
        rho_e   = rho_0 + (delta_S if P=='S' else delta_T) + delta_R*(e**gamma)
        alpha_e = alpha_0 + (tilde_S if P=='S' else tilde_T) + tilde_A*((1-e)**gamma)
        s = float(np.clip(rho_e*s + alpha_e*(1-s), 0, 1))
        path.append(s)
    return np.array(path)

# ========== Run ==========
print("=" * 65)
print("延伸三: Cost Design Deepening (Min et al. 2016)")
print("  c_A(s) = 1.2*(1+s)  →  1.20 at s=0, 2.40 at s=1")
print("  c_R(s) = 0.8*(1+0.3s) →  0.80 at s=0, 1.04 at s=1")
print("=" * 65)

print("\nSolving discrete baseline (flat costs)...")
s_grid, V_flat, policy_flat = value_iteration_disc(profit_flat, 'flat')

print("Solving discrete s-dependent cost model...")
_, V_sdep, policy_sdep = value_iteration_disc(profit_sdep, 'sdep')

print("Solving continuous effort model (s-dep costs, gamma=0.5)...")
_, V_cont, pP_cont, pe_cont = value_iteration_cont_sdep()
e_star = effort_grid[pe_cont]

# --- Thresholds ---
thr_flat = find_thresholds(s_grid, policy_flat)
thr_sdep = find_thresholds(s_grid, policy_sdep)
print("\n=== Policy Thresholds ===")
for t in thr_flat:
    print(f"  [flat]  {ACTION_LABELS[t[1]]:12s} -> {ACTION_LABELS[t[2]]:12s}  at s = {t[0]:.3f}")
for t in thr_sdep:
    print(f"  [sdep]  {ACTION_LABELS[t[1]]:12s} -> {ACTION_LABELS[t[2]]:12s}  at s = {t[0]:.3f}")

# --- Steady states ---
T_SIM = 150
path_flat_sim = simulate_disc(s_grid, policy_flat, 0.1, T=T_SIM)
path_sdep_sim = simulate_disc(s_grid, policy_sdep, 0.1, T=T_SIM)
path_cont_sim = simulate_cont(s_grid, pP_cont, pe_cont, 0.1, T=T_SIM)
ss_flat = path_flat_sim[-1]
ss_sdep = path_sdep_sim[-1]
ss_cont = path_cont_sim[-1]

print(f"\n=== Steady States (from s0=0.1) ===")
print(f"  Discrete flat:      {ss_flat:.3f}")
print(f"  Discrete s-dep:     {ss_sdep:.3f}  (delta = {ss_sdep - ss_flat:+.3f})")
print(f"  Continuous s-dep:   {ss_cont:.3f}  (delta = {ss_cont - ss_flat:+.3f})")

# --- Dimension separation ---
diff_mask = policy_flat != policy_sdep
if np.any(diff_mask):
    # Check which pricing (T/S) changes
    P_flat_changed = set(ACTIONS[policy_flat[i]][0] for i in range(len(s_grid)) if diff_mask[i])
    P_sdep_changed = set(ACTIONS[policy_sdep[i]][0] for i in range(len(s_grid)) if diff_mask[i])
    C_flat_changed = set(ACTIONS[policy_flat[i]][1] for i in range(len(s_grid)) if diff_mask[i])
    C_sdep_changed = set(ACTIONS[policy_sdep[i]][1] for i in range(len(s_grid)) if diff_mask[i])
    thr_P_flat = set(t[1][0] for t in thr_flat) | set(t[2][0] for t in thr_flat)
    thr_P_sdep = set(t[1][0] for t in thr_sdep) | set(t[2][0] for t in thr_sdep)
    print(f"\n=== Dimension Separation ===")
    print(f"  Policy differs at {np.sum(diff_mask)}/{len(s_grid)} grid points")
    print(f"  T/S boundary (flat): {[t[0] for t in thr_flat if t[1][0]!=t[2][0] or t[1][0]=='T']}")
    print(f"  T/S boundary (sdep): {[t[0] for t in thr_sdep if t[1][0]!=t[2][0] or t[1][0]=='T']}")
    print(f"  A/R boundary (flat): {[t[0] for t in thr_flat if t[1][1]!=t[2][1]]}")
    print(f"  A/R boundary (sdep): {[t[0] for t in thr_sdep if t[1][1]!=t[2][1]]}")

# --- Table ---
print(f"\n{'s':<8} {'Flat P*':<12} {'Sdep P*':<12} {'V_flat':<12} {'V_sdep':<12} {'Gain'}")
print("-" * 68)
for sv in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx  = np.argmin(np.abs(s_grid - sv))
    af   = ACTION_LABELS[ACTIONS[policy_flat[idx]]]
    asd  = ACTION_LABELS[ACTIONS[policy_sdep[idx]]]
    print(f"{s_grid[idx]:<8.2f} {af:<12} {asd:<12} "
          f"{V_flat[idx]:<12.3f} {V_sdep[idx]:<12.3f} {V_sdep[idx]-V_flat[idx]:.3f}")

# ========== Plot ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
BLUE = '#2B4CC9'; RED = '#E63946'; TEAL = '#2A9D8F'; DARK = '#264653'; ORANGE = '#F4A261'

# --- Panel 1: Cost functions ---
ax = axes[0]
s_line = np.linspace(0, 1, 200)
ax.plot(s_line, [c_A_sdep(s) for s in s_line], color=RED,  linewidth=2.2, label=r'$c_A(s)=1.2(1+s)$')
ax.plot(s_line, [c_R_sdep(s) for s in s_line], color=TEAL, linewidth=2.2, label=r'$c_R(s)=0.8(1+0.3s)$')
ax.axhline(c_A_FLAT, color=RED,  linestyle='--', linewidth=1.2, alpha=0.5, label=r'$c_A$ flat = 1.2')
ax.axhline(c_R_FLAT, color=TEAL, linestyle='--', linewidth=1.2, alpha=0.5, label=r'$c_R$ flat = 0.8')
ax.annotate('×2 at s=1', xy=(1.0, c_A_sdep(1.0)), xytext=(0.75, 2.3),
            fontsize=8, color=RED, arrowprops=dict(arrowstyle='->', color=RED))
ax.annotate('+30% at s=1', xy=(1.0, c_R_sdep(1.0)), xytext=(0.6, 1.1),
            fontsize=8, color=TEAL, arrowprops=dict(arrowstyle='->', color=TEAL))
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('Cost', fontsize=11)
ax.set_title('Scale/Saturation-Dependent Costs\n(Min et al. 2016)', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0.6, 2.7)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- Panel 2: Policy map comparison ---
ax = axes[1]
cmap = {0: RED, 1: ORANGE, 2: TEAL, 3: DARK}
y_f = 0.65; y_s = 0.35
h   = 0.12
for i in range(len(s_grid) - 1):
    ax.fill_between([s_grid[i], s_grid[i+1]],
                    [y_f - h]*2, [y_f + h]*2, color=cmap[policy_flat[i]], alpha=0.85)
    ax.fill_between([s_grid[i], s_grid[i+1]],
                    [y_s - h]*2, [y_s + h]*2, color=cmap[policy_sdep[i]], alpha=0.85)

ax.text(-0.04, y_f, 'Flat', va='center', fontsize=9, color=BLUE, fontweight='bold')
ax.text(-0.04, y_s, 'S-dep', va='center', fontsize=9, color=RED, fontweight='bold')

for t in thr_flat:
    ax.axvline(t[0], color=BLUE, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(t[0]+0.01, y_f + h + 0.04, f'{t[0]:.3f}', fontsize=8, color=BLUE)
for t in thr_sdep:
    ax.axvline(t[0], color=RED, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(t[0]+0.01, y_s - h - 0.09, f'{t[0]:.3f}', fontsize=8, color=RED)

from matplotlib.patches import Patch
patches = [Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a]) for a in ACTIONS]
ax.legend(handles=patches, fontsize=8, loc='upper left', framealpha=0.8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_yticks([])
ax.set_xlabel('Market share s', fontsize=11)
ax.set_title('Policy Map: Threshold Shift\n(A/R boundary; T/S nearly fixed)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.2)

# --- Panel 3: Simulation paths ---
ax = axes[2]
t_axis = np.arange(T_SIM + 1)
ax.plot(t_axis, path_flat_sim, color=BLUE,   linewidth=2,   linestyle='--',
        label=f'Discrete flat  (s∞≈{ss_flat:.3f})')
ax.plot(t_axis, path_sdep_sim, color=RED,    linewidth=2,
        label=f'Discrete s-dep (s∞≈{ss_sdep:.3f}, Δ={ss_sdep-ss_flat:+.3f})')
ax.plot(t_axis, path_cont_sim, color=TEAL,   linewidth=2,   linestyle=':',
        label=f'Cont. s-dep    (s∞≈{ss_cont:.3f}, Δ={ss_cont-ss_flat:+.3f})')
ax.axhline(ss_flat, color=BLUE,  linestyle=':', linewidth=0.8, alpha=0.5)
ax.axhline(ss_sdep, color=RED,   linestyle=':', linewidth=0.8, alpha=0.5)
ax.axhline(ss_cont, color=TEAL,  linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Period', fontsize=11)
ax.set_ylabel('Market share s', fontsize=11)
ax.set_title('Steady-State Comparison\n(Continuous control more robust)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, T_SIM)

plt.suptitle(
    '延伸三: Cost Design Deepening (Min et al. 2016)\n'
    r'$c_A(s)=1.2(1+s)$, $c_R(s)=0.8(1+0.3s)$  —  threshold $0.541\to0.489$; dimension separation',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/cost_extension.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")
plt.show()
