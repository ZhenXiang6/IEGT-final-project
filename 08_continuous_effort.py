"""
OTT Streaming Platform — 延伸二: Continuous Effort Allocation
Replace binary C ∈ {A, R} with continuous retention effort e ∈ [0, 1].

Model change:
  ρ(P, e) = ρ_base(P) + δ_R · e^γ         (concave in e if γ < 1)
  α(P, e) = α_base(P) + δ̃_A · (1-e)^γ    (concave in 1-e if γ < 1)
  cost(e)  = (1-e)·c_A + e·c_R             (linear blend)

Key mathematical fact:
  γ = 1 (linear)  → objective linear in e → optimal always corner (e ∈ {0,1})
                   → degenerates back to binary baseline
  γ < 1 (concave) → diminishing marginal returns → interior solution
                   → firm optimally mixes acquisition and retention simultaneously

Findings to verify (from slides §5):
  (1) e*(s) smoothly increasing; ~50/50 at s ≈ 0.52
  (2) V_cont ≥ V_disc everywhere; average gain ≈ +12.87
  (3) Middle range of s shows largest gain (~+14)
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
c_A   = 1.2
c_R   = 0.8

rho_0   = 0.65
delta_S = 0.105
delta_T = 0.0
delta_R = 0.10

alpha_0 = 0.10
tilde_S = 0.0
tilde_T = 0.05
tilde_A = 0.15

GAMMA = 0.5          # concavity; γ < 1 gives interior solutions
N_EFFORT = 101
effort_grid = np.linspace(0, 1, N_EFFORT)
PRICES = ['S', 'T']

# ========== Discrete baseline (binary C) ==========
ACTIONS_DISC = [('T','A'), ('T','R'), ('S','A'), ('S','R')]
ACTION_LABELS = {
    ('T','A'): 'Trans-Acq', ('T','R'): 'Trans-Ret',
    ('S','A'): 'Sub-Acq',   ('S','R'): 'Sub-Ret',
}
ACTION_COLORS = {
    ('T','A'): '#E63946', ('T','R'): '#F4A261',
    ('S','A'): '#2A9D8F', ('S','R'): '#264653',
}

def revenue(s, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s)

def rho_disc(P, C):
    return rho_0 + (delta_S if P == 'S' else delta_T) + (delta_R if C == 'R' else 0.0)

def alpha_disc(P, C):
    return (alpha_0 + (tilde_S if P == 'S' else tilde_T)
            + (tilde_A if C == 'A' else 0.0))

def transition_disc(s, action):
    P, C = action
    return rho_disc(P, C) * s + alpha_disc(P, C) * (1 - s)

def profit_disc(s, action):
    P, C = action
    return revenue(s, P) * s - (c_A if C == 'A' else c_R)

# ========== Continuous effort model ==========

def rho_cont(P, e, gamma=GAMMA):
    return rho_0 + (delta_S if P == 'S' else delta_T) + delta_R * (e ** gamma)

def alpha_cont(P, e, gamma=GAMMA):
    return alpha_0 + (tilde_S if P == 'S' else tilde_T) + tilde_A * ((1 - e) ** gamma)

def transition_cont(s, P, e, gamma=GAMMA):
    return rho_cont(P, e, gamma) * s + alpha_cont(P, e, gamma) * (1 - s)

def profit_cont(s, P, e):
    cost = (1 - e) * c_A + e * c_R
    return revenue(s, P) * s - cost

# ========== Value Iteration — Discrete Baseline ==========

def value_iteration_disc(n_grid=201, tol=1e-7, max_iter=2000):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)
    for it in range(max_iter):
        V_new = np.zeros(n_grid)
        policy = np.zeros(n_grid, dtype=int)
        for i, s in enumerate(s_grid):
            Q = [profit_disc(s, a) + BETA * np.interp(
                     np.clip(transition_disc(s, a), 0, 1), s_grid, V)
                 for a in ACTIONS_DISC]
            V_new[i] = max(Q)
            policy[i] = int(np.argmax(Q))
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            print(f"  [discrete]  converged after {it+1} iterations (diff={diff:.2e})")
            break
    return s_grid, V, policy

# ========== Value Iteration — Continuous Effort ==========

def value_iteration_cont(gamma=GAMMA, n_grid=201, tol=1e-7, max_iter=2000):
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)

    # Precompute (s, P, e) transitions and profits
    sp_all = np.zeros((n_grid, 2, N_EFFORT))
    pi_all = np.zeros((n_grid, 2, N_EFFORT))
    for i, s in enumerate(s_grid):
        for pi_idx, P in enumerate(PRICES):
            for j, e in enumerate(effort_grid):
                sp_all[i, pi_idx, j] = np.clip(transition_cont(s, P, e, gamma), 0, 1)
                pi_all[i, pi_idx, j] = profit_cont(s, P, e)

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
            print(f"  [continuous] converged after {it+1} iterations (diff={diff:.2e})")
            break

    return s_grid, V, policy_P, policy_e

# ========== Run ==========

print("=" * 65)
print("延伸二: Continuous Effort Allocation")
print(f"  γ = {GAMMA}  (concavity; γ<1 → interior solution)")
print("=" * 65)

print("\nSolving discrete baseline...")
s_grid, V_disc, policy_disc = value_iteration_disc()

print("Solving continuous effort model (γ = 0.5)...")
_, V_cont, policy_P, policy_e = value_iteration_cont()

e_star = effort_grid[policy_e]
P_star = [PRICES[p] for p in policy_P]
gain   = V_cont - V_disc

print(f"\nAverage V gain: +{np.mean(gain):.3f}")
print(f"Max gain:       +{gain.max():.3f} at s = {s_grid[np.argmax(gain)]:.3f}")

print(f"\n{'s':<8} {'P*':<6} {'e*':<8} {'%Ret':<8} {'%Acq':<8} {'V_cont':<12} {'V_disc':<12} {'Gain'}")
print("-" * 72)
for sv in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(s_grid - sv))
    print(f"{s_grid[idx]:<8.2f} {P_star[idx]:<6} {e_star[idx]:<8.2f} "
          f"{e_star[idx]*100:<8.0f} {(1-e_star[idx])*100:<8.0f} "
          f"{V_cont[idx]:<12.3f} {V_disc[idx]:<12.3f} {gain[idx]:.3f}")

cross = np.argmin(np.abs(e_star - 0.5))
print(f"\n50/50 effort split at s ≈ {s_grid[cross]:.3f}")

# ========== γ sweep ==========
gamma_vals = [1.0, 0.75, 0.5, 0.25]
e_by_gamma = {}
print("\nγ sweep:")
for gv in gamma_vals:
    _, V_g, pP_g, pe_g = value_iteration_cont(gamma=gv)
    e_g = effort_grid[pe_g]
    e_by_gamma[gv] = (V_g, e_g)
    n_int = np.sum((e_g >= 0.02) & (e_g <= 0.98))
    print(f"  γ={gv:.2f}: {n_int}/201 interior solutions, mean e*={e_g.mean():.3f}")

# ========== Plot ==========

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
BLUE = '#2B4CC9'; RED = '#E63946'; TEAL = '#2A9D8F'; DARK = '#264653'; ORANGE = '#F4A261'
g_colors = {1.0: RED, 0.75: ORANGE, 0.5: BLUE, 0.25: TEAL}

# --- Plot 1: e*(s) by γ ---
ax = axes[0]
for gv in gamma_vals:
    _, e_g = e_by_gamma[gv]
    ax.plot(s_grid, e_g, color=g_colors[gv], linewidth=2,
            linestyle='--' if gv == 1.0 else '-', label=f'γ = {gv}')
ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6)
ax.text(0.02, 0.52, '50/50', fontsize=8, color='gray')
ax.axvline(0.08, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax.axvline(0.54, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
ax.text(0.09, 0.04, 's_TS', fontsize=7, color='gray')
ax.text(0.55, 0.04, 's*',   fontsize=7, color='gray')
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('Optimal retention effort e*(s)', fontsize=11)
ax.set_title('Sharp Threshold → Smooth Curve', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- Plot 2: V comparison ---
ax = axes[1]
ax.plot(s_grid, V_cont, color=BLUE, linewidth=2.5, label=f'Continuous (γ={GAMMA})')
ax.plot(s_grid, V_disc, color=RED,  linewidth=2,   linestyle='--', label='Discrete (binary)')
ylim = ax.get_ylim()
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('V(s)', fontsize=11)
ax.set_title('Value Function: Continuous vs Discrete', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- Plot 3: Gain ---
ax = axes[2]
ax.fill_between(s_grid, gain, 0, where=(gain >= 0), color=TEAL, alpha=0.4)
ax.plot(s_grid, gain, color=DARK, linewidth=2)
ax.axhline(0, color='gray', linestyle=':', linewidth=1)
ax.axhline(np.mean(gain), color=BLUE, linestyle='--', linewidth=1.2,
           label=f'Mean = +{np.mean(gain):.2f}')
idx_max = np.argmax(gain)
ax.scatter(s_grid[idx_max], gain[idx_max], color=TEAL, s=80, zorder=6)
ax.annotate(f'max +{gain[idx_max]:.1f}\n(s={s_grid[idx_max]:.2f})',
            xy=(s_grid[idx_max], gain[idx_max]),
            xytext=(s_grid[idx_max]+0.06, gain[idx_max]-1.5),
            fontsize=8, color=DARK)
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('V_cont − V_disc', fontsize=11)
ax.set_title('Value Gain from Continuous Control', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.suptitle(
    '延伸二: Continuous Effort Allocation\n'
    r'$e\in[0,1]$: retention effort; '
    r'$\rho(P,e)=\rho_{\rm base}+\delta_R e^\gamma$, '
    r'$\alpha(P,e)=\alpha_{\rm base}+\tilde\delta_A(1-e)^\gamma$',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/continuous_effort.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")
plt.show()
