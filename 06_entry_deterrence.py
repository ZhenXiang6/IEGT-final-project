"""
OTT Streaming Platform — Extension 3: Entry Deterrence Threshold
Find the critical s_N* where V(0) = 0: the point beyond which rational entry is impossible.

Key question:
  Netflix's fixed market share s_N acts as a structural entry barrier — not through
  predatory pricing or exclusionary conduct, but purely via market-space compression.
  We sweep s_N ∈ [0, 0.95] and compute V(0) under optimal policy.
  The threshold s_N* where V(0) = 0 is the entry deterrence boundary.

Model identical to 05_competitor_fixed.py. The novel contribution is asking:
  at what s_N does even the best possible challenger strategy yield negative NPV?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ========== Parameters (identical to 05) ==========
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

def revenue(s_C, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s_C)

def cost(C):
    return c_A if C == 'A' else c_R

def rho_val(P, C):
    dP = delta_S if P == 'S' else delta_T
    dC = delta_R if C == 'R' else delta_A
    return rho_0 + dP + dC

def alpha_val(P, C):
    tP = tilde_S if P == 'S' else tilde_T
    tC = tilde_A if C == 'A' else tilde_R
    return alpha_0 + tP + tC

def transition(s_C, action, s_N):
    P, C = action
    available = max(1.0 - s_N - s_C, 0.0)
    return rho_val(P, C) * s_C + alpha_val(P, C) * available

def profit(s_C, action):
    P, C = action
    return revenue(s_C, P) * s_C - cost(C)

# ========== Value Iteration ==========

def value_iteration(s_N, n_grid=201, tol=1e-7, max_iter=2000):
    s_max = 1.0 - s_N
    if s_max <= 0:
        return np.array([0.0]), np.array([0.0]), np.array([0], dtype=int)
    s_grid = np.linspace(0, s_max, n_grid)
    V = np.zeros(n_grid)

    for _ in range(max_iter):
        V_new = np.zeros(n_grid)
        policy = np.zeros(n_grid, dtype=int)
        for i, s_C in enumerate(s_grid):
            Q = []
            for a in ACTIONS:
                sp = np.clip(transition(s_C, a, s_N), 0.0, s_max)
                Q.append(profit(s_C, a) + BETA * np.interp(sp, s_grid, V))
            V_new[i] = max(Q)
            policy[i] = int(np.argmax(Q))
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new

    return s_grid, V_new, policy

# ========== Sweep s_N ==========

s_N_sweep = np.round(np.arange(0.0, 0.96, 0.01), 2)

print("=" * 65)
print("Extension 3: Entry Deterrence Threshold")
print("Sweeping s_N ∈ [0, 0.95], computing V(0) under optimal policy")
print("=" * 65)

V0_vals     = []
steady_state = []

for s_N in s_N_sweep:
    s_grid, V, policy = value_iteration(s_N)
    V0 = float(V[0])
    V0_vals.append(V0)

    # Find steady-state (last action is Sub-Ret region upper end)
    s_star = s_grid[-1]   # default: ceiling
    for i in range(len(policy) - 2, 0, -1):
        if ACTIONS[policy[i]] == ('S', 'A') and ACTIONS[policy[i+1]] == ('S', 'R'):
            s_star = s_grid[i+1]
            break
    steady_state.append(float(s_star))

V0_vals      = np.array(V0_vals)
steady_state = np.array(steady_state)

# ========== Find s_N* where V(0) = 0 ==========

threshold_sN = None
for i in range(len(V0_vals) - 1):
    if V0_vals[i] >= 0 and V0_vals[i+1] < 0:
        # Linear interpolation
        denom = V0_vals[i+1] - V0_vals[i]
        threshold_sN = s_N_sweep[i] - V0_vals[i] * (s_N_sweep[i+1] - s_N_sweep[i]) / denom
        break

print(f"\nV(0) at s_N = 0.00 : {V0_vals[0]:.3f}")
print(f"V(0) at s_N = 0.30 : {V0_vals[30]:.3f}")
print(f"V(0) at s_N = 0.50 : {V0_vals[50]:.3f}")
print(f"V(0) at s_N = 0.70 : {V0_vals[70]:.3f}")

if threshold_sN is not None:
    print(f"\n>>> Entry deterrence threshold s_N* = {threshold_sN:.3f}")
    print(f"    For s_N > {threshold_sN:.3f}, V(0) < 0 — rational entry is blocked.")
else:
    print("\n>>> V(0) does not cross 0 in [0, 0.95].")
    print(f"    V(0) range: [{V0_vals.min():.3f}, {V0_vals.max():.3f}]")

# ========== Plot ==========

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ENTRY_COLOR  = '#2A9D8F'
DETER_COLOR  = '#E63946'
THRESH_COLOR = '#264653'

# --- Plot 1: V(0) vs s_N ---
ax = axes[0]

if threshold_sN is not None:
    ax.axvspan(0, threshold_sN, alpha=0.08, color=ENTRY_COLOR, label='Entry viable')
    ax.axvspan(threshold_sN, s_N_sweep[-1], alpha=0.08, color=DETER_COLOR, label='Entry deterred')
    ax.axvline(threshold_sN, color=THRESH_COLOR, linestyle='--', linewidth=1.5)
    ax.text(threshold_sN + 0.01, ax.get_ylim()[0] if hasattr(ax, '_get_ylim') else 0,
            f's_N* = {threshold_sN:.3f}', fontsize=9, color=THRESH_COLOR, va='bottom')

ax.plot(s_N_sweep, V0_vals, 'k-', linewidth=2.5, zorder=5)
ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

ax.set_xlabel("Netflix market share s_N", fontsize=11)
ax.set_ylabel("V(0)  — challenger NPV from zero share", fontsize=10)
ax.set_title("Entry Deterrence: V(0) vs s_N", fontsize=12, fontweight='bold')
ax.set_xlim(0, s_N_sweep[-1])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate threshold after ylim is set
if threshold_sN is not None:
    ylim = ax.get_ylim()
    ax.text(threshold_sN + 0.01, ylim[0] + 0.03 * (ylim[1] - ylim[0]),
            f's_N* ≈ {threshold_sN:.3f}', fontsize=9, color=THRESH_COLOR, va='bottom')

# --- Plot 2: Steady-state ceiling vs s_N ---
ax = axes[1]
available = 1.0 - s_N_sweep
ax.plot(s_N_sweep, available, 'k--', linewidth=1.5, alpha=0.5, label='Market ceiling (1 - s_N)')
ax.plot(s_N_sweep, steady_state, color='#2A9D8F', linewidth=2.5, label='Steady-state s*')

if threshold_sN is not None:
    ax.axvline(threshold_sN, color=THRESH_COLOR, linestyle='--', linewidth=1.5,
               label=f's_N* = {threshold_sN:.3f}')

ax.set_xlabel("Netflix market share s_N", fontsize=11)
ax.set_ylabel("Market share", fontsize=11)
ax.set_title("Steady-state vs Market Ceiling", fontsize=12, fontweight='bold')
ax.set_xlim(0, s_N_sweep[-1])
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: V(0) decomposition — break-even intuition ---
ax = axes[2]

# Approximate NPV of steady-state revenue vs fixed entry cost
# steady-state profit ≈ p_S * s* - c_R  (Sub-Ret in steady state)
# discounted sum ≈ π_ss / (1 - β)
pi_ss  = np.array([p_S * ss - c_R for ss in steady_state])
npv_ss = pi_ss / (1 - BETA)   # perpetuity approximation

ax.plot(s_N_sweep, V0_vals, 'k-',  linewidth=2.5, label='V(0) exact (value iteration)', zorder=5)
ax.plot(s_N_sweep, npv_ss,  color='#F4A261', linewidth=1.8,
        linestyle='--', label='Steady-state NPV approx (π_ss / (1-β))', alpha=0.8)
ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

if threshold_sN is not None:
    ax.axvline(threshold_sN, color=THRESH_COLOR, linestyle='--', linewidth=1.5,
               label=f's_N* = {threshold_sN:.3f}')

ax.set_xlabel("Netflix market share s_N", fontsize=11)
ax.set_ylabel("NPV", fontsize=11)
ax.set_title("V(0) vs Steady-state Perpetuity Approx", fontsize=12, fontweight='bold')
ax.set_xlim(0, s_N_sweep[-1])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(
    'Extension 3: Structural Entry Deterrence\n'
    r'$V(0; s_N)$: Challenger NPV from zero market share under optimal policy',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/entry_deterrence.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")

# ========== Summary table ==========
print("\n" + "=" * 65)
print("Summary: V(0) and steady-state at selected s_N values")
print(f"{'s_N':<8} {'Available':<12} {'s*':<12} {'V(0)':<12} {'Entry?'}")
print("-" * 58)
report_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for sn in report_vals:
    idx = int(round(sn / 0.01))
    if idx < len(s_N_sweep):
        viable = "Yes" if V0_vals[idx] >= 0 else "No"
        print(f"{s_N_sweep[idx]:<8.2f} {1-s_N_sweep[idx]:<12.2f} "
              f"{steady_state[idx]:<12.3f} {V0_vals[idx]:<12.3f} {viable}")

if threshold_sN is not None:
    print(f"\nEntry deterrence threshold s_N* = {threshold_sN:.4f}")

plt.show()
