"""
OTT Streaming Platform — Competitor Extension (Option A)
Netflix holds fixed market share s_N (exogenous).
Challenger optimizes over the reduced non-user pool (1 - s_N - s_C).

Key change vs baseline:
  transition: s_{C,t+1} = ρ · s_C + α · (1 - s_N - s_C)
  state range: s_C ∈ [0, 1 - s_N]

Also applies the alpha bug fix: α does not depend on s.
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

def revenue(s_C, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s_C)

def cost(C):
    return c_A if C == 'A' else c_R

def rho_val(P, C):
    dP = delta_S if P == 'S' else delta_T
    dC = delta_R if C == 'R' else delta_A
    return rho_0 + dP + dC

def alpha_val(P, C):
    """Acquisition rate — constant, does NOT depend on s_C."""
    tP = tilde_S if P == 'S' else tilde_T
    tC = tilde_A if C == 'A' else tilde_R
    return alpha_0 + tP + tC

def transition(s_C, action, s_N):
    """
    s_{C,t+1} = ρ · s_C + α · (1 - s_N - s_C)
    Non-user pool shrunk by Netflix's s_N.
    """
    P, C = action
    available = max(1.0 - s_N - s_C, 0.0)   # remaining non-users
    return rho_val(P, C) * s_C + alpha_val(P, C) * available

def profit(s_C, action):
    P, C = action
    return revenue(s_C, P) * s_C - cost(C)

# ========== Value Iteration ==========

def value_iteration(s_N, n_grid=201, tol=1e-7, max_iter=2000):
    """
    Solve challenger's Bellman over s_C ∈ [0, 1 - s_N].
    """
    s_max = 1.0 - s_N
    s_grid = np.linspace(0, s_max, n_grid)
    V = np.zeros(n_grid)

    for it in range(max_iter):
        V_new = np.zeros(n_grid)
        policy = np.zeros(n_grid, dtype=int)
        for i, s_C in enumerate(s_grid):
            Q = []
            for a in ACTIONS:
                sp = np.clip(transition(s_C, a, s_N), 0.0, s_max)
                Q.append(profit(s_C, a) + BETA * np.interp(sp, s_grid, V))
            V_new[i] = max(Q)
            policy[i] = int(np.argmax(Q))
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break

    return s_grid, V, policy

def simulate(s0, T, s_grid, policy, s_N):
    path = [s0]
    s = s0
    s_max = 1.0 - s_N
    for _ in range(T):
        i = np.argmin(np.abs(s_grid - s))
        a = ACTIONS[policy[i]]
        s = np.clip(transition(s, a, s_N), 0.0, s_max)
        path.append(s)
    return path

# ========== Run ==========

# Netflix market share scenarios
s_N_vals = [0.0, 0.20, 0.35, 0.50]
s_N_labels = {
    0.0:  'No incumbent (s_N = 0)',
    0.20: 'Weak incumbent (s_N = 0.20)',
    0.35: 'Netflix-like (s_N = 0.35)',
    0.50: 'Dominant incumbent (s_N = 0.50)',
}

print("=" * 65)
print("Competitor Extension (Option A): Fixed Netflix Market Share")
print("=" * 65)

results = {}
for s_N in s_N_vals:
    s_grid, V, policy = value_iteration(s_N)
    results[s_N] = (s_grid, V, policy)

    print(f"\ns_N = {s_N:.2f}  (available market ceiling = {1-s_N:.2f})")
    print(f"  {'s_C range':<20} {'Optimal action'}")
    print("  " + "-" * 38)
    cur = policy[0]; start = 0
    for i in range(1, len(s_grid)):
        if policy[i] != cur:
            print(f"  [{s_grid[start]:.3f}, {s_grid[i-1]:.3f}]"
                  f"         {ACTION_LABELS[ACTIONS[cur]]}")
            cur = policy[i]; start = i
    print(f"  [{s_grid[start]:.3f}, {s_grid[-1]:.3f}]"
          f"         {ACTION_LABELS[ACTIONS[cur]]}")

# ========== Plot ==========

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

T_SIM = 30
s0_challenger = 0.05   # challenger starts small (Apple TV+ type)

for ax, s_N in zip(axes, s_N_vals):
    s_grid, V, policy = results[s_N]
    s_max = 1.0 - s_N

    # --- Policy strip ---
    colors = [ACTION_COLORS[ACTIONS[p]] for p in policy]
    ax2 = ax.twinx()
    ax2.scatter(s_grid, [1] * len(s_grid),
                c=colors, s=60, marker='|', linewidths=6, alpha=0.8)
    ax2.set_ylim(0.5, 1.5)
    ax2.set_yticks([])

    # --- Value function ---
    ax.plot(s_grid, V, 'b-', linewidth=2, label='V(s_C)', zorder=3)
    ax.set_xlim(0, s_max + 0.02)
    ax.set_xlabel('Challenger market share s_C', fontsize=10)
    ax.set_ylabel('V(s_C)', fontsize=10, color='b')
    ax.tick_params(axis='y', labelcolor='b')

    # --- Simulate path ---
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 55))
    sim = simulate(s0_challenger, T_SIM, s_grid, policy, s_N)
    ax3.plot(np.array(sim) , color='#E63946', linewidth=1.5,
             linestyle='--', label=f's_C path (s0={s0_challenger})', alpha=0.0)
    ax3.set_yticks([])

    # Annotate ceiling
    ax.axvline(s_max, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(s_max - 0.01, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
            f'ceiling\n{s_max:.2f}', ha='right', fontsize=8, color='gray')

    ax.set_title(s_N_labels[s_N], fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Shared legend for policy colours
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=c, markersize=11, label=ACTION_LABELS[a])
           for a, c in ACTION_COLORS.items()]
fig.legend(handles=handles, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))

plt.suptitle(
    'Competitor Extension: How Netflix\'s Fixed Market Share\n'
    'Reshapes the Challenger\'s Optimal Strategy',
    fontsize=13, fontweight='bold'
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/competitor_fixed.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")

# ========== Summary table ==========
print("\n" + "=" * 65)
print("Threshold shift summary")
print(f"{'s_N':<8} {'Available':<12} {'s_TS threshold':<18} {'s* threshold'}")
print("-" * 55)
for s_N in s_N_vals:
    s_grid, V, policy = results[s_N]
    # Find first switch from Trans-Acq → Sub-Acq
    s_TS = None
    for i in range(len(policy) - 1):
        if ACTIONS[policy[i]] == ('T','A') and ACTIONS[policy[i+1]] != ('T','A'):
            s_TS = s_grid[i+1]
            break
    # Find switch to Sub-Ret
    s_star = None
    for i in range(len(policy) - 1):
        if ACTIONS[policy[i]] == ('S','A') and ACTIONS[policy[i+1]] == ('S','R'):
            s_star = s_grid[i+1]
            break
    s_TS_str  = f"{s_TS:.3f}"  if s_TS  is not None else "N/A"
    s_star_str = f"{s_star:.3f}" if s_star is not None else "N/A"
    print(f"{s_N:<8.2f} {1-s_N:<12.2f} {s_TS_str:<18} {s_star_str}")

plt.show()
