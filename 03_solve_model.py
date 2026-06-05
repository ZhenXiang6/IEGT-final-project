"""
OTT Streaming Platform Two-Dimensional Strategic Model
Value Iteration Solver

State: s ∈ [0, 1] (market share)
Action: (P, C) ∈ {T, S} × {A, R}
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ========== Parameters (calibrated from literature) ==========
BETA = 0.95          # discount factor
p_S = 15.0           # SVOD monthly fee
p_T = 5.0            # TVOD per-purchase
phi0 = 0.5           # baseline purchase frequency
phi1 = 0.5           # network effect on purchase frequency
c_A = 1.2            # acquisition cost
c_R = 0.8            # retention cost

# Retention parameters
rho_0    = 0.65      # baseline retention
delta_S  = 0.105     # SVOD retention boost (Iyengar et al. 2011)
delta_T  = 0.0
delta_R  = 0.10      # retention strategy boost
delta_A  = 0.0

# Acquisition parameters
alpha_0  = 0.10      # baseline acquisition
tilde_S  = 0.0
tilde_T  = 0.05      # TVOD acquisition advantage (low commitment)
tilde_A  = 0.15      # acquisition strategy boost
tilde_R  = 0.0
theta    = 1.0       # saturation exponent

# Actions
ACTIONS = [('T','A'), ('T','R'), ('S','A'), ('S','R')]
ACTION_LABELS = {
    ('T','A'): 'Trans-Acq',
    ('T','R'): 'Trans-Ret',
    ('S','A'): 'Sub-Acq',
    ('S','R'): 'Sub-Ret',
}
ACTION_COLORS = {
    ('T','A'): '#E63946',  # red
    ('T','R'): '#F4A261',  # orange
    ('S','A'): '#2A9D8F',  # teal
    ('S','R'): '#264653',  # dark blue/green
}

# ========== Model primitives ==========

def revenue(s, P):
    """Per-customer revenue under pricing model P."""
    if P == 'S':
        return p_S
    elif P == 'T':
        return p_T * (phi0 + phi1 * s)

def cost(C):
    """Per-period marketing cost under customer strategy C."""
    return c_A if C == 'A' else c_R

def rho(P, C):
    """Retention rate."""
    dP = delta_S if P == 'S' else delta_T
    dC = delta_R if C == 'R' else delta_A
    return rho_0 + dP + dC

def alpha(s, P, C):
    """Acquisition rate (depends on s via saturation)."""
    tP = tilde_S if P == 'S' else tilde_T
    tC = tilde_A if C == 'A' else tilde_R
    return (alpha_0 + tP + tC) * (1 - s ** theta)

def transition(s, action):
    """s' = ρ·s + α·(1-s)."""
    P, C = action
    return rho(P, C) * s + alpha(s, P, C) * (1 - s)

def profit(s, action):
    """Per-period profit π(s, a) = R(s,P)·s − C(C)."""
    P, C = action
    return revenue(s, P) * s - cost(C)

# ========== Value Iteration ==========

def value_iteration(n_grid=201, tol=1e-7, max_iter=2000):
    """Solve Bellman equation via value iteration with linear interpolation."""
    s_grid = np.linspace(0, 1, n_grid)
    V = np.zeros(n_grid)
    
    for it in range(max_iter):
        V_new = np.zeros(n_grid)
        policy = np.zeros(n_grid, dtype=int)
        
        for i, s in enumerate(s_grid):
            Q_values = []
            for a_idx, a in enumerate(ACTIONS):
                s_prime = transition(s, a)
                s_prime = np.clip(s_prime, 0, 1)
                # Linear interpolation of V at s'
                V_sp = np.interp(s_prime, s_grid, V)
                Q = profit(s, a) + BETA * V_sp
                Q_values.append(Q)
            V_new[i] = max(Q_values)
            policy[i] = int(np.argmax(Q_values))
        
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            print(f"Converged after {it+1} iterations (diff = {diff:.2e})")
            break
    else:
        print(f"Reached max_iter = {max_iter} (diff = {diff:.2e})")
    
    return s_grid, V, policy

# ========== Simulate optimal path ==========

def simulate(s0, T, s_grid, policy):
    """Simulate market share path from s0 under optimal policy."""
    s_path = [s0]
    a_path = []
    pi_path = []
    s = s0
    for t in range(T):
        i = np.argmin(np.abs(s_grid - s))
        a_idx = policy[i]
        a = ACTIONS[a_idx]
        a_path.append(a)
        pi_path.append(profit(s, a))
        s = transition(s, a)
        s = np.clip(s, 0, 1)
        s_path.append(s)
    return s_path, a_path, pi_path

# ========== Run ==========

print("=" * 60)
print("OTT Two-Dimensional Strategic Model — Value Iteration")
print("=" * 60)

s_grid, V, policy = value_iteration()

# Print policy summary
print("\n--- Optimal Policy a*(s) ---")
print(f"{'Range of s':<20} {'Optimal action':<15}")
print("-" * 35)
current_a = policy[0]
range_start = 0
for i in range(1, len(s_grid)):
    if policy[i] != current_a:
        print(f"[{s_grid[range_start]:.3f}, {s_grid[i-1]:.3f}]     "
              f"{ACTION_LABELS[ACTIONS[current_a]]}")
        current_a = policy[i]
        range_start = i
print(f"[{s_grid[range_start]:.3f}, {s_grid[-1]:.3f}]     "
      f"{ACTION_LABELS[ACTIONS[current_a]]}")

# Print value function endpoints
print(f"\n--- Value Function ---")
print(f"V(s=0.0)  = {V[0]:.3f}")
print(f"V(s=0.1)  = {np.interp(0.1, s_grid, V):.3f}")
print(f"V(s=0.3)  = {np.interp(0.3, s_grid, V):.3f}")
print(f"V(s=0.5)  = {np.interp(0.5, s_grid, V):.3f}")
print(f"V(s=0.7)  = {np.interp(0.7, s_grid, V):.3f}")
print(f"V(s=1.0)  = {V[-1]:.3f}")

# Simulate from different starting points
print("\n--- Simulated Paths ---")
for s0, label in [(0.05, "Newcomer (Apple TV+ type)"),
                  (0.30, "Mid-share (Disney+ type)"),
                  (0.65, "Incumbent (Netflix type)")]:
    s_path, a_path, pi_path = simulate(s0, 25, s_grid, policy)
    print(f"\n{label}: s_0 = {s0}")
    print(f"  s after 5y:  {s_path[5]:.3f}")
    print(f"  s after 25y: {s_path[-1]:.3f}")
    print(f"  Total discounted profit: "
          f"{sum(BETA**t * pi for t, pi in enumerate(pi_path)):.3f}")
    # Print policy transitions
    transitions = []
    cur = a_path[0]
    start = 0
    for t in range(1, len(a_path)):
        if a_path[t] != cur:
            transitions.append((start, t-1, cur))
            cur = a_path[t]
            start = t
    transitions.append((start, len(a_path)-1, cur))
    for s, e, a in transitions:
        print(f"  t=[{s},{e}]: {ACTION_LABELS[a]}")

# ========== Plot ==========

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1. Policy map
ax = axes[0, 0]
colors = [ACTION_COLORS[ACTIONS[p]] for p in policy]
ax.scatter(s_grid, [1]*len(s_grid), c=colors, s=80, marker='|', linewidths=8)
ax.set_xlabel('Market share s', fontsize=11)
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.set_title('Optimal Policy a*(s)', fontsize=12, fontweight='bold')
handles = [plt.Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=c, markersize=12, label=ACTION_LABELS[a])
           for a, c in ACTION_COLORS.items()]
ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, -0.3), ncol=2)

# 2. Value function
ax = axes[0, 1]
ax.plot(s_grid, V, 'b-', linewidth=2)
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('V(s)', fontsize=11)
ax.set_title('Value Function V(s)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Stage profit by cell
ax = axes[1, 0]
for a in ACTIONS:
    pi_vals = [profit(s, a) for s in s_grid]
    ax.plot(s_grid, pi_vals, label=ACTION_LABELS[a],
            color=ACTION_COLORS[a], linewidth=2)
ax.set_xlabel('Market share s', fontsize=11)
ax.set_ylabel('Per-period profit π(s, a)', fontsize=11)
ax.set_title('Stage Profit by Cell', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 4. Simulated market share paths
ax = axes[1, 1]
for s0, label, col in [(0.05, "s₀=0.05 (Newcomer)", '#E63946'),
                        (0.30, "s₀=0.30 (Mid-share)", '#F4A261'),
                        (0.65, "s₀=0.65 (Incumbent)", '#264653')]:
    s_path, _, _ = simulate(s0, 25, s_grid, policy)
    ax.plot(range(26), s_path, marker='o', markersize=4,
            label=label, color=col, linewidth=2)
ax.set_xlabel('Time t', fontsize=11)
ax.set_ylabel('Market share s_t', fontsize=11)
ax.set_title('Simulated Market Share Paths', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/home/claude/output/strategy_map.png', dpi=150, bbox_inches='tight')
print("\n✓ Figure saved to strategy_map.png")
