"""
OTT Streaming Platform — 延伸四: Stackelberg Duopoly (MPE)
Leader (large incumbent, e.g. Netflix) moves first;
Follower (challenger, e.g. Apple TV+) observes and best-responds.

State: (s_L, s_C) with s_L + s_C <= 1.  Non-user pool = 1 - s_L - s_C.

Transition (symmetric poaching):
  s_L' = rho_L*s_L + alpha_L*pool - poach_by_C + poach_by_L
  s_C' = rho_C*s_C + alpha_C*pool + poach_by_C - poach_by_L
  poach_by_C = GAMMA_POACH * s_L  if Follower uses Acquisition
  poach_by_L = GAMMA_POACH * s_C  if Leader    uses Acquisition

Solving method: coupled Bellman value iteration with Stackelberg operator.
  For each (s_L, s_C):
    1. Follower best-responds to each a_L: a_C*(a_L) = argmax Q_C
    2. Leader picks a_L knowing Follower's best response
  Damped V updates (DAMPING=0.6) stabilize the non-contraction operator.

Key finding: symmetric equilibrium at ~(0.402, 0.402);
  competition expands total market but each firm gets less than monopoly.
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

GAMMA_POACH = 0.05   # fraction of opponent's base poached when using Acquisition
N_GRID      = 41     # grid size per dimension (step = 0.025)
DAMPING     = 0.6    # value update damping: V = d*V_new + (1-d)*V_old
TOL         = 1e-4   # convergence tolerance (policy-stable even if V not fully converged)
MAX_ITER    = 500

ACTIONS = [('T','A'), ('T','R'), ('S','A'), ('S','R')]
ACTION_LABELS = {
    ('T','A'): 'Trans-Acq', ('T','R'): 'Trans-Ret',
    ('S','A'): 'Sub-Acq',   ('S','R'): 'Sub-Ret',
}
ACTION_COLORS = {
    ('T','A'): '#E63946', ('T','R'): '#F4A261',
    ('S','A'): '#2A9D8F', ('S','R'): '#264653',
}

# ========== Primitives ==========
def revenue(s, P):
    return p_S if P == 'S' else p_T * (phi0 + phi1 * s)

def rho_base(P, C):
    return rho_0 + (delta_S if P == 'S' else delta_T) + (delta_R if C == 'R' else 0.0)

def alpha_base(P, C):
    return alpha_0 + (tilde_S if P == 'S' else tilde_T) + (tilde_A if C == 'A' else 0.0)

def stage_profit(s, action):
    P, C = action
    return revenue(s, P) * s - (c_A if C == 'A' else c_R)

# ========== Grid ==========
s_arr = np.linspace(0, 1, N_GRID)
ds    = s_arr[1] - s_arr[0]

V_L = np.zeros((N_GRID, N_GRID))
V_C = np.zeros((N_GRID, N_GRID))
policy_L = np.zeros((N_GRID, N_GRID), dtype=int)
policy_C = np.zeros((N_GRID, N_GRID), dtype=int)


def interp2d(V, sL, sC):
    """Bilinear interpolation; clips to feasible region."""
    sL = float(np.clip(sL, 0.0, 1.0))
    sC = float(np.clip(sC, 0.0, max(0.0, 1.0 - sL)))
    iL_f = sL / ds
    iC_f = sC / ds
    iL0 = min(int(iL_f), N_GRID - 2)
    iC0 = min(int(iC_f), N_GRID - 2)
    iL1 = iL0 + 1
    iC1 = iC0 + 1
    wL = iL_f - iL0
    wC = iC_f - iC0
    return (V[iL0, iC0] * (1 - wL) * (1 - wC) +
            V[iL1, iC0] *      wL  * (1 - wC) +
            V[iL0, iC1] * (1 - wL) *      wC  +
            V[iL1, iC1] *      wL  *      wC)


def do_transition(sL, sC, a_L, a_C):
    """Next-period shares under Stackelberg actions."""
    pool   = max(1.0 - sL - sC, 0.0)
    rho_L  = rho_base(*a_L);  alpha_L = alpha_base(*a_L)
    rho_C  = rho_base(*a_C);  alpha_C = alpha_base(*a_C)

    # Each firm acquires from the non-user pool proportionally
    new_L  = alpha_L * pool
    new_C  = alpha_C * pool

    # Symmetric poaching: Acquisition implies raiding opponent's base
    poach_by_C = GAMMA_POACH * sL if a_C[1] == 'A' else 0.0
    poach_by_L = GAMMA_POACH * sC if a_L[1] == 'A' else 0.0

    sL_new = rho_L * sL + new_L - poach_by_C + poach_by_L
    sC_new = rho_C * sC + new_C + poach_by_C - poach_by_L

    sL_new = float(np.clip(sL_new, 0.0, 1.0))
    sC_new = float(np.clip(sC_new, 0.0, 1.0 - sL_new))
    return sL_new, sC_new


# ========== Stackelberg Value Iteration ==========
print("=" * 65)
print("延伸四: Stackelberg Duopoly (Markov Perfect Equilibrium)")
print(f"  Grid {N_GRID}x{N_GRID}, GAMMA_POACH={GAMMA_POACH}, DAMPING={DAMPING}")
print("=" * 65)

prev_polL = policy_L.copy()
prev_polC = policy_C.copy()

for it in range(MAX_ITER):
    V_L_new  = np.full((N_GRID, N_GRID), -1e9)
    V_C_new  = np.full((N_GRID, N_GRID), -1e9)
    pol_L_new = np.zeros((N_GRID, N_GRID), dtype=int)
    pol_C_new = np.zeros((N_GRID, N_GRID), dtype=int)

    for iL in range(N_GRID):
        for iC in range(N_GRID):
            sL = s_arr[iL]
            sC = s_arr[iC]
            if sL + sC > 1.0 + 1e-9:
                continue  # infeasible state

            best_qL = -np.inf
            best_aL_idx = 0
            best_aC_idx = 0

            for iaL, a_L in enumerate(ACTIONS):
                # --- Follower's best response given a_L ---
                best_qC = -np.inf
                br_iaC  = 0
                for iaC, a_C in enumerate(ACTIONS):
                    sLn, sCn = do_transition(sL, sC, a_L, a_C)
                    vC_next  = interp2d(V_C, sLn, sCn)
                    qC = stage_profit(sC, a_C) + BETA * vC_next
                    if qC > best_qC:
                        best_qC = qC
                        br_iaC  = iaC

                # --- Leader's Q given Follower's best response ---
                a_C_br        = ACTIONS[br_iaC]
                sLn, sCn      = do_transition(sL, sC, a_L, a_C_br)
                vL_next       = interp2d(V_L, sLn, sCn)
                qL = stage_profit(sL, a_L) + BETA * vL_next

                if qL > best_qL:
                    best_qL    = qL
                    best_aL_idx = iaL
                    best_aC_idx = br_iaC

            # Value for Follower at equilibrium action pair
            a_L_eq = ACTIONS[best_aL_idx]
            a_C_eq = ACTIONS[best_aC_idx]
            sLn, sCn   = do_transition(sL, sC, a_L_eq, a_C_eq)
            vC_at_eq   = interp2d(V_C, sLn, sCn)
            qC_eq      = stage_profit(sC, a_C_eq) + BETA * vC_at_eq

            V_L_new[iL, iC]  = best_qL
            V_C_new[iL, iC]  = qC_eq
            pol_L_new[iL, iC] = best_aL_idx
            pol_C_new[iL, iC] = best_aC_idx

    diff = max(np.max(np.abs(V_L_new[V_L_new > -1e8] - V_L[V_L_new > -1e8])),
               np.max(np.abs(V_C_new[V_C_new > -1e8] - V_C[V_C_new > -1e8])))

    V_L = DAMPING * V_L_new + (1 - DAMPING) * V_L
    V_C = DAMPING * V_C_new + (1 - DAMPING) * V_C
    policy_L = pol_L_new
    policy_C = pol_C_new

    pol_changed = (np.sum(policy_L != prev_polL) + np.sum(policy_C != prev_polC))
    prev_polL = policy_L.copy()
    prev_polC = policy_C.copy()

    if (it + 1) % 50 == 0 or diff < TOL:
        print(f"  iter {it+1:3d}: diff={diff:.4f}, policy_changes={pol_changed}")

    if diff < TOL:
        print(f"  Converged (diff < {TOL}) after {it+1} iterations.")
        break
else:
    print(f"  Max iterations reached (diff={diff:.4f}). Policy may still be stable.")


# ========== Simulation ==========
def simulate(sL0, sC0, T=60):
    sL, sC = sL0, sC0
    path = [(sL, sC)]
    for _ in range(T):
        iL = np.argmin(np.abs(s_arr - sL))
        iC = np.argmin(np.abs(s_arr - sC))
        # Clip to feasible
        if sL + sC > 1.0:
            sC = 1.0 - sL
        iL = np.clip(iL, 0, N_GRID - 1)
        iC = np.clip(iC, 0, N_GRID - 1)
        a_L = ACTIONS[policy_L[iL, iC]]
        a_C = ACTIONS[policy_C[iL, iC]]
        sL, sC = do_transition(sL, sC, a_L, a_C)
        path.append((sL, sC))
    return np.array(path)

print("\n=== Equilibrium Simulation ===")
starts = [(0.05, 0.05), (0.50, 0.10), (0.10, 0.50), (0.30, 0.30)]
labels = ['Both small', 'L large/C small', 'L small/C large', 'Both mid']
paths  = [simulate(*s0) for s0 in starts]

for (s0, lbl, path) in zip(starts, labels, paths):
    sL_eq, sC_eq = path[-1]
    total_eq = sL_eq + sC_eq
    iL = np.argmin(np.abs(s_arr - path[-1, 0]))
    iC = np.argmin(np.abs(s_arr - path[-1, 1]))
    if iL < N_GRID and iC < N_GRID and path[-1,0]+path[-1,1] <= 1.01:
        aL_eq = ACTION_LABELS[ACTIONS[policy_L[iL, iC]]]
        aC_eq = ACTION_LABELS[ACTIONS[policy_C[iL, iC]]]
    else:
        aL_eq = aC_eq = 'n/a'
    print(f"  {lbl:22s} s0=({s0[0]:.2f},{s0[1]:.2f}) -> "
          f"s_eq=({sL_eq:.3f},{sC_eq:.3f}), total={total_eq:.3f}  "
          f"L:{aL_eq} / C:{aC_eq}")

# ========== Policy snapshot at key states ==========
print("\n=== Policy at selected states ===")
print(f"  {'(s_L, s_C)':<18} {'Leader':<14} {'Follower'}")
print("  " + "-" * 48)
check_states = [(0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.4, 0.4),
                (0.6, 0.1), (0.1, 0.6), (0.5, 0.3), (0.3, 0.5)]
for (sl, sc) in check_states:
    if sl + sc > 1.0:
        continue
    iL = np.argmin(np.abs(s_arr - sl))
    iC = np.argmin(np.abs(s_arr - sc))
    aL = ACTION_LABELS[ACTIONS[policy_L[iL, iC]]]
    aC = ACTION_LABELS[ACTIONS[policy_C[iL, iC]]]
    print(f"  ({sl:.1f}, {sc:.1f})           {aL:<14} {aC}")


# ========== Plot ==========
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
BLUE = '#2B4CC9'; RED = '#E63946'; TEAL = '#2A9D8F'; DARK = '#264653'; ORANGE = '#F4A261'

color_idx = {0: RED, 1: ORANGE, 2: TEAL, 3: DARK}

# --- Panel 1: Leader's policy heatmap ---
ax = axes[0]
img_L = np.full((N_GRID, N_GRID, 4), np.nan)
for iL in range(N_GRID):
    for iC in range(N_GRID):
        if s_arr[iL] + s_arr[iC] <= 1.0 + 1e-9:
            c = matplotlib.colors.to_rgba(color_idx[policy_L[iL, iC]], alpha=0.85)
            img_L[iC, iL] = c  # transpose: x=s_L, y=s_C

ax.imshow(img_L, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
ax.plot([0, 1], [1, 0], 'k--', linewidth=1.2, alpha=0.5)
ax.set_xlabel('Leader share $s_L$', fontsize=11)
ax.set_ylabel('Challenger share $s_C$', fontsize=11)
ax.set_title("Leader's Optimal Policy", fontsize=12, fontweight='bold')
from matplotlib.patches import Patch
patches = [Patch(color=ACTION_COLORS[a], label=ACTION_LABELS[a]) for a in ACTIONS]
ax.legend(handles=patches, fontsize=8, loc='upper right')

# --- Panel 2: Follower's policy heatmap ---
ax = axes[1]
img_C = np.full((N_GRID, N_GRID, 4), np.nan)
for iL in range(N_GRID):
    for iC in range(N_GRID):
        if s_arr[iL] + s_arr[iC] <= 1.0 + 1e-9:
            c = matplotlib.colors.to_rgba(color_idx[policy_C[iL, iC]], alpha=0.85)
            img_C[iC, iL] = c

ax.imshow(img_C, origin='lower', extent=[0, 1, 0, 1], aspect='auto')
ax.plot([0, 1], [1, 0], 'k--', linewidth=1.2, alpha=0.5)
ax.set_xlabel('Leader share $s_L$', fontsize=11)
ax.set_ylabel('Challenger share $s_C$', fontsize=11)
ax.set_title("Follower's Best-Response Policy", fontsize=12, fontweight='bold')
ax.legend(handles=patches, fontsize=8, loc='upper right')

# --- Panel 3: Simulation paths ---
ax = axes[2]
path_colors = [BLUE, RED, TEAL, DARK]
for path, lbl, col in zip(paths, labels, path_colors):
    ax.plot(path[:, 0], path[:, 1], color=col, linewidth=2, label=lbl)
    ax.scatter(path[0, 0], path[0, 1], color=col, s=60, marker='o', zorder=5)
    ax.scatter(path[-1, 0], path[-1, 1], color=col, s=100, marker='*', zorder=6)

ax.plot([0, 1], [1, 0], 'k--', linewidth=1.2, alpha=0.4)
ax.plot([0, 1], [0, 0], 'gray', linewidth=0.8, alpha=0.3)
ax.plot([0, 0], [0, 1], 'gray', linewidth=0.8, alpha=0.3)
ax.set_xlabel('Leader share $s_L$', fontsize=11)
ax.set_ylabel('Challenger share $s_C$', fontsize=11)
ax.set_title('Equilibrium Paths\n(circles=start, stars=end)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)

plt.suptitle(
    '延伸四: Stackelberg Duopoly (Markov Perfect Equilibrium)\n'
    'Leader moves first; Follower best-responds; symmetric poaching ($\\gamma_{poach}=0.05$)',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out_path = '/Users/chenyiting/NTU/IEGT/IEGT-final-project/stackelberg_game.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {out_path}")
plt.show()
