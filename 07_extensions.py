"""
Extension — competition with two-sided wavering customers (incumbent vs entrant).

Leader (Netflix) starts at s_N=0.7, challenger (Apple TV+) at s_C=0.  Each period a
fraction of EACH firm's base is "wavering" (contestable) and can be poached by the rival:
W = omega * s.  The wavering fraction is the inverse of stickiness, so the established
incumbent has STICKY customers (low omega_N) and the new entrant FOOTLOOSE ones
(high omega_C): omega_N < omega_C.  Both also acquire from the unserved pool
U = 1 - s_N - s_C and play their single-firm optimal action at their own share (so the
leader defends with retention R and the small entrant attacks with acquisition A).

    s_N' = rho_N s_N + alpha_N U - alpha_C * omega_N s_N + alpha_N * omega_C s_C
    s_C' = rho_C s_C + alpha_C U - alpha_N * omega_C s_C + alpha_C * omega_N s_N

Note alpha_C*omega_N acts like an extra (competition-specific) churn on the leader, so the
firm's effective retention against the rival is rho - alpha_rival*omega.  Result: the
incumbent is eroded but stays ahead purely because its base is stickier (low omega) and it
poaches the entrant's footloose customers back; if the incumbent's base became as footloose
as the entrant's, it would lose the lead.  Run in the venv.  Writes fig_ext3.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import _P, value_iteration, ACTIONS, revenue

p = _P({})
BLUE, RED = "#003EA8", "#E63946"
g, V, pol, _it = value_iteration(p, n=401)
WN, WC = 0.05, 0.15   # incumbent sticky (low), entrant footloose (high)


def act(s):
    return ACTIONS[int(pol[int(np.argmin(np.abs(g - s)))])]


def rho(s, a):
    P, C = a
    dP = p["delta_S"] if P == "S" else p["delta_T"]
    dC = p["delta_R"] if C == "R" else p["delta_A"]
    return min(p["rho_0"] + dP + dC + p["kappa_rho"] * s, p["rho_max"])


def alp(s, a):
    P, C = a
    tP = p["tilde_S"] if P == "S" else p["tilde_T"]
    tC = p["tilde_A"] if C == "A" else p["tilde_R"]
    return p["alpha_0"] + tP + tC + p["kappa_alpha"] * s


def sim(wN, wC, sN=0.7, sC=0.0, T=200):
    TN, TC = [sN], [sC]
    for t in range(T):
        U = max(0.0, 1 - sN - sC)
        aN, aC = act(sN), act(sC)
        rN, arN = rho(sN, aN), alp(sN, aN)
        rC, arC = rho(sC, aC), alp(sC, aC)
        nN = rN * sN + arN * U - arC * wN * sN + arN * wC * sC
        nC = rC * sC + arC * U - arN * wC * sC + arC * wN * sN
        nN, nC = max(0.0, nN), max(0.0, nC)
        if nN + nC > 1:
            k = 1 / (nN + nC); nN *= k; nC *= k
        sN, sC = nN, nC
        TN.append(sN); TC.append(sC)
    return np.array(TN), np.array(TC)


if __name__ == "__main__":
    tN, tC = sim(WN, WC)
    print("omega_N=%.2f (sticky) omega_C=%.2f (footloose): leader 0.7 -> %.3f, challenger 0 -> %.3f, gap %.3f"
          % (WN, WC, tN[-1], tC[-1], tN[-1] - tC[-1]))
    # sweep the INCUMBENT's wavering (its stickiness), entrant fixed footloose
    oms = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    fN, fC = [], []
    for om in oms:
        a, b = sim(om, WC)
        fN.append(a[-1]); fC.append(b[-1])
        print("  omega_N=%.2f (omega_C=%.2f) -> leader*=%.3f challenger*=%.3f gap=%.3f"
              % (om, WC, a[-1], b[-1], a[-1] - b[-1]))

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3))
    T = len(tN)
    ax[0].plot(range(T), tN, color=BLUE, lw=2.8, label="leader  s_N (sticky)")
    ax[0].plot(range(T), tC, color=RED, lw=2.8, label="challenger  s_C (footloose)")
    ax[0].fill_between(range(T), tC, tN, where=(tN >= tC), alpha=0.10, color=BLUE)
    ax[0].annotate("", (45, tN[-1]), (45, tC[-1]), arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax[0].text(47, (tN[-1] + tC[-1]) / 2 - 0.02, "gap ~0.13", fontsize=9)
    ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 0.8)
    ax[0].set_xlabel("period t"); ax[0].set_ylabel("market share")
    ax[0].set_title("Sticky incumbent stays ahead", fontweight="bold")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="upper right")
    ax[1].plot(oms, fN, color=BLUE, lw=2.8, marker="o", label="leader  s_N*")
    ax[1].plot(oms, fC, color=RED, lw=2.8, marker="o", label="challenger  s_C*")
    ax[1].axvline(0.05, color="gray", ls=":", lw=1.0)
    ax[1].set_xlabel("incumbent wavering  omega_N  (entrant fixed omega_C = 0.15)")
    ax[1].set_ylabel("long-run share")
    ax[1].set_title("Stickiness is the moat (sweep omega_N)", fontweight="bold")
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("./fig_ext3.png", dpi=160, bbox_inches="tight"); plt.close()
    print("saved fig_ext3.png")
