"""
Extension — competition with two-sided wavering customers (incumbent vs challenger).

Leader (Netflix) starts at s_N=0.7, challenger (Apple TV+) at s_C=0.  Each period a
fraction of EACH firm's base is "wavering" (contestable) and can be poached by the rival:
W_N = omega_N * s_N , W_C = omega_C * s_C , with the leader MORE footloose (omega_N > omega_C).
Both firms also acquire from the unserved pool U = 1 - s_N - s_C and play their single-firm
optimal action at their own share (so the leader switches retention->acquisition below 0.54).
The incumbent has a stickier base (brand/original content/habit): a retention bonus on rho_N.

    s_N' = (rho_N + b) s_N + alpha_N U - alpha_C * omega_N s_N + alpha_N * omega_C s_C
    s_C' =  rho_C    s_C + alpha_C U - alpha_N * omega_C s_C + alpha_C * omega_N s_N

Result: the leader is eroded by the challenger but stays ahead; the more footloose its
customers (omega_N), the smaller the surviving gap.  Run in the venv.  Writes fig_ext3.png.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import _P, value_iteration, ACTIONS, revenue

p = _P({})
BLUE, RED = "#003EA8", "#E63946"
g, V, pol, _it = value_iteration(p, n=401)
BONUS = 0.10          # incumbent retention advantage (brand / content stickiness)
WN, WC = 0.15, 0.075  # leader twice as footloose as the challenger


def act(s):
    return ACTIONS[int(pol[int(np.argmin(np.abs(g - s)))])]


def rho(s, a, bonus=0.0):
    P, C = a
    dP = p["delta_S"] if P == "S" else p["delta_T"]
    dC = p["delta_R"] if C == "R" else p["delta_A"]
    return min(p["rho_0"] + dP + dC + p["kappa_rho"] * s + bonus, p["rho_max"])


def alp(s, a):
    P, C = a
    tP = p["tilde_S"] if P == "S" else p["tilde_T"]
    tC = p["tilde_A"] if C == "A" else p["tilde_R"]
    return p["alpha_0"] + tP + tC + p["kappa_alpha"] * s


def sim(wN, wC, bonus=BONUS, sN=0.7, sC=0.0, T=200):
    TN, TC = [sN], [sC]
    cross = None
    for t in range(T):
        U = max(0.0, 1 - sN - sC)
        aN, aC = act(sN), act(sC)
        if sN < 0.54 and cross is None:
            cross = t
        rN, arN = rho(sN, aN, bonus), alp(sN, aN)
        rC, arC = rho(sC, aC), alp(sC, aC)
        nN = rN * sN + arN * U - arC * wN * sN + arN * wC * sC
        nC = rC * sC + arC * U - arN * wC * sC + arC * wN * sN
        nN, nC = max(0.0, nN), max(0.0, nC)
        if nN + nC > 1:
            k = 1 / (nN + nC); nN *= k; nC *= k
        sN, sC = nN, nC
        TN.append(sN); TC.append(sC)
    return np.array(TN), np.array(TC), cross


if __name__ == "__main__":
    tN, tC, cross = sim(WN, WC)
    print("omega_N=%.2f omega_C=%.3f, bonus=%.2f: leader 0.7 -> %.3f, challenger 0 -> %.3f, gap %.3f"
          % (WN, WC, BONUS, tN[-1], tC[-1], tN[-1] - tC[-1]))
    oms = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    fN, fC = [], []
    for om in oms:
        a, b, _ = sim(om, om / 2)
        fN.append(a[-1]); fC.append(b[-1])
        print("  omega_N=%.2f (omega_C=%.3f) -> leader*=%.3f challenger*=%.3f gap=%.3f"
              % (om, om / 2, a[-1], b[-1], a[-1] - b[-1]))

    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3))
    T = len(tN)
    ax[0].plot(range(T), tN, color=BLUE, lw=2.8, label="leader  s_N")
    ax[0].plot(range(T), tC, color=RED, lw=2.8, label="challenger  s_C")
    ax[0].fill_between(range(T), tC, tN, where=(tN >= tC), alpha=0.10, color=BLUE)
    ax[0].annotate("", (40, tN[-1]), (40, tC[-1]), arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax[0].text(44, (tN[-1] + tC[-1]) / 2 - 0.02, "gap ~0.24", fontsize=9)
    ax[0].set_xlim(0, 60); ax[0].set_ylim(0, 0.8)
    ax[0].set_xlabel("period t"); ax[0].set_ylabel("market share")
    ax[0].set_title("Leader eroded but stays ahead (omega_N=0.15)", fontweight="bold")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="upper right")
    ax[1].plot(oms, fN, color=BLUE, lw=2.8, marker="o", label="leader  s_N*")
    ax[1].plot(oms, fC, color=RED, lw=2.8, marker="o", label="challenger  s_C*")
    ax[1].axvline(0.15, color="gray", ls=":", lw=1.0)
    ax[1].set_xlabel("leader wavering fraction  omega_N  (omega_C = omega_N/2)")
    ax[1].set_ylabel("long-run share")
    ax[1].set_title("More wavering -> gap narrows (leader still ahead)", fontweight="bold")
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout(); plt.savefig("./fig_ext3.png", dpi=160, bbox_inches="tight"); plt.close()
    print("saved fig_ext3.png")
