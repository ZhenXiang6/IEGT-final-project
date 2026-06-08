"""
Experiment harness for the enhanced OTT dynamic strategy model.

Importable API (no side effects on import):
    from experiments import analyze, sweep1, grid2, multiplicity_scan, DEFAULTS, ACTIONS
    r = analyze(kappa_rho=0.12, kappa_alpha=0.30)          # one scenario -> dict of outcomes
    s = sweep1('beta', [0.90,0.95,0.99])                   # 1-D parameter sweep
    g = grid2('kappa_rho',[0,.1,.2], 'kappa_alpha',[0,.2,.4])

Model (same calibration as 06_enhanced_model.py):
  state s in [0,1]; action (P,C) in {T,S}x{A,R}
  R(s,S)=p_S ; R(s,T)=p_T*(phi0+phi1*s)
  C_A(s)=c_A*(1+lamA*s) ; C_R(s)=c_R*(1+lamR*s)
  rho(P,C,s)=min(rho0+dP+dC+kappa_rho*s, rho_max)
  alpha(P,C,s)=a0 + dP~ + dC~ + kappa_alpha*s            (Bass: inflow alpha*(1-s)=(p+q s)(1-s))
  s'=rho*s+alpha*(1-s) ; V(s)=max_a{pi+beta V(s')} ; value iteration
"""
import numpy as np

DEFAULTS = dict(
    beta=0.95,
    p_S=15.0, p_T=5.0, phi0=0.5, phi1=0.5,
    rho_0=0.65, delta_S=0.105, delta_T=0.0, delta_R=0.10, delta_A=0.0,
    kappa_rho=0.08, rho_max=0.98,
    alpha_0=0.10, tilde_S=0.0, tilde_T=0.05, tilde_A=0.15, tilde_R=0.0,
    kappa_alpha=0.20,
    c_A=1.2, c_R=0.8, lambda_A=1.0, lambda_R=0.3,
)

ACTIONS = [("T", "A"), ("T", "R"), ("S", "A"), ("S", "R")]
LABEL = {("T", "A"): "TA", ("T", "R"): "TR", ("S", "A"): "SA", ("S", "R"): "SR"}


def _P(p):  # merge overrides
    d = dict(DEFAULTS); d.update(p); return d


def revenue(s, P, p):
    return p["p_S"] if P == "S" else p["p_T"] * (p["phi0"] + p["phi1"] * s)

def cost(s, C, p):
    return p["c_A"] * (1 + p["lambda_A"] * s) if C == "A" else p["c_R"] * (1 + p["lambda_R"] * s)

def rho(s, P, C, p):
    dP = p["delta_S"] if P == "S" else p["delta_T"]
    dC = p["delta_R"] if C == "R" else p["delta_A"]
    return min(p["rho_0"] + dP + dC + p["kappa_rho"] * s, p["rho_max"])

def alpha(s, P, C, p):
    tP = p["tilde_S"] if P == "S" else p["tilde_T"]
    tC = p["tilde_A"] if C == "A" else p["tilde_R"]
    return p["alpha_0"] + tP + tC + p["kappa_alpha"] * s

def transition(s, a, p):
    P, C = a
    return float(np.clip(rho(s, P, C, p) * s + alpha(s, P, C, p) * (1 - s), 0.0, 1.0))

def profit(s, a, p):
    P, C = a
    return revenue(s, P, p) * s - cost(s, C, p)


def value_iteration(p, n=201, tol=1e-7, max_iter=3000):
    g = np.linspace(0, 1, n); V = np.zeros(n); it = 0
    for it in range(max_iter):
        Vn = np.empty(n); pol = np.empty(n, dtype=int)
        for i, s in enumerate(g):
            best, bi = -1e18, 0
            for ai, a in enumerate(ACTIONS):
                q = profit(s, a, p) + p["beta"] * np.interp(transition(s, a, p), g, V)
                if q > best:
                    best, bi = q, ai
            Vn[i], pol[i] = best, bi
        d = np.max(np.abs(Vn - V)); V = Vn
        if d < tol:
            break
    return g, V, pol, it + 1


def regimes(g, pol):
    out = []; cur = int(pol[0]); st = 0
    for i in range(1, len(g)):
        if pol[i] != cur:
            out.append([round(float(g[st]), 3), round(float(g[i - 1]), 3), LABEL[ACTIONS[cur]]])
            cur, st = int(pol[i]), i
    out.append([round(float(g[st]), 3), round(float(g[-1]), 3), LABEL[ACTIONS[cur]]])
    return out


def fixed_points(g, pol, p):
    f = np.array([transition(s, ACTIONS[int(pol[i])], p) for i, s in enumerate(g)])
    d = f - g; res = []
    for i in range(len(d) - 1):
        if d[i] == 0 and (i == 0 or d[i - 1] * d[i] > 0):
            pass
        if d[i] * d[i + 1] < 0:
            x = g[i] - d[i] * (g[i + 1] - g[i]) / (d[i + 1] - d[i])
            idx = int(np.argmin(np.abs(g - x))); i0, i1 = max(0, idx - 3), min(len(g) - 1, idx + 3)
            slope = (f[i1] - f[i0]) / (g[i1] - g[i0] + 1e-12)
            res.append([round(float(x), 4), "stable" if slope < 1 else "unstable"])
    # boundary attractors: s=1 absorbing if f(1)>=1 region pulls up; s=0 if f(0)<=0 (rare)
    return f, res


def _switch_points(g, pol, which):
    """which='P' -> boundaries where pricing flips T<->S ; 'C' -> A<->R. Returns list of s."""
    vals = ["S" if ACTIONS[int(k)][0] == "S" else "T" for k in pol] if which == "P" \
        else ["R" if ACTIONS[int(k)][1] == "R" else "A" for k in pol]
    sw = []
    for i in range(1, len(g)):
        if vals[i] != vals[i - 1]:
            sw.append([round(float(g[i]), 3), f"{vals[i-1]}->{vals[i]}"])
    return sw


def simulate(s0, T, g, pol, p):
    s = s0; disc = 0.0
    for t in range(T):
        a = ACTIONS[int(pol[int(np.argmin(np.abs(g - s)))])]
        disc += p["beta"] ** t * profit(s, a, p)
        s = transition(s, a, p)
    return float(s), float(disc)


def analyze(n=201, T=40, **over):
    p = _P(over)
    g, V, pol, iters = value_iteration(p, n=n)
    f, fps = fixed_points(g, pol, p)
    reg = regimes(g, pol)
    cells = []
    for _, _, c in reg:
        if c not in cells:
            cells.append(c)
    sinf = {}; disc = {}
    for s0 in (0.05, 0.30, 0.65):
        si, dd = simulate(s0, 25, g, pol, p)
        sinf[str(s0)] = round(si, 4); disc[str(s0)] = round(dd, 3)
    inc = (disc["0.65"] - disc["0.05"]) / abs(disc["0.05"]) if disc["0.05"] != 0 else float("nan")
    # thresholds
    Psw = _switch_points(g, pol, "P"); Csw = _switch_points(g, pol, "C")
    stable = [x for x in fps if x[1] == "stable"]
    return dict(
        params={k: over[k] for k in over},
        regimes=reg, cells=cells,
        P_switches=Psw, C_switches=Csw,
        fixed_points=fps, n_stable=len(stable),
        stable_states=[x[0] for x in stable],
        s_inf=sinf, disc_profit=disc, incumbent_adv=round(inc, 4),
        V={"0.0": round(float(V[0]), 2), "0.5": round(float(np.interp(0.5, g, V)), 2),
           "1.0": round(float(V[-1]), 2)},
        iters=iters,
    )


def sweep1(param, values, n=151, **base):
    rows = []
    for v in values:
        r = analyze(n=n, **{**base, param: v})
        rows.append(dict(val=round(float(v), 4),
                         cells=r["cells"], thr=[x[0] for x in r["C_switches"]] + ["|"] + [x[0] for x in r["P_switches"]],
                         n_stable=r["n_stable"], stable=r["stable_states"],
                         sinf_lo=r["s_inf"]["0.05"], sinf_hi=r["s_inf"]["0.65"],
                         inc_adv=r["incumbent_adv"]))
    return rows


def grid2(pa, va, pb, vb, n=121, **base):
    out = []
    for a in va:
        row = []
        for b in vb:
            r = analyze(n=n, **{**base, pa: a, pb: b})
            row.append(dict(a=round(float(a), 3), b=round(float(b), 3),
                            cells="".join(c[0] for c in r["cells"]),
                            n_stable=r["n_stable"], sinf_hi=r["s_inf"]["0.65"], inc=r["incumbent_adv"]))
        out.append(row)
    return out


def multiplicity_scan(param, values, n=401, **base):
    """Report number of stable fixed points as `param` varies (path-dependence hunt)."""
    rows = []
    for v in values:
        r = analyze(n=n, **{**base, param: v})
        rows.append(dict(val=round(float(v), 4), n_stable=r["n_stable"],
                         fps=r["fixed_points"]))
    return rows


if __name__ == "__main__":
    import json
    print("=== BASELINE ===")
    b = analyze(n=401)
    print(json.dumps(b, ensure_ascii=False, indent=1))
    print("\n=== sweep beta ===")
    for r in sweep1("beta", [0.85, 0.90, 0.93, 0.95, 0.97, 0.99]):
        print(r)
    print("\n=== sweep kappa_rho (multiplicity) ===")
    for r in multiplicity_scan("kappa_rho", np.round(np.arange(0, 0.31, 0.03), 2)):
        print(r["val"], "n_stable=", r["n_stable"], r["fps"])
