# OTT Streaming Platform: Two-Dimensional Dynamic Strategic Model

課程研究報告 — Information Economics and Game Theory, 2026

---

## Overview

This project builds and solves a **single-firm, infinite-horizon, discounted dynamic optimization model** for OTT streaming platforms (e.g., Netflix, Disney+, HBO Max). The firm jointly chooses a **pricing model** (SVOD vs. TVOD) and a **customer strategy** (acquisition vs. retention) each period, given its current market share `s`.

The model is described by a Bellman equation and solved numerically via **value iteration**.

---

## Model Summary

| Element | Specification |
|---|---|
| State variable | Market share `s ∈ [0, 1]` |
| Action space | `(P, C) ∈ {T, S} × {A, R}` — 4 discrete cells |
| Transition | `s' = ρ(P,C) · s + α(P,C) · (1 − s)` |
| Stage profit | `π(s, a) = R(s, P) · s − C(C)` |
| Bellman equation | `V(s) = max_a { π(s,a) + β V(s'(s,a)) }` |
| Discount factor | `β = 0.95` |

**Revenue functions**

- SVOD: `R(s, S) = p_S = 15` (flat per-subscriber)
- TVOD: `R(s, T) = p_T (φ₀ + φ₁ s)` (scale-sensitive purchase frequency)

**Retention rate** `ρ(P, C) = ρ₀ + δ_P + δ_C`

- Baseline `ρ₀ = 0.65`; SVOD boost `δ_S = 0.105` (Iyengar et al. 2011); Retention strategy boost `δ_R = 0.10`

**Acquisition rate** `α(P, C) = α₀ + δ̃_P + δ̃_C` (does not depend on `s`; saturation enters only via `(1−s)` in the transition)

---

## File Structure

```
.
├── 03_solve_model.py            # Baseline model (value iteration)
├── 04_extension1.py             # Extension 1: network externality & multiple steady states
├── 05_competitor_fixed.py       # Competitor extension: fixed Netflix market share
├── 08_continuous_effort.py      # Extension 2: continuous effort allocation e ∈ [0,1]
├── 07_stackelberg_game.py       # Extension 4: Stackelberg duopoly (Markov Perfect Equilibrium)
├── 06_entry_deterrence.py       # Extension 3: entry deterrence threshold V(0) = 0
├── OTT_strategy_slides.tex      # Beamer presentation (XeLaTeX, Traditional Chinese)
├── OTT_strategy_slides.pdf      # Compiled slide deck (29 pages, royal-blue + grid theme)
├── extension1_multi_steady.png  # Figure: bifurcation / phase / path-dependence
├── competitor_fixed.png         # Figure: challenger value & policy under fixed s_N
├── continuous_effort.png        # Figure: e*(s) curves, V comparison, value gain
├── stackelberg_game.png         # Figure: strategy maps (L & C), phase portrait, time series
├── entry_deterrence.png         # Figure: V(0) vs s_N, steady-state compression, NPV decomposition
└── README.md
```

### Running the code

```bash
# Baseline model
python 03_solve_model.py

# Extension 1: multiple steady states
python 04_extension1.py

# Competitor extension
python 05_competitor_fixed.py

# Extension 2: continuous effort allocation
python 08_continuous_effort.py

# Extension 4: Stackelberg duopoly
python 07_stackelberg_game.py

# Extension 3: entry deterrence threshold
python 06_entry_deterrence.py
```

Dependencies: `numpy`, `matplotlib`

> Note: the `savefig`/`out_path` in each script is hardcoded to an absolute path from the
> original author's machine. Change it to a local/relative path (e.g. `./strategy_map.png`)
> before running, or the script will error.

### Enhanced model + verified-literature deck

`06_enhanced_model.py` is the **enhanced** model behind the current deck. On top of the report
baseline it adds: a retention network externality `ρ=ρ₀+δ+κ_ρ·s` (Katz–Shapiro 1985; Klemperer
1987), an acquisition-side **network externality** `α=α₀+δ̃+κ_α·s` (the deck frames both `κ_ρ` and
`κ_α` as network externalities; mathematically `κ_α` is the Bass imitation coefficient, inflow
`α(s)(1−s)`), and **market-scale costs** `C_A(s)=c_A(1+λ_A s)`,
`C_R(s)=c_R(1+λ_R s)` with `λ_A>λ_R` (Min et al. 2016). Value iteration on a 401-point grid gives
a 3-regime policy, steady state `s*≈0.744` (vs 0.505 without externalities), ~23% incumbent
advantage, and a `κ_ρ`-bifurcation toward market concentration. Run it in a venv (the default
`python3` has no numpy):

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python 06_enhanced_model.py   # writes fig_value_paths/phase/bifurcation/buildingblocks/cost.png
```

The literature behind every assumption was surveyed and **adversarially verified online** (a
multi-agent workflow); the 40 verified citations (DOIs + evidence URLs) are in `RESEARCH_dossier.md`.

`experiments.py` is an importable sweep harness over the enhanced calibration
(`analyze / sweep1 / grid2 / multiplicity_scan`, no import side effects). A systematic scan of every
exogenous parameter is written up in **`EXPERIMENTS_REPORT.md`** (Traditional Chinese, 9 findings) —
e.g. the three-regime structure is invariant across the whole tested range, the two switching
thresholds decouple (pricing ← price ratio; acquisition/retention ← cost scale), incumbent advantage
*shrinks* with patience, scale-dependent acquisition cost acts as an entry barrier, and strong
retention externalities produce a tipping jump and genuine path dependence (two stable steady states).

`OTT_enhanced_slides.tex` is the **current** Beamer deck (clean white theme, no decorations, royal
blue `#003EA8`, 22 pages, 6 sections; each cited paper carries a one-line explanation). Build with
`xelatex` twice. (`OTT_strategy_slides.tex` is the earlier report-based deck.)

### Building the presentation

The slide deck (`OTT_strategy_slides.tex`) summarizes the full report in Beamer. It is written
in Traditional Chinese and **must be compiled with XeLaTeX** (CJK via `ctexbeamer` + the
`PingFang TC` system font on macOS). Run twice so the table of contents, grid background, and
frame counter resolve:

```bash
xelatex OTT_strategy_slides.tex
xelatex OTT_strategy_slides.tex
```

Output: `OTT_strategy_slides.pdf` (29 pages, royal-blue + grid-paper theme). It follows a
6-section flow — 研究背景與動機 → 文獻回顧 → 模型建構 → 求解結果與解讀 → 延伸模型與結果
(all four extensions, one slide each) → 結論. The deck embeds `extension1_multi_steady.png`;
the strategy map, value function, and 2×2 grids are drawn inline with TikZ.

---

## Results

### 1. Baseline Model — Three-Regime Optimal Strategy

The value iteration converges in ~350 iterations. The optimal policy `a*(s)` exhibits a clean **3-regime structure**:

| Market share range | Optimal strategy | Economic interpretation |
|---|---|---|
| `s ∈ [0, 0.08]` | **Trans-Acq** | Early-stage: SVOD stage profit is negative; TVOD lowers commitment barrier for new users |
| `s ∈ [0.08, 0.54]` | **Sub-Acq** | Growth phase: SVOD dominates on revenue; acquisition dynamic margin is strong |
| `s ∈ [0.54, 1.00]` | **Sub-Ret** | Mature phase: retention dynamic margin dominates; flat-rate bias fully realized |

The Trans-Ret cell `(T, R)` is **never selected** under baseline calibration — TVOD's lower per-user revenue makes retention investment uneconomical at any market share.

**Incumbent advantage.** All three firm types converge to the same steady state `s∞ ≈ 0.505`, but their 25-year discounted profits differ by ~22% due to the **convexity of V(s)**:

| Firm type | s₀ | s₂₅ | Discounted profit |
|---|---|---|---|
| Newcomer (Apple TV+) | 0.05 | 0.505 | 79.56 |
| Mid-share (Disney+) | 0.30 | 0.505 | 86.23 |
| Incumbent (Netflix) | 0.65 | 0.505 | 97.41 |

Early high market share generates compounding discounted cash flows that latecomers cannot recover — explaining why Disney+ and Apple TV+ absorb billions in early-stage losses to enter the market.

**Sub-Ret is an incumbent-only strategy.** Only the Sub-Acq steady state (s ≈ 0.505) is self-consistent with the optimal policy map. Newcomers starting from low `s₀` never naturally reach the Sub-Ret zone; that regime is only visited by incumbents experiencing market share decline.

---

### 2. Extension 1 — Network Externality & Multiple Steady States

We introduce a **retention network externality**: `ρ(P, C, s) = ρ₀ + δ_P + δ_C + κ · s`. As `κ` increases, the transition function `f(s) = s'` becomes non-linear and can intersect the diagonal more than once.

| κ | Fixed points | Structure |
|---|---|---|
| 0.00–0.14 | 1 (~0.385–0.423) | Unique stable equilibrium |
| **0.16** | **3 [0.434, 0.899, 1.000]** | **Saddle-node bifurcation begins** |
| **0.18** | **3 [0.466, 0.766, 1.000]** | **Three-equilibrium zone** |
| 0.20+ | 1 (1.000) | Full monopoly as only attractor |

A **saddle-node bifurcation** occurs near `κ ≈ 0.15`. The three fixed points at `κ = 0.18`:

| Fixed point | Stability | Interpretation |
|---|---|---|
| s* = 0.466 | Stable | Low equilibrium: market share trap |
| s* = 0.766 | Unstable | Critical mass threshold |
| s* = 1.000 | Stable | Winner-takes-all monopoly |

Note: in our calibration, the SVOD baseline retention is already high (`ρ(S,R) = 0.855`), so `ρ` hits its upper bound of 1 at moderate `s` once `κ ≥ 0.20`. The high steady state is therefore `s = 1` (full market dominance) rather than an interior value. This makes the winner-takes-all interpretation stronger: platforms that cross the critical mass threshold are not merely dominant — they converge to monopoly.

**Path dependence.** At `κ = 0.18`, the long-run outcome depends entirely on the starting position relative to the unstable fixed point:

- `s₀ < 0.766` → converge to low equilibrium (s ≈ 0.47), e.g., Disney+ / Apple TV+
- `s₀ > 0.766` → converge to full monopoly (s → 1), e.g., Netflix in early-mover markets

This formalizes the **critical mass** phenomenon (Rohlfs 1974; Katz & Shapiro 1985): when retention network externalities are sufficiently strong, initial market position permanently determines long-run fate. Small platforms cannot escape the low-equilibrium trap by effort alone — they require a discrete jump past the critical threshold.

Output: `extension1_multi_steady.png` (bifurcation diagram, phase diagram at `κ = 0.18`, simulated paths from 21 starting points)

---

### 3. Competitor Extension — Fixed Netflix Market Share

We extend the baseline to a **partial competition setting**: Netflix occupies a fixed market share `s_N` (exogenous), leaving a reduced non-user pool `(1 − s_N − s_C)` for the challenger to capture.

**Transition (challenger):**
```
s_{C,t+1} = ρ(P,C) · s_C + α(P,C) · (1 − s_N − s_C)
```

The challenger's state range is `s_C ∈ [0, 1 − s_N]`; the Bellman structure is otherwise identical to the baseline.

#### Results

| s_N | Available market | Trans→Sub threshold | Sub-Acq→Sub-Ret threshold |
|---|---|---|---|
| 0.00 | 1.00 | 0.085 | 0.545 |
| 0.20 | 0.80 | 0.068 | 0.424 |
| 0.35 | 0.65 | 0.055 | 0.335 |
| 0.50 | 0.50 | 0.043 | 0.250 |

**Finding 1 — Proportional threshold compression.** Both strategy thresholds scale approximately in proportion to the available market `(1 − s_N)`:

- Trans→Sub threshold ≈ `0.085 × (1 − s_N)`
- Sub-Acq→Sub-Ret threshold ≈ `0.54 × (1 − s_N)`

The **3-regime structure is preserved** regardless of Netflix's dominance — the entire strategy map simply compresses leftward.

**Finding 2 — Relative position effect.** A challenger with the same absolute market share (e.g., `s_C = 0.30`) finds itself in different strategic regimes depending on `s_N`:

- `s_N = 0.00`: s_C = 0.30 → **Sub-Acq** (still in growth phase)
- `s_N = 0.50`: s_C = 0.30 > s* = 0.25 → **Sub-Ret** (treated as mature incumbent)

Netflix's market dominance forces challengers into retention-focused strategies earlier than their absolute size would otherwise dictate. This maps to the real-world observation that Disney+ pivoted toward retention (password-sharing crackdowns, bundle strategies) sooner in markets where Netflix was already dominant.

**Finding 3 — Ceiling effect on incumbent advantage.** As `s_N` grows, the maximum achievable `s_C` shrinks, compressing the range over which incumbent advantage accumulates. In highly saturated markets (`s_N = 0.50`), the challenger's value function ceiling is roughly half that of an uncontested market.

Output: `competitor_fixed.png` (value function + policy strip for each `s_N` scenario)

---

### 4. Extension 2 — Continuous Effort Allocation

The baseline forces a binary choice C ∈ {A, R}. This extension replaces it with a continuous retention effort `e ∈ [0, 1]`, where `(1 − e)` is acquisition effort:

```
ρ(P, e) = ρ_base(P) + δ_R · e^γ
α(P, e) = α_base(P) + δ̃_A · (1 − e)^γ
cost(e)  = (1 − e) · c_A + e · c_R
```

`γ = 1` (linear) → objective linear in `e` → optimal always corner → degenerates to binary. `γ < 1` (concave) → diminishing marginal returns → interior solution → firm mixes A and R simultaneously.

**Findings (γ = 0.5):** `e*(s)` smoothly increasing (50/50 at s ≈ 0.535); `V_cont ≥ V_disc` everywhere (avg +12.87, max +14.0 at s ≈ 0.585); γ sweep confirms 0% interior solutions at γ=1 vs ~92% at γ=0.25.

Output: `continuous_effort.png`

---

### 5. Extension 4 — Stackelberg Duopoly (Markov Perfect Equilibrium)

The baseline is a single-firm model. This extension introduces a second firm and game-theoretic interaction: a **Stackelberg duopoly** where the Leader (L, Netflix-like incumbent) moves first each period, and the Follower (C, Challenger) observes L's action and best responds.

**State:** `(s_L, s_C) ∈ {s_L + s_C ≤ 1}` — a 2D state space on a 41×41 grid.

**Transition:**
```
available = 1 - s_L - s_C
s_L' = ρ_L(a_L)·s_L + [α_L/(α_L+α_C)]·available − γ·[a_C=Acq]·s_L
s_C' = ρ_C(a_C)·s_C + [α_C/(α_L+α_C)]·available + γ·[a_C=Acq]·s_L
```
The `γ·s_L` term (γ = 0.05) captures poaching: C steals users from L when playing Acq.

**Solution concept:** Markov Perfect Equilibrium via nested Bellman iteration:
- For each `a_L`, compute C's best response `a_C*(a_L)`
- L picks `a_L*` anticipating C's response
- Both value functions iterated until convergence (~200–500 iterations)

**Three key questions:**

**Q1 — Accommodate vs. Deter:** Does L stay in Sub-Ret (defend profit, yield acquisition space) or switch to Sub-Acq (crowd out C) as C grows?

**Q2 — Challenger differentiation:** Does C retreat to Trans-Acq (low-commitment TVOD niche) to avoid direct SVOD competition when L is dominant?

**Q3 — Co-existence equilibrium:** Do paths converge to a stable duopoly `(s_L*, s_C*)` or does one firm dominate?

Run `python 07_stackelberg_game.py` for exact results.

Output: `stackelberg_game.png` (L's strategy map, C's strategy map, phase portrait, time series, value function slice)

---

### 5. Extension 3 — Entry Deterrence Threshold

Extension 05 asks *how* a challenger should compete given Netflix's fixed market share. This extension asks the prior question: **should the challenger enter at all?**

We define the entry deterrence condition as `V(0; s_N) < 0`: even under the globally optimal policy, the challenger's discounted NPV from zero market share is negative. We sweep `s_N ∈ [0, 0.95]` and locate `s_N*` where `V(0)` crosses zero.

**Structure of V(0):**

At `s_C = 0`, stage profit is always negative (`π = revenue × 0 − cost = −c_A`). Entry is only rational if the discounted future profits from growing market share outweigh the initial burn. As `s_N` rises, three forces simultaneously compress V(0):

1. **Ceiling compression**: steady-state `s*` shrinks proportionally, reducing long-run revenue
2. **Path compression**: the growth trajectory from 0 to `s*` shortens, generating less cumulative surplus
3. **Entry cost unchanged**: `c_A = 1.2` per period regardless of market size

**Key result — entry deterrence threshold `s_N*`:**

Run `python 06_entry_deterrence.py` to compute the exact value. The script reports `s_N*` and produces a three-panel figure:

- Panel 1: `V(0)` vs `s_N` with entry-viable / deterred zones shaded
- Panel 2: steady-state `s*` vs market ceiling `(1 − s_N)`
- Panel 3: exact `V(0)` versus a perpetuity approximation `π_ss / (1 − β)` — shows how much of the NPV gap comes from the growth path vs. the steady-state level

**Economic interpretation:**

The threshold `s_N*` is a structural entry barrier arising purely from market-space compression — not from predatory pricing, exclusionary contracts, or strategic deterrence by Netflix. This formalizes a stronger claim than Extension 05: incumbents with sufficient market share do not need to *act* to deter entry; the market geometry does it for them.

This maps to Bain's (1956) concept of innocent entry barriers, extended to a dynamic setting: the barrier is not a sunk cost but a negative-NPV condition imposed by the state space itself.

Output: `entry_deterrence.png`

---

## Known Limitations

1. `s_N` is treated as **time-invariant**. In reality Netflix's share fluctuates; see `04_extension1.py` for a dynamic treatment.
2. The model abstracts away **content investment** as a separate state variable; `δ_R` proxies retention effort as a fixed cost rather than an optimized stock.
3. Retention and acquisition parameters are **design-based baselines** (except `δ_S = 0.105` from Iyengar et al. 2011 and `β = 0.95`); conclusions on threshold magnitudes should not be overstated.

---

## Key References

- Erickson (1992) *Marketing Science* — Lanchester transition dynamics
- Rhouma & Zaccour (2018) *Management Science* — methodology template
- Iyengar et al. (2011) *Marketing Science* — δ_S calibration
- Bakos & Brynjolfsson (1999) *Management Science* — SVOD bundling theory
- Rohlfs (1974); Katz & Shapiro (1985); Arthur (1989) — critical mass, lock-in
