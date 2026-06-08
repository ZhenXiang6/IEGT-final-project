# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A course research project (Information Economics & Game Theory) that models an OTT streaming
platform as an **infinite-horizon discounted dynamic optimization problem** and solves it by
**value iteration**. The single state is market share `s ∈ [0,1]`; each period the firm picks one
of 4 actions `(P,C) ∈ {T,S}×{A,R}` (pricing × customer strategy). `README.md` holds the full
economic specification, calibration sources, and the written-up results — read it before changing
any parameter or interpreting output, and keep it in sync when results change.

## Running

```bash
pip install numpy matplotlib        # only dependencies; not present in the default python3
python 03_solve_model.py            # baseline: 3-regime policy, value fn, simulated paths
python 04_extension1.py             # + κ·s retention network externality → bifurcation / multiple steady states
python 05_competitor_fixed.py       # + fixed incumbent share s_N → challenger's reduced-pool problem
```

Each script is standalone, prints a policy/results summary to stdout, and writes one PNG. There is
no build, test, or lint setup, and no `requirements.txt`.

**Gotcha — broken output paths.** `savefig` targets are hardcoded to other machines and will crash
or silently write nothing on this repo: `03` → `/home/claude/output/...`, `04` and `05` →
`/Users/chenyiting/NTU/...`. Change these to a local/relative path (e.g. `./strategy_map.png`)
before running. The committed PNGs (`extension1_multi_steady.png`, `competitor_fixed.png`) are
prior outputs.

### Presentation (`OTT_strategy_slides.tex`)

A Beamer deck summarizing the whole report. It is **Traditional Chinese** and **must build with
XeLaTeX** (`ctexbeamer` + the `PingFang TC` macOS system font); `pdflatex` will fail on the CJK.
The local TeX Live is the *basic* scheme, so the deck deliberately uses only built-in Beamer +
TikZ — no `metropolis`, `pgfplots`, or `booktabs`. Run twice for the TOC/frame counter:

```bash
xelatex OTT_strategy_slides.tex && xelatex OTT_strategy_slides.tex
```

The deck follows the **report's** corrected model (acquisition `α` is `s`-independent; single
`(1−s)` saturation), which is the spec in the report's Appendix A — not the still-quadratic form
left in `03_solve_model.py`. The project's color palette is reused as the slide accent colors.

## Architecture

The three numbered scripts (`03/04/05` = report sections; `01/02` were earlier, uncommitted steps)
are **independent re-implementations of one template**, not modules importing shared code. Each file
repeats the same blocks top-to-bottom:

1. **Parameters** — calibrated constants (`BETA=0.95`, `p_S`, `p_T`, retention `rho_0/delta_*`,
   acquisition `alpha_0/tilde_*`, costs `c_A/c_R`).
2. **Primitives** — `revenue(s,P)`, `cost(C)`, `rho(...)`, `alpha(...)`, `transition(...)`, `profit(s,a)`.
3. **`value_iteration()`** — the solver: a 201-point grid over `s`, Bellman update
   `Q = profit + BETA·V(s')` for all 4 actions, off-grid `V(s')` via `np.interp`, iterate to
   `tol=1e-7`. Returns `(s_grid, V, policy)` where `policy` holds the argmax action index per grid point.
4. **`simulate()`** — roll market share forward from `s0` under the converged `policy`.
5. Print policy-by-`s`-range summary, then `matplotlib` plotting.

`ACTIONS`, `ACTION_LABELS`, and `ACTION_COLORS` are duplicated verbatim across all three files.

**Consequence:** there is no single source of truth for the calibration. Changing a parameter or a
primitive means editing the same lines in **all three scripts** to keep them consistent — they are
meant to share one calibration even though the code doesn't enforce it.

### What actually differs between scripts — the transition function

Each "extension" *is* a change to the law of motion `s' = ρ·s + α·(available pool)`; everything else
is near-identical. When comparing or porting logic, focus here:

- **`03_solve_model.py` (baseline):** `α(s)=ᾱ·(1−sᶿ)` with `θ=1`, and the transition multiplies by
  another `(1−s)`, so saturation enters quadratically: `s' = ρ·s + ᾱ·(1−s)²`.
- **`04_extension1.py`:** adds a retention **network externality** `ρ(P,C,s)=ρ₀+δ_P+δ_C+κ·s`
  (capped at 1). This makes `f(s)` non-linear so it can cross the diagonal 3× → saddle-node
  bifurcation and path dependence. The script sweeps `κ∈[0,0.5]`, finds fixed points by sign-change,
  and classifies stability by local slope (`<1` ⇒ stable). Same `(1−s)²` saturation as baseline.
- **`05_competitor_fixed.py`:** incumbent holds an **exogenous fixed share `s_N`**; challenger solves
  over `s_C ∈ [0, 1−s_N]` with `s' = ρ·s_C + ᾱ·(1−s_N−s_C)`. Note the comment-flagged **"alpha bug
  fix": here `α` is `s`-independent** (linear available pool), unlike `03/04`'s `(1−s)²`. So the
  baseline and this extension do **not** use the same saturation form — keep this in mind before
  claiming results are directly comparable.

## 工作紀錄 / TODO

**本次 (2026-06-08):**
- 新增 `CLAUDE.md`(架構導向)、`.gitignore`(LaTeX/Python 產物)。
- 依 `OTT_strategy_model_report-2.pdf` 製作 Beamer 簡報 `OTT_strategy_slides.tex`
  (XeLaTeX + ctexbeamer + PingFang TC),已編譯為 PDF 並逐頁檢視。
- 依使用者自訂的 6 章節大綱(研究背景與動機 / 文獻回顧 / 模型建構 / 求解結果與解讀 /
  延伸模型與結果 / 結論)重建簡報,改採 **寶藍色系 + 方格紙背景** 主題(`rblue=#2B4CC9`,
  grid 由 `\setbeamertemplate{background}` 的 TikZ 畫出),29 頁;四個延伸各一張。
- 更新 `README.md`:加入簡報建置說明與輸出路徑陷阱提醒。

**TODO(後續可做):**
- 修正三支腳本寫死的 `savefig` 路徑為相對路徑,並重生 `strategy_map.png`(報告 Figure 1)。
- 簡報目前用 `extension1_multi_steady.png`(由 04 程式碼跑出,κ=0.18 數值)做插圖,
  與報告 §7 表格(κ=0.3)數值不完全一致;若要嚴格對齊,需重跑 04 或改用報告數值重繪。
- 統一 `03/04/05` 的飽和項形式(目前 baseline 為 `(1−s)²`、competitor 為單一 `(1−s)`)。
- 簡報 `\author{報告團隊}` 為佔位字,需替換成實際報告人姓名。
