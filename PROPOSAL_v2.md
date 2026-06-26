# DICE-Firm: Disentangled Identifiable Counterfactual Engine for Cross-Country Firm Growth

**Workshop proposal — refined draft (v2)**

---

## 0. One-paragraph summary

We propose **DICE-Firm**, a generative model that learns two disentangled latent variables for company growth — a **macro-regime latent** `z_M(c,t)` shared across firms within a country–time block, and a **firm-identity latent** `z_F(i)` invariant across countries and macro regimes. The model is trained on a multi-country firm panel and supports counterfactual generation by **swapping the macro latent while holding the firm latent fixed**. Unlike MacroVAE (ICAIF 2025), which conditions on macro indicators but does not separate firm identity from macro effects, DICE-Firm (i) operates on firm-level fundamentals rather than asset returns, (ii) treats country as a *multi-environment* signal that gives **identifiability** of the disentangled latents under standard nonlinear-ICA assumptions, and (iii) supports rigorous transfer evaluation on countries unseen at training time.

---

## 1. Problem and why it is not solved yet

Let `x_{i,c,t} ∈ R^T` be the growth trajectory (revenue YoY, EBITDA margin, etc.) of firm `i` in country `c` over time. Let `m_{c,t} ∈ R^K` denote contemporaneous macro covariates (GDP YoY, CPI, policy rate, unemployment, industrial production). We assume the data-generating process

```
x = g(z_F, z_M, ε),
z_F  ⫫  z_M | i,c,t      (structural assumption)
z_M  ↩  m   (macro covariates partially observe the regime)
```

where `g` is an unknown nonlinear decoder. We want to (a) recover `z_F` and `z_M` up to identifiable equivalence, and (b) generate `x' = g(z_F, z_M', ε)` for a swapped regime `z_M'` while preserving firm identity.

### 1.1 What the closest published work does and does not do

| Work | Year | Unit | Macro handling | Firm-invariant latent? | Cross-country transfer? |
|---|---|---|---|---|---|
| **MacroVAE** (Kubiak et al.) | 2025 | asset returns (futures) | direct FiLM conditioning | ✗ | global only, no transfer test |
| **Time-Causal VAE** (Acciaio et al.) | 2024 | market returns | causal transport loss | ✗ | ✗ |
| **TNCM-VAE / Causal Market Simulators** | 2025 | market returns | structural causal layer | ✗ | ✗ |
| **DiffLOB** | 2025 | LOB micro-structure | regime intervention | ✗ | ✗ |
| **Domain-Generalization β-VAE for TS** | 2024 | generic forecasting | domain-specific factor | partial | yes, but no counterfactual |
| **CRADLE-VAE** (single-cell) | 2024 | gene expression | counterfactual artifact disentanglement | yes (perturbation vs. artifact) | n/a |
| **DICE-Firm (ours)** | — | **firm fundamentals** | **learned macro latent** | **yes, identifiable** | **yes, with held-out countries** |

The two architectural moves that no prior work combines:

1. **Macro as a learned latent**, not as a conditioning vector. This is exactly the limitation MacroVAE flags as future work.
2. **Country as the auxiliary variable `u`** in the iVAE/identifiable-nonlinear-ICA sense, giving us *identifiability up to permutation and elementwise transform* of `z_F`.

This combination is the contribution.

---

## 2. Identifiability claim (the part that makes this a workshop paper, not a project)

We piggy-back on the identifiable-VAE result of Khemakhem et al. (2020) and its conditional-prior extensions. Let `u = (c, t)` be the auxiliary index. Assume:

- **(A1)** The prior factorizes: `p(z_F, z_M | u) = p(z_F) · p(z_M | u)` with `p(z_M | u)` in a conditional exponential family of sufficient dimension.
- **(A2)** The decoder `g` is injective and piecewise smooth.
- **(A3)** Macro covariates `m_{c,t}` are a sufficient statistic for `z_M`, i.e. `z_M ⊥ x | m`.

Then `z_F` is identifiable up to permutation and componentwise bijection across countries, and `z_M` is identifiable up to affine transformation of macro covariates. We do not need a full proof in a workshop paper — citing the theorem and stating which assumptions we test empirically is enough. The empirical test is the firm-invariance evaluation in §6.

This is the lever that moves the paper from "yet another conditional VAE" to "principled cross-country disentanglement".

---

## 3. Method: DICE-Firm

### 3.1 Architecture

```
┌────────────────────┐
│ macro series m_c,t │──► Macro Encoder ─► q(z_M | m, c, t)
└────────────────────┘                              │
                                                    ▼
┌────────────────────┐                       ┌─────────────┐
│ firm series x_i,c,t│──► Firm Encoder  ────►│  Generator  │──► x̂
└────────────────────┘    q(z_F | x_i,·)     │  pθ(x|z_F,  │
                                             │   z_M)      │
                                             └─────────────┘
                                                    ▲
                                  counterfactual: swap z_M
```

- **Firm encoder** `q_φF(z_F | x_{i,·,·})`: 1D-CNN + self-attention over the firm's full multi-country trajectory if available, else over the single-country trajectory. Output is a *time-invariant* vector.
- **Macro encoder** `q_φM(z_M | m_{c,·})`: Transformer over the macro panel of country `c` around time `t`, returning a sequence `{z_M(c,t)}`.
- **Generator** `p_θ(x | z_F, z_M)`: latent-diffusion decoder over the trajectory (continuous-time score model). We use diffusion *only here*, not as the whole framework, which keeps the model identifiable and tractable.

### 3.2 Training objective

```
L = L_rec + β·L_KL + λ_inv·L_inv + λ_macro·L_macro + λ_cyc·L_cyc + λ_con·L_con
```

| Term | Definition | Purpose |
|---|---|---|
| `L_rec` | `-E[log p_θ(x | z_F, z_M)]` | Reconstruction |
| `L_KL` | `KL[q(z_F|x) ‖ p(z_F)] + KL[q(z_M|m) ‖ p(z_M|c,t)]` | ELBO regularizer; conditional prior on `z_M` is what gives identifiability |
| `L_inv` | Adversarial: gradient-reverse a country classifier on `z_F` | Firm latent contains no country info |
| `L_macro` | `‖f(z_M) − m_{c,t}‖²` for a small MLP `f` | Macro latent must predict observed macro covariates (anchoring) |
| `L_cyc` | `‖E(D(z_F, z_M')) − z_F‖²` after regime swap | Cycle consistency: firm identity survives a swap |
| `L_con` | InfoNCE: same firm across years = positive, different firms = negative | Contrastive firm invariance |

The contrastive loss `L_con` is the practical workhorse for `z_F` invariance and is what makes the model trainable without huge multi-country coverage per firm.

### 3.3 Counterfactual protocol

For evaluation, define **regime templates** `z_M^R` for R ∈ {recession, expansion, high-inflation, rate-hike, recovery} by averaging encoded `z_M` over country–time blocks labeled by an NBER-style recession/expansion indicator and a CPI/policy-rate quantile bucket. For each firm `i`, generate:

```
x_i^R = D(z_F(i), z_M^R) + ε,    R ∈ regimes
```

This is exactly the procedure MacroVAE uses on macro grids, but applied to firm-level trajectories with an *identified* firm component held fixed.

---

## 4. Data plan

### 4.1 Minimal viable version (US-only, what you can do in one quarter)

| Source | Role | Notes |
|---|---|---|
| **FRED** | Macro panel | GDP YoY, CPI, FFR, unemployment, IP — monthly, 1990– |
| **SEC EDGAR (10-K, 10-Q)** | Firm panel | Revenue, COGS, EBIT, total assets — quarterly, ~3000 firms with continuous 10y history |
| (Optional) **CRSP** | Sanity check | Tie firm fundamentals to market returns |

This is enough for an ICAIF/NeurIPS-workshop submission and avoids the multi-country data-cleaning trap.

### 4.2 Multi-country extension (the version that beats MacroVAE)

| Source | Role | Notes |
|---|---|---|
| **World Bank WDI / OECD** | Macro panel | GDP, CPI, rates for ~30 countries |
| **Compustat Global** (if you can get WRDS) | Firm panel | Best coverage |
| **Orbis** (if Bureau van Dijk access) | Firm panel | Private firms too |
| **EDGAR + LSE + TSE filings** | Firm panel | Free fallback for US, UK, Japan |

Currency normalization to constant-USD growth rates, real (deflated) revenue, and **strict time-based split** (train ≤ 2018, validate 2019–2020, test 2021–2024) — the test window must include COVID so we can evaluate counterfactual generation against a real macro shock.

### 4.3 Preprocessing

- Winsorize growth rates at the 1%/99% level per country–year
- Log-transform revenue / assets
- Forward-fill macro to firm reporting frequency; pad missing firm quarters with a learned mask token
- Convert to **YoY growth** to remove level effects; the model never sees raw dollar amounts

---

## 5. Baselines (be ruthless here — this is what reviewers attack)

1. **No-disentanglement CVAE** — single latent `z`, macro concatenated. *This is the MacroVAE-equivalent for fundamentals.*
2. **Unconditional latent diffusion** on firm trajectories
3. **Dynamic factor model** (Stock–Watson) with firm fixed effects
4. **Panel ARIMAX** with macro covariates
5. **β-VAE without country-prior** on `z_M`
6. **DICE-Firm without `L_inv`** (ablation)
7. **DICE-Firm without `L_cyc`** (ablation)
8. **DICE-Firm without `L_con`** (ablation)

If DICE-Firm only beats #1, the paper is weak. The story we want is: DICE-Firm wins on counterfactual validity and invariance even when it ties on raw forecasting.

---

## 6. Evaluation — four blocks

### 6.1 Reconstruction & forecasting
- RMSE / MAPE on held-out quarters
- CRPS for probabilistic forecasts
- Distributional distance (Wasserstein-2) between real and generated firm trajectories

### 6.2 Counterfactual validity
- **Stylized-fact preservation under regime swap**: volatility clustering, growth persistence (AR(1)), tail behavior. Test that swapped trajectories match the stylized facts of the *target* regime, not the source.
- **Natural-experiment test**: pick firm-country pairs that experienced a known shock (e.g., US firms in 2008Q4). Generate the counterfactual "no recession" trajectory and check whether the gap matches the consensus estimate from the economics literature on the cost of the 2008 recession.

### 6.3 Invariance and transfer
- **Leave-one-country-out (LOCO)** training. Encode firms from the held-out country with the firm encoder; check whether reconstruction quality degrades less than the no-disentanglement baseline.
- **Firm-identity stability**: same firm across years should yield `z_F` with cosine similarity above an ablation-set threshold; different firms should not.
- **Country classifier on `z_F`** must achieve below-chance accuracy.

### 6.4 Identifiability check
- **Multi-environment R²**: with two countries that share firms (e.g., dual-listed firms), encode the same firm from each country's data. Correlation between the two `z_F` recoveries should be high. This is the empirical proxy for the identifiability claim in §2.

---

## 7. Risks and how we mitigate them

| Risk | Mitigation |
|---|---|
| Reviewers say "this is MacroVAE for fundamentals" | §2 identifiability claim + LOCO transfer + dual-listed firm test. None of these is in MacroVAE. |
| Multi-country data is hard to get | Ship the US-only version as the main paper and the multi-country as the extension result. |
| Firm-identity `z_F` collapses (β-VAE failure mode) | Contrastive `L_con` + adversarial `L_inv` are exactly designed to prevent collapse. Run the β-warmup schedule. |
| Counterfactuals are unfalsifiable | Use the 2008 / COVID natural-experiment evaluation as ground truth. |
| Identifiability assumptions don't hold in practice | Frame identifiability as motivation, not as a theorem to prove. State which assumptions are empirically tested. |

---

## 8. Contributions

1. The first generative model for **firm-level cross-country panels** with disentangled macro-regime and firm-identity latents.
2. A **multi-environment training scheme** that uses country as the auxiliary variable, with identifiability framing.
3. A **counterfactual evaluation protocol** combining synthetic regime swaps with natural-experiment ground truth (2008, COVID).
4. A **leave-one-country-out transfer benchmark** that no prior firm-panel generative model has reported.
5. Open-source implementation and a curated US-only firm panel from SEC EDGAR aligned with FRED macro.

---

## 9. References to cite (updated)

1. Kubiak, S. et al. **MacroVAE: Counterfactual Financial Scenario Generation via Macroeconomic Conditioning.** ICAIF 2025. ← the paper you must beat.
2. Khemakhem, I. et al. **Variational Autoencoders and Nonlinear ICA: A Unifying Framework.** AISTATS 2020. ← identifiability.
3. Lippe, P. et al. **CITRIS / iCITRIS: Causal Identifiability from Temporal Intervened Sequences.** ICML 2022, 2023. ← multi-environment identifiability.
4. Acciaio, B. et al. **Time-Causal VAE.** 2024.
5. Schweizer et al. **Toward Causal Market Simulators.** 2025.
6. Lu, C. et al. **Invariant Causal Representation Learning (iCaRL).** ICLR 2022.
7. Khemakhem et al. **ICE-BeeM**, 2020. ← energy-based identifiability.
8. DiffLOB. 2025.
9. Kim et al. **CRADLE-VAE.** 2024. ← counterfactual disentanglement analog from biology.
10. Stock, J.H., Watson, M.W. **Dynamic Factor Models.** ← classic baseline.

---

## 10. Timeline (10 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Build SEC EDGAR + FRED pipeline; aligned panel of ~3000 US firms |
| 3 | Implement DICE-Firm v0 (firm encoder + macro encoder + VAE decoder, no diffusion yet) |
| 4 | Add `L_inv`, `L_con`, `L_cyc`; run on US-only |
| 5 | Baselines #1–#5; reconstruction and forecasting tables |
| 6 | Counterfactual evaluation (regime swap + 2008 natural experiment) |
| 7 | Add latent-diffusion decoder; rerun |
| 8 | World Bank + multi-country panel (if data ready); LOCO experiments |
| 9 | Ablations + identifiability check |
| 10 | Writing + open-source release |