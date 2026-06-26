# DICE-Firm — Implementation Guide

This document is the engineering blueprint for the proposal **DICE-Firm: Disentangled Identifiable Counterfactual Engine for Cross-Country Firm Growth**. It is meant to be followable end-to-end by one researcher with a single GPU (24 GB is enough for the US-only version).

> **Stack:** Python 3.11, PyTorch 2.4, PyTorch Lightning, Hydra, Weights & Biases, DuckDB for data, `sec-edgar-downloader` and `fredapi` for ingestion.

---

## 0. Repository layout

```
dice-firm/
├── configs/
│   ├── data/
│   │   ├── us_only.yaml
│   │   └── multi_country.yaml
│   ├── model/
│   │   ├── dice_vae.yaml
│   │   └── dice_diffusion.yaml
│   ├── train.yaml
│   └── eval.yaml
├── src/
│   ├── data/
│   │   ├── fred.py
│   │   ├── edgar.py
│   │   ├── worldbank.py
│   │   ├── panel.py            # builds aligned firm-macro panel
│   │   └── dataset.py          # PyTorch Dataset
│   ├── models/
│   │   ├── encoders.py
│   │   ├── decoder_vae.py
│   │   ├── decoder_diffusion.py
│   │   ├── dice_firm.py        # main LightningModule
│   │   └── baselines/
│   │       ├── cvae.py
│   │       ├── uncond_diffusion.py
│   │       └── dynamic_factor.py
│   ├── losses/
│   │   ├── elbo.py
│   │   ├── adversarial_invariance.py
│   │   ├── contrastive.py
│   │   └── cycle.py
│   ├── eval/
│   │   ├── forecasting.py
│   │   ├── counterfactual.py
│   │   ├── invariance.py
│   │   └── stylized_facts.py
│   └── utils/
│       ├── seed.py
│       └── windows.py
├── scripts/
│   ├── 01_download_fred.py
│   ├── 02_download_edgar.py
│   ├── 03_build_panel.py
│   ├── 04_train.py
│   ├── 05_eval.py
│   └── 06_make_paper_tables.py
├── tests/
├── pyproject.toml
└── README.md
```

---

## 1. Environment

```bash
mamba create -n dice python=3.11 -y && mamba activate dice
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.4.0 hydra-core==1.3 wandb duckdb pandas pyarrow numpy scipy scikit-learn
pip install fredapi sec-edgar-downloader yfinance wbdata
pip install einops xformers      # attention efficiency
pip install diffusers transformers   # only used in the diffusion decoder
```

Set `FRED_API_KEY` in your shell. EDGAR doesn't require a key but requires a polite `User-Agent` string (your name + email).

---

## 2. Data pipeline

### 2.1 Macro panel (FRED)

`src/data/fred.py`

```python
import pandas as pd
from fredapi import Fred

SERIES = {
    "gdp_yoy": "A191RL1Q225SBEA",   # real GDP YoY
    "cpi":     "CPIAUCSL",
    "core_cpi":"CPILFESL",
    "ffr":     "FEDFUNDS",
    "unrate":  "UNRATE",
    "indpro":  "INDPRO",
    "ust10":   "DGS10",
    "vix":     "VIXCLS",
}

def fetch_us_macro(api_key: str, start="1990-01-01") -> pd.DataFrame:
    fred = Fred(api_key=api_key)
    df = pd.concat(
        {k: fred.get_series(v, observation_start=start) for k, v in SERIES.items()},
        axis=1,
    )
    df = df.resample("Q").last().ffill()
    # Convert to stationary form
    df["cpi_yoy"]      = df["cpi"].pct_change(4)
    df["core_cpi_yoy"] = df["core_cpi"].pct_change(4)
    df["indpro_yoy"]   = df["indpro"].pct_change(4)
    df = df[["gdp_yoy", "cpi_yoy", "core_cpi_yoy", "ffr",
             "unrate", "indpro_yoy", "ust10", "vix"]].dropna()
    return df
```

### 2.2 Firm panel (SEC EDGAR)

EDGAR's XBRL financial facts API is the right entry point. Use `https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json`.

`src/data/edgar.py` — key facts to extract:

```
Revenues, RevenueFromContractWithCustomerExcludingAssessedTax,
CostOfRevenue, OperatingIncomeLoss, NetIncomeLoss,
Assets, AssetsCurrent, Liabilities, StockholdersEquity,
CashAndCashEquivalentsAtCarryingValue, ResearchAndDevelopmentExpense
```

For each CIK, build a quarterly time series. Compute:

- `revenue_yoy = revenue_t / revenue_{t-4} - 1`
- `op_margin = operating_income / revenue`
- `asset_growth = assets_t / assets_{t-4} - 1`
- `r_and_d_intensity = rnd / revenue`

Keep firms with ≥ 40 consecutive quarters. You'll land near 2,500–3,500 firms.

### 2.3 Aligned panel

`src/data/panel.py` produces a parquet file with schema:

```
firm_id (str), country (str), date (qe quarter end),
x_revenue_yoy, x_op_margin, x_asset_growth, x_rnd_intensity,
m_gdp_yoy, m_cpi_yoy, m_ffr, m_unrate, m_indpro_yoy, m_vix,
sector (sic code), mask (bool)   # mask=False on padded quarters
```

Time-based split: train ≤ 2018Q4, val 2019Q1–2020Q4, test 2021Q1–2024Q4.

### 2.4 Dataset

`src/data/dataset.py`

```python
import torch
from torch.utils.data import Dataset

class FirmWindowDataset(Dataset):
    """
    Returns sliding windows:
      x:  (T, F_x)  firm features
      m:  (T, F_m)  macro features
      c:  int       country id
      i:  int       firm id
      ok: (T,)      mask for valid quarters
    """
    def __init__(self, panel_df, window=20, stride=4, split="train"):
        self.window = window
        self.windows = self._build_windows(panel_df, split, stride)

    def _build_windows(self, df, split, stride):
        windows = []
        for (firm_id, country), g in df.groupby(["firm_id", "country"]):
            g = g.sort_values("date").reset_index(drop=True)
            g = self._apply_split(g, split)
            for s in range(0, len(g) - self.window + 1, stride):
                w = g.iloc[s : s + self.window]
                if w["mask"].mean() < 0.7:
                    continue
                windows.append((firm_id, country, w))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        firm_id, country, w = self.windows[idx]
        x = torch.tensor(w[[c for c in w.columns if c.startswith("x_")]].values,
                         dtype=torch.float32)
        m = torch.tensor(w[[c for c in w.columns if c.startswith("m_")]].values,
                         dtype=torch.float32)
        ok = torch.tensor(w["mask"].values, dtype=torch.bool)
        return {
            "x": x, "m": m, "ok": ok,
            "firm_id": firm_id, "country": country,
        }
```

---

## 3. Model

### 3.1 Encoders

`src/models/encoders.py`

```python
import torch, torch.nn as nn
from einops import rearrange

class FirmEncoder(nn.Module):
    """Time-invariant firm latent z_F ∈ R^{d_F}."""
    def __init__(self, in_dim, d_model=128, d_F=32, n_layers=4, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.head_mu     = nn.Linear(d_model, d_F)
        self.head_logvar = nn.Linear(d_model, d_F)

    def forward(self, x, mask):
        h = self.input_proj(x) + self.pos[:, : x.size(1)]
        h = self.transformer(h, src_key_padding_mask=~mask)
        # Masked mean pool
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head_mu(pooled), self.head_logvar(pooled)


class MacroEncoder(nn.Module):
    """Sequence latent z_M(t) ∈ R^{d_M}."""
    def __init__(self, in_dim, d_model=128, d_M=16, n_layers=3, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(enc, n_layers)
        self.head_mu     = nn.Linear(d_model, d_M)
        self.head_logvar = nn.Linear(d_model, d_M)

    def forward(self, m):
        h = self.input_proj(m) + self.pos[:, : m.size(1)]
        h = self.transformer(h)
        return self.head_mu(h), self.head_logvar(h)
```

### 3.2 VAE decoder

`src/models/decoder_vae.py`

```python
class VAEGenerator(nn.Module):
    def __init__(self, d_F, d_M, out_dim, d_model=128, n_layers=4, n_heads=4, T_max=64):
        super().__init__()
        self.firm_proj  = nn.Linear(d_F, d_model)
        self.macro_proj = nn.Linear(d_M, d_model)
        self.pos = nn.Parameter(torch.randn(1, T_max, d_model) * 0.02)
        dec = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model,
                                         batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerEncoder(dec, n_layers)
        self.head = nn.Linear(d_model, 2 * out_dim)  # mean + logvar per timestep

    def forward(self, z_F, z_M):
        T = z_M.size(1)
        h = self.firm_proj(z_F).unsqueeze(1).expand(-1, T, -1) \
          + self.macro_proj(z_M) \
          + self.pos[:, :T]
        h = self.decoder(h)
        out = self.head(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar
```

### 3.3 Latent-diffusion decoder (drop-in v2)

Use `diffusers` `DDPMScheduler` over a 1D U-Net conditioned on `(z_F, z_M)` via cross-attention. Train the VAE version first, swap this in for the final results. Skip this on iteration 1.

### 3.4 The full LightningModule

`src/models/dice_firm.py`

```python
import lightning as L
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


class DiceFirm(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.firm_enc  = FirmEncoder(cfg.x_dim, d_F=cfg.d_F)
        self.macro_enc = MacroEncoder(cfg.m_dim, d_M=cfg.d_M)
        self.decoder   = VAEGenerator(cfg.d_F, cfg.d_M, cfg.x_dim)
        # Anchoring head: z_M should predict observed macro
        self.macro_pred = nn.Linear(cfg.d_M, cfg.m_dim)
        # Adversary: tries to predict country from z_F
        self.country_clf = nn.Sequential(
            nn.Linear(cfg.d_F, 64), nn.ReLU(), nn.Linear(64, cfg.n_countries)
        )

    # ---------- reparameterization ----------
    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    # ---------- losses ----------
    def step(self, batch, stage):
        x, m, ok = batch["x"], batch["m"], batch["ok"]
        c = batch["country_id"]

        mu_F, lv_F = self.firm_enc(x, ok)
        mu_M, lv_M = self.macro_enc(m)
        z_F = self._rsample(mu_F, lv_F)
        z_M = self._rsample(mu_M, lv_M)

        # Reconstruction
        rec_mu, rec_lv = self.decoder(z_F, z_M)
        rec_var = rec_lv.exp().clamp(min=1e-4)
        rec_ll = -0.5 * ((x - rec_mu) ** 2 / rec_var + rec_lv)
        L_rec = -(rec_ll * ok.unsqueeze(-1)).sum() / ok.sum().clamp(min=1)

        # KL terms
        L_KL_F = -0.5 * (1 + lv_F - mu_F.pow(2) - lv_F.exp()).sum(-1).mean()
        L_KL_M = -0.5 * (1 + lv_M - mu_M.pow(2) - lv_M.exp()).sum(-1).mean()

        # Macro anchoring
        m_hat = self.macro_pred(z_M)
        L_macro = F.mse_loss(m_hat, m)

        # Adversarial invariance (gradient reversal)
        c_logits = self.country_clf(grad_reverse(mu_F, alpha=self.hparams.adv_alpha))
        L_inv = F.cross_entropy(c_logits, c)

        # Contrastive: same firm in this batch → positives.
        # Assumes the loader returns each firm at multiple time windows.
        L_con = info_nce(mu_F, batch["firm_id"], tau=0.1)

        # Cycle consistency: swap z_M and check z_F survives
        z_M_perm = z_M[torch.randperm(z_M.size(0))]
        x_cf, _  = self.decoder(z_F, z_M_perm)
        mu_F2, _ = self.firm_enc(x_cf, ok)
        L_cyc = F.mse_loss(mu_F2, mu_F.detach())

        β = self.hparams.beta
        loss = (
            L_rec
            + β * (L_KL_F + L_KL_M)
            + self.hparams.lam_macro * L_macro
            + self.hparams.lam_inv   * L_inv
            + self.hparams.lam_con   * L_con
            + self.hparams.lam_cyc   * L_cyc
        )

        self.log_dict(
            {f"{stage}/{k}": v for k, v in dict(
                L_rec=L_rec, L_KL_F=L_KL_F, L_KL_M=L_KL_M, L_macro=L_macro,
                L_inv=L_inv, L_con=L_con, L_cyc=L_cyc, loss=loss,
            ).items()},
            prog_bar=True, on_step=False, on_epoch=True,
        )
        return loss

    def training_step(self, batch, _):   return self.step(batch, "train")
    def validation_step(self, batch, _): return self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-5)


def info_nce(z, labels, tau=0.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e4)
    labels = torch.tensor([hash(x) for x in labels], device=z.device)
    pos = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos.fill_diagonal_(False)
    if pos.sum() == 0:
        return z.new_zeros(())
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return -mean_log_prob_pos[pos.any(1)].mean()
```

### 3.5 Hyperparameters (start here)

```yaml
# configs/model/dice_vae.yaml
d_F: 32
d_M: 16
x_dim: 4        # revenue_yoy, op_margin, asset_growth, rnd_intensity
m_dim: 6
n_countries: 1  # for US-only
beta: 1.0
lam_macro: 1.0
lam_inv:   0.3
lam_con:   0.5
lam_cyc:   0.1
adv_alpha: 0.3
lr: 3e-4
```

KL-warmup: start `beta=0`, ramp linearly to `1.0` over the first 10 epochs. This is the standard fix for posterior collapse.

---

## 4. Training

```bash
python scripts/04_train.py +data=us_only +model=dice_vae \
    trainer.max_epochs=80 trainer.precision=bf16-mixed
```

Batches of ~256 windows; on a single 24 GB GPU, US-only converges in ~2 hours.

Validation criterion for early stopping: `val/L_rec + 0.1 * val/L_con` (we don't want to stop on a posterior-collapse epoch where `L_rec` is artificially low but `L_con` is high).

---

## 5. Evaluation

### 5.1 Forecasting & reconstruction (§6.1 of the proposal)

`src/eval/forecasting.py` computes RMSE, MAPE, CRPS on the test split. Baselines #1–#4 must use the same windows.

### 5.2 Counterfactual validity (§6.2)

Two passes:

**(a) Regime templates.** Bucket all `z_M` from the training set by NBER recession indicator + CPI quartile + FFR quartile. Compute the centroid per bucket. These are the regime templates.

**(b) 2008 natural experiment.** For each firm with a continuous 2006Q1–2010Q4 history:

```python
z_F = firm_enc(x_{2006Q1:2007Q4})          # encode pre-shock firm
z_M_normal = template["expansion"]
z_M_observed = macro_enc(m_{2008Q1:2009Q4})
x_observed = decoder(z_F, z_M_observed)
x_counterfactual_no_recession = decoder(z_F, z_M_normal)
gap_pred = (x_observed - x_counterfactual_no_recession).mean()
```

Compare `gap_pred` to consensus estimates of the 2008 recession's effect on corporate revenue (e.g., ~10% revenue drop on average). The model should land in the right ballpark and the *cross-sectional dispersion* of `gap_pred` should correlate with known sector vulnerability (financials and cyclicals worst hit).

### 5.3 Invariance & transfer (§6.3)

```python
# Leave-one-country-out, multi-country only
for held_out in countries:
    train_set = panel[panel.country != held_out]
    test_set  = panel[panel.country == held_out]
    model = train(train_set)
    rec_loss = evaluate(model, test_set)["L_rec"]
    print(held_out, rec_loss)

# Country classifier on z_F — must be near chance
clf = train_country_classifier_on(z_F_train, country_train)
acc = clf.score(z_F_test, country_test)
# Report: random_baseline_acc, dice_acc, no_inv_baseline_acc
```

### 5.4 Stylized facts

Use `arch` or hand-rolled tests for:
- Autocorrelation of growth (should match real per regime)
- Volatility clustering (Ljung-Box on squared growth)
- Tail index (Hill estimator on the lower tail)

Apply per-regime and compare real vs. swapped trajectories.

---

## 6. Reproducibility

- Fix seeds (`utils/seed.py` sets numpy, torch, cuda, cudnn deterministic).
- Pin all package versions in `pyproject.toml`.
- Save the exact panel parquet + git SHA as a W&B artifact.
- Release a Docker image after the paper is submitted.

---

## 7. Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `L_rec` plateaus high, `L_KL` near 0 | Posterior collapse on `z_F` | Lower β, increase `lam_con`, longer KL warmup |
| `L_inv` won't go down | Adversary too weak | Wider country classifier, more updates per encoder step |
| Counterfactual trajectories look like the source regime | `z_M` is being ignored | Increase `lam_macro`, check that `m_hat` actually tracks `m` |
| `L_cyc` explodes early | Decoder unstable on out-of-distribution `z_M` | Warmup `lam_cyc` from 0 |
| Same firm gets different `z_F` across years | Window length too short, or `L_con` weight too low | Bump `lam_con` to 1.0; widen window to 24 quarters |
| Diffusion decoder produces noisy trajectories | Too few diffusion steps at sampling | Use DDIM with 50 steps |

---

## 8. Suggested run order for the paper

1. **Week 3 deliverable:** US-only DICE-Firm beats baseline CVAE on `L_rec` and on counterfactual stylized-fact tests.
2. **Week 5:** Sector dispersion of 2008 counterfactual gaps correlates with known vulnerability rankings (this is the headline figure).
3. **Week 7:** Add diffusion decoder, show another 1–2 % RMSE improvement.
4. **Week 8:** Multi-country LOCO transfer experiment with at least 5 countries.
5. **Week 9:** Ablations table.
6. **Week 10:** Write up.

---

## 9. License and data ethics

- FRED: free, public domain, no restriction.
- SEC EDGAR: public; respect the 10 req/s rate limit and include a User-Agent.
- World Bank WDI: CC-BY 4.0.
- Compustat: license required; do not redistribute raw data, only model weights and pipeline code.

---

## 10. What to release on GitHub

- `src/`, `scripts/`, `configs/`
- `data/processed/us_panel.parquet` (derivable from public sources via `scripts/`)
- Pretrained checkpoints for US-only
- A reproduction notebook that loads the checkpoint and reproduces the 2008 counterfactual figure
- The paper PDF and an arXiv link