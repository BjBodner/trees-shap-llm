# Trees + SHAP + LLMs = ?

Companion code for the article **"Trees + SHAP + LLMs = ?"** (subtitle: *My recipe for explaining ML model predictions in plain English*) by Benjamin Bodner.

## What the article shows
The recipe has four parts: a LightGBM model on expert-engineered features, TreeSHAP for each prediction, a feature glossary, and an LLM narrator (Claude Haiku 5.5) that returns checkable JSON.

It was tested on five desks: card fraud, AML, insurance claims and SOC alerts (all simulated with planted typologies, so every explanation is graded against the truth), plus credit (real UCI data). Findings:
- **Features did most of the explaining.** Hidden names and embeddings gave 0%, the kitchen sink 56%, expert features 80%, and expert features plus glossary 83%, on the 143 alerts all three feature sets caught.
- **SHAP's top-10 list hid the card-present flag.** Pinning the channel features fixed most of it, without the glossary: names + percentiles went from 66% to 83%.
- **Holding everything else fixed, the glossary added 4.5 points on fraud** (top 10) **and 5.0 on AML** [+2.0, +8.0]. On fraud it made the narrator more confident, both right and wrong. False account-takeover calls rose from 6 to 29, and 22 of them cited the row that says distance from home catches "takeovers from far away". A pre-registered follow-up found it adds little once the evidence is complete: -1.0 points with the channel pinned, +4.0 with all 43 features.
- **Credit.** Letters never mentioned protected attributes. The compliance flag over-fired, and the glossary reduced its false alarms on the engineered model from 85% to 51%.
- **Scorecard.** The original pre-registration: 11 of 24 predictions held. The follow-up: 3 of 6 held.

## Reproduce
```bash
python -m venv .venv            # Windows: python -m venv .venv && .venv\Scripts\activate
. .venv/bin/activate
pip install -r code/requirements.txt
python run_all.py               # all scripts in one namespace; stdout -> results/<script>.txt
```
- **No API key needed.** Every LLM answer is cached in `llm_cache/responses.jsonl` (5,536 answers), and the narrator replays them. `results/09_aml_desk.txt` and `results/13_figures.txt` print the replay count and the prompt drift (0).
- **The article's 9 code blocks are `code/01`–`09`, shown in full.** Paste them in order into one notebook, either from this repo's root or with `TSL_HOME` set to a clone. They rebuild every table and figure in the article's main text, the hero image included (`code/06_hero.py`, from the hook case in `code/05_fraud_desk.py`).
- **Repo-only scripts** (`code/10`–`14`) run after them in the same namespace:
  - `10_extras.py`: every other pre-registered fraud condition (abbreviations, the glossary without percentiles, top 3/5/grouped, shuffled SHAP, no SHAP, the repeat), the AML "glossary" rung, the low-value trap (E4) and two more simulated worlds. It uses the full narrator from the article's first version and asserts that it writes the article's prompts byte for byte.
  - `11_credit_desk.py`, `12_more_desks.py`: credit (real UCI data), insurance claims and SOC alerts.
  - `13_figures.py`: figures 5–8 below. `14_scorecard.py`: every pre-registered prediction, held or failed.
- **To re-ask the model,** put `ANTHROPIC_API_KEY=...` in `.env`. If the key isn't scoped to a workspace, also put `ANTHROPIC_WORKSPACE_ID=...` in `.env.local`. Haiku 5.5 has no temperature control, so live answers can be worded differently.
- **Runtime** on an i7-10870H laptop with 16 GB RAM (CPU only): about 6 minutes for the article's blocks on a quiet machine, and about an hour for everything. Most of it is feature engineering, plus the two extra simulated worlds in `10_extras.py`.
- **Windows note:** Smart App Control can block freshly downloaded scikit-learn DLLs. The venv was built with `--system-site-packages`, reusing the system's scikit-learn 1.9.0.

## Layout
| Path | What |
|---|---|
| `code/01_setup.py` … `code/09_aml_desk.py` | The article's code blocks, in order |
| `code/10_extras.py` … `code/14_scorecard.py` | Repo-only: extra conditions, credit, two quick desks, figures 5–8, the scorecard |
| `tsl.py` | Simulators (fraud, AML, claims, SOC), the 43-feature fraud builder, the credit loader, the plot theme and small formatting helpers |
| `glossary/*.csv` | Feature glossaries (pipe-separated) and `typologies.csv` |
| `llm_cache/responses.jsonl` | Every narrator answer: key, prompt hash, model, date, tokens, JSON |
| `results/*.txt` | Captured stdout of each script, plus `env.txt` |
| `images/` | All figures (dark theme, Okabe-Ito colours) |
| `PREREGISTRATION.md` | Predictions, Amendments 1–3, the follow-up (Addendum F) and outcomes. `PREREGISTRATION_v1_original.md` reproduces the hash recorded before any result. |
| `DECISIONS.md`, `CHANGELOG.md` | Running log: picks, pilots, bugs, deviations, the follow-up; the v4–v5 simplification |
| `research/literature_notes.md` | 37 sources, each checked on its primary page |
| `LICENSE` | MIT (code). The UCI credit data is CC BY 4.0; everything else is simulated. |

The article's source bundle (not in this repo) also holds the draft, `REVISION_MEMO.md`, the review reports and the publish pack.

## Claim → file
| Claim in the article | File, where |
|---|---|
| PR-AUC 0.872 / 0.763 / 0.732; over 3 worlds 0.897 / 0.762 / 0.731 | `results/03_models.txt`; `results/10_extras.txt`, last 3 lines |
| Names 68%, names + percentiles 66%, glossary + percentiles 70%; unclear 18% → 6%; confidently wrong 16% → 24%; value errors 12% → 6% | `results/05_fraud_desk.txt`, summary table |
| Top 10 + channel 82%, all 43 88%; $0.666 / $0.708 / $1.148 per 1,000 | `results/05_fraud_desk.txt`, summary table |
| Counterfeit cards 26% / 72% / 86%, and 82% with the channel but no glossary | `results/08_evidence.txt`, second line |
| The hook case F138763 and its answer; the hero image | `results/05_fraud_desk.txt`, `results/06_hero.txt` |
| E1: 143 alerts; 0% / 56% / 0% / 80% / 83%; 1.5x | `results/07_same_alerts.txt` |
| `is_card_present` in 0 of 50 counterfeit top-10 lists; the glossary effect -1.0 / +4.0 / +4.5; false takeover calls 6 → 29, 22 citing distance; 17 of 43 rows | `results/08_evidence.txt` |
| AML 89% / 90% / 96%; +5.0 pts [+2.0, +8.0] like for like, +6.5 pts [+3.0, +10.5] vs plain names; value errors 6% → 11%; corridors 82% → 98% | `results/09_aml_desk.txt` |
| Abbreviations 78%, 29 of 50 unclear, + glossary 68% and 40 false calls; top 3 72% / 52%; shuffled SHAP 47%, faithful@5 82% → 74%; no SHAP 84%; repeat 94%; trap 4% | `results/10_extras.txt` |
| Credit ROC-AUC 0.777 / 0.777; 50 protected-driver denials; flags 90% / 94%; false alarms 75% / 68% / 85% / 51% | `results/11_credit_desk.txt`, `results/13_figures.txt` |
| Claims 98% → 100%; SOC 93% → 95% | `results/12_more_desks.txt` |
| 11 of 24 predictions held; follow-up 3 of 6; 5,536 answers for $3.45 | `results/14_scorecard.txt` |

## Figures
1. `00_hero.png`: shap's waterfall for the hook case, crossed out, next to the LLM's message to the client.
2. `01_shap_waterfall.png`: plot 1 of the hero at full size.
3. `02_same_alerts.png`: Result 1, the same alerts with different features.
4. `03_glossary_by_view.png`: the follow-up, glossary vs no glossary in three evidence views.
5. `04_ladder.png` (repo only): the six-rung glossary ladder on fraud and AML.
6. `05_topk.png` (repo only): SHAP list length, counterfeit cards and cost.
7. `06_cross_exam.png` (repo only): shuffled SHAP and no SHAP.
8. `07_credit_protected.png` (repo only): credit compliance and false alarms.

## Data and licences
- UCI Default of Credit Card Clients (Yeh & Lien), CC BY 4.0, via OpenML (data_id 42477).
- All other data is simulated by `tsl.py`.

## Publish checklist
See `publish/checklist.md`.
