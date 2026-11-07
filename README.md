# Trees + SHAP + LLMs = ?

Companion code for the article **"Trees + SHAP + LLMs = ?"** (subtitle: *My recipe for explaining ML model predictions in plain English*) by Benjamin Bodner.

## What the article shows
The recipe: a LightGBM model on expert-engineered features, SHAP for each alert, a feature glossary, and an LLM narrator (Claude Haiku 5.5) that returns checkable JSON. It was tested on simulated card fraud (four planted fraud patterns, so every explanation is graded against the truth), with short summaries of AML, credit (real UCI data), insurance claims and security alerts.

- **What you show the narrator decides the score.** Pinning one fact the SHAP top-10 list left out (shop or online, stored as the two flags `is_card_present` and `is_online`) took the narrator from 63% to 82% correct on 371 cases (+18.9 points), almost all of it on counterfeit cards. A second simulated dataset gave +17.1.
- **Percentiles probably hurt.** Names plus percentiles scored 63%, names alone 68% (-4.3 points [-7.5, -1.1]). A plain "only 0.5% of normal purchases are this high" brought it back to 68%. Not confirmed in world 2.
- **Better features helped about as much as better evidence.** 62 features instead of 43: PR-AUC 0.872 to 0.934, AUROC 0.9973 to 0.9991, 390 of 400 fraud events caught instead of 371, and the narrator went from 63% to 80% at identical evidence (each model explaining its own caught fraud).
- **The glossary is a small effect, and it steers.** +5.1 points over the percentile version, +0.8 over plain names (noise), +1.5 on the better model. One row ("takeovers from far away") produced 38 false takeover calls.
- **Edge cases.** The simulator has the habits that fool real fraud models (trips, holiday silence, app-store micro-bursts, VPNs, new phones). A note I added to fix one case ("inactive for weeks means dormant card") misfired on 32 of 187 takeover and card-testing alerts; removing it raised overall accuracy from 82% to 89%.

## Reproduce
```bash
python -m venv .venv            # Windows: python -m venv .venv && .venv\Scripts\activate
. .venv/bin/activate
pip install -r code/requirements.txt
python run_all.py               # all scripts in one namespace; stdout -> results/<script>.txt
```
- **No API key needed.** Every LLM answer is cached in `llm_cache/responses.jsonl` (14,592 answers), and the narrator replays them. Each script that asks the model prints the replay count and the prompt drift (0).
- **The article's 12 code blocks (ten steps, `code/01`-`10`)** (block 4 is three files: `04_evidence`, `04_llm`, `04_verify`), shown in full. Paste them in order into one notebook, from this repo's root or with `TSL_HOME` set to a clone. They rebuild every table and figure in the main text except five, which come from repo scripts (`code/11`, `13`, `16`, `19` and `20`).
- **Repo-only scripts** (`code/11`-`18`) run after them in the same namespace:
  - `11_feature_rounds.py`: how the 19 new features were chosen (cross-validation on train + val rows only).
  - `12_aml_desk.py`: the AML desk. `14_credit_desk.py`, `15_more_desks.py`: credit (real UCI data), insurance claims and security alerts.
  - `13_world2.py`: the fraud desk again with a new seed.
  - `16_dormant_trap.py`: why some takeover and card-testing alerts are called "dormant card".
  - `17_feature_ceiling.py`: what a plain tree names from the 43 inputs alone. `18_cost.py`: what it all cost.
  - `19_dormant_ablation.py`: the same alerts again with the "inactive for weeks" note removed. `20_habits.py`: which honest habits set off the false alarms.
- **To re-ask the model,** put `ANTHROPIC_API_KEY=...` in `.env`. If the key isn't scoped to a workspace, also put `ANTHROPIC_WORKSPACE_ID=...` in `.env.local`. Haiku 5.5 has no temperature control, so live answers can be worded differently.
- **Runtime** on an i7-10870H laptop with 16 GB RAM (CPU only), replaying the cache: about 5 minutes for blocks 1-10, about 15 minutes for everything.
- **Windows note:** Smart App Control can block freshly downloaded scikit-learn DLLs. The venv was built with `--system-site-packages`, reusing the system's scikit-learn 1.9.0.

## Layout
| Path | What |
|---|---|
| `code/01_setup.py` ... `code/10_better_features.py` | The article's code blocks, in order |
| `code/11_feature_rounds.py` ... `code/18_cost.py` | Repo-only scripts (see above) |
| `tsl.py` | Simulators (fraud, AML, claims, SOC), the 43-feature fraud builder, the credit loader, the plot theme and small helpers |
| `glossary/*.csv` | Feature glossaries (pipe-separated): `fraud.csv` (43 rows), `fraud_v6.csv` (those plus the 19 new features), AML, claims, SOC, and `typologies.csv` |
| `llm_cache/responses.jsonl` | Every narrator answer: key, prompt hash, model, date, tokens, JSON |
| `results/*.txt` | Captured stdout of each script, plus `env.txt` |
| `PREREGISTRATION.md`, `DECISIONS.md`, `CHANGELOG.md` | Records: what was decided and when, including the v6 round |

The first version of the article, with its own numbers (measured on 200 cases instead of all 371 caught events), is in this repository's git history, before the v6 commit.
