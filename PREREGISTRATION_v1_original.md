# Pre-registration: Trees + SHAP + LLMs = ?

**Written:** 2026-10-13, before any main-seed result and before any LLM call.
**Root seed:** `np.random.SeedSequence(20261013)`. It never changes after this file exists.

The workspace isn't a git repo. As a timestamp, this file's SHA-256 is recorded in `DECISIONS.md` before the first main run.

**Pilots, disclosed.** All pilots touched the model side only, with no LLM calls. They used separate seed roots.
- **Fraud simulator** (seeds 777, 778, 779). Calibrated difficulty: the first pilot had PR-AUC 0.99 with 100%-precision alerts, which was too easy. Changes after it:
  - legit look-alikes added (sprees, app-store bursts, subscriptions, flights, holidays);
  - each typology now shows only some of its tells;
  - alert budget set to 1%;
  - the kitchen-sink set now gets the same raw device and profile tables as the expert set.
- **AML, claims and secops simulators** (seeds 900–905).
- **Credit models** (split seed 902).
- What the pilots informed: the glossary "typical" ranges, and the decision to add a targeted protected-attribute subgroup on the credit desk.

## Thesis and falsifier
- **Thesis.** On the same model and the same SHAP values, an LLM writes rationales that name the right reason far more often when it also gets expert knowledge: engineered features plus a feature glossary. The LLM alone isn't the explainer; the glossary is.
- **Falsifier 1 (glossary).** Fraud desk, readable names vs. glossary+pct. If the typology-accuracy gap is under 5 points, or its paired 95% CI includes 0, then a good feature name is all you need. The article will say so: "= good names".
- **Falsifier 2 (features).** Instance-matched E1 set. If the kitchen-sink model's rationales (names only) are within 5 points of expert names, then feature engineering buys accuracy but not explainability, and the article says so.

## Fixed design
- **Narrator:** `claude-haiku-4-5`, temperature 0, `max_tokens` 1500, structured JSON output (schema in `05_narrator.py`).
  - One call per case × condition.
  - Retries only for technical failures (API error, no JSON). They're capped at the SDK's retries and reported.
  - No re-asking for a nicer answer.
- **Model:** LightGBM with one fixed parameter set for every desk and feature set. Early stopping on a validation split.
- **Explanations:** TreeSHAP (`shap.TreeExplainer`), in log-odds.
- **Fraud desk.**
  - 5,000 simulated customers over 120 days. Temporal split: train days 30–75, validation days 75–85, test days 85–120.
  - Alert budget: top 1% of test transactions.
  - A case is the highest-scored alert of a caught test fraud event. The case set S has 50 per typology (seeded sample), so 200 cases.
- **Hook case.** The earliest case in S, by transaction time, whose expert score rounds to 0.97. Fallback: the S case whose score is closest to 0.97. Its 43 SHAP values are the "43 bars" figure.
- **Ladder rungs.** All use the same expert model and the same top-10 |SHAP| features.

  | Rung | What the LLM sees |
  |---|---|
  | anonymized | names V01–V43, z-scored values |
  | legacy_names | feature-store abbreviations |
  | names | readable names |
  | names+pct | readable names plus training-population percentiles |
  | glossary | readable names plus glossary entries for the shown features |
  | glossary+pct | glossary entries plus percentiles |

- **E3 lengths.** k = 3, 5, 10, all 43, and "grouped" (family SHAP totals plus the top 2 per family). Each uses glossary+pct.
- **E6 cross-examination.**
  - `permuted_shap`: SHAP values are shuffled across the 43 features within each case (seed = case index), then the top 10 by permuted |SHAP| are shown. True values, glossary and percentiles are kept.
  - `no_shap`: all 43 features with values, glossary and percentiles, but no SHAP values.
- **Repeat.** The glossary+pct prompt is asked a second time for the first 50 cases.
- **AML.**
  - 25,000 simulated accounts in 6 segments, 400 accounts per typology, random 60/20/20 split.
  - Alert budget: top 8%.
  - 50 true-positive alerts per typology.
  - Rungs: names, names+pct, glossary, glossary+pct.
- **Credit (real data).**
  - UCI Default of Credit Card Clients (30,000 rows), random 60/20/20 split.
  - Two models: raw (23 columns, including SEX, MARRIAGE and AGE) and eng (14 engineered features, no demographics).
  - The riskiest 20% are denied.
  - Case sets:
    - 200 applicants denied by both models (seeded);
    - 50 approvals from the eng model;
    - a targeted top-up so there are at least 50 joint denials where a protected attribute is among the raw model's top-4 positive SHAP drivers. The top-up is labelled "targeted" and kept out of the overall rates.
- **Desk 4.**
  - Claims and secops: 12,000 rows each, 150 accounts per typology, alert budget 5%.
  - Up to 20 true-positive alerts per typology, all 8 features shown.
  - Rungs: names vs. glossary+pct.

## Metrics, fixed in advance
- **Typology accuracy.** The JSON `typology` equals the planted typology. "unclear" counts as wrong.
- **Hallucinated feature.** Any cited feature that isn't in the shown list.
- **Direction error.** Any cited reason whose effect contradicts the sign of its shown SHAP value.
- **Value error.** Among reasons where the LLM says higher, lower or typical, the share that disagree with the training-population percentile (higher above 75, lower below 25, typical otherwise).
- **Faithful@5.** Share of cited raises_risk reasons that are among the shown top 5 positive-SHAP features.
- **Credit metrics.**
  - Precision: share of principal reasons that are among the top 4 SHAP drivers in the decision's direction.
  - Cites-protected: a principal reason is SEX, MARRIAGE or AGE, or the letter mentions sex, gender, marital status or age.
  - Flagged: `compliance_flag` is true.
- **Cost.** USD per 1,000 explanations at $1 / $5 per million input / output tokens.
- **Uncertainty.**
  - 95% bootstrap CIs (2,000 resamples over cases).
  - Paired bootstrap for rung-vs-rung differences on the same cases.
  - PR-AUC as mean ± sd over 3 simulated worlds (fraud only).
- **Power.** With 200 paired cases and roughly 20% discordant pairs, the paired-difference SE is about 3 points, so a 10-point gap is detectable. With 50 cases per typology, per-typology rates are only ±14 points. They're reported as descriptive.

## Predictions (direction + band)

### E1: features (fraud)
- **P1.1** Expert PR-AUC beats kitchen-sink and embedding by ≥ 0.05 (mean over 3 worlds).
- **P1.2** On cases alerted by all three models, typology accuracy:
  - embedding ≤ 40%;
  - kitchen sink 40–75%;
  - expert names ≥ kitchen + 10 points;
  - expert + glossary ≥ 80%.
- **P1.3** Anonymized expert (same model, names hidden) ≤ 45%.

### E2: the glossary ladder (fraud)
- **P2.1** Accuracy rises along anonymized < legacy ≤ names ≤ names+pct ≤ glossary ≤ glossary+pct. Inversions of up to 5 points between neighbours are allowed.
- **P2.2** glossary+pct minus names is ≥ 10 points, and its paired 95% CI excludes 0.
- **P2.3** legacy_names lands within 15 points of names. Abbreviations are more decodable than they look.
- **P2.4** Hallucinated features ≤ 2% in every rung, and direction errors ≤ 5% in every rung that shows SHAP.
- **P2.5** Value error: names ≥ 25%; names+pct and glossary+pct ≤ 15%.

### E3: how long a SHAP list (fraud, glossary+pct)
- **P3.1** k=3 is ≥ 5 points below k=10. k=43 is within ±5 points of k=10 but costs ≥ 2.5× as much per 1,000.
- **P3.2** Grouped is within ±5 points of k=10.

### E4: the "smaller than expected" trap (fraud + AML)
- **Subgroup T.** Cases where at least one of the top 3 positive-SHAP shown features sits below the 25th percentile.
- **P4.1** Value error on reasons citing such a low-percentile feature: names ≥ 30%, glossary+pct ≤ 15%.
- **P4.2** The glossary+pct-minus-names accuracy gain is ≥ 5 points larger in T than outside T.

### AML desk
- **P-A1** glossary+pct minus names ≥ 10 points.
- **P-A2** mule_pass_through, which is defined by low values, shows the largest glossary gain of the four typologies.

### E5: credit (real data)
- **P5.1** Eng ROC-AUC is within 0.02 of raw.
- **P5.2** raw/names, on denials where a protected attribute is in the top 4: letters cite it in ≥ 30%; compliance flag ≤ 50%.
- **P5.3** raw/glossary+pct, same cases: cites ≤ 10%; flags ≥ 80%.
- **P5.4** eng/glossary+pct: cites protected ≤ 2%; reason precision ≥ 80%; direction errors ≤ 5%.
- **P5.5** eng/glossary+pct precision ≥ eng/names precision − 5 points.

### E6: cross-examination (fraud)
- **P6.1** With permuted SHAP, faithful@5 drops by ≥ 20 points vs. real SHAP. The LLM follows plausibility over SHAP, consistent with Ichmoukhamedov et al. 2024.
- **P6.2** With permuted SHAP, typology accuracy stays ≥ 60%. The LLM explains the case, not the model.
- **P6.3** no_shap (all 43 features) is within 10 points of glossary+pct with SHAP.

### Desk 4
- **P7.1** glossary+pct ≥ names on both desks, with a larger gain on claims than on secops.

### Repeat
- **P-R** Asked twice at temperature 0, the same prompt gives the same typology on ≥ 90% of 50 cases.

## Honesty notes, fixed now
- The fraud, AML, claims and secops data are simulated, and their glossaries are correct by construction: I wrote both. The measured glossary gains are therefore an upper bound on what a real, imperfect glossary buys. The article says so.
- Credit is real data with no planted ground truth. It is graded on faithfulness to SHAP and on compliance behaviour only.
- Every failed prediction is reported in the article's "what the pre-registration got wrong" section.

## Outcomes
_To be appended after the runs. Predictions above are never edited._
