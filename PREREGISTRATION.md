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

## Amendment 1 (2026-10-13), made before ANY LLM call
- **Narrator changed** to `claude-haiku-5-5` at Benjy's request; it was released 2026-10-07.
  - The model only allows default sampling, so there's no temperature setting.
  - Adaptive thinking runs with `effort: "medium"` (the default, set explicitly).
  - `max_tokens` is 4000, to leave room for thinking.
  - Pricing is $0.10 / $0.50 per million input / output tokens; the cost metric uses these prices.
- **P-R reworded** to "asked twice at default sampling", from "at temperature 0". The 90% band is unchanged.
- **customer_message rule** is now "1 to 2 plain-English sentences, at most 35 words", so it fits the hero as a 2–3 line explanation.
- **Code layout:**
  - The schema now lives in `code/04_narrator.py`.
  - Simulators and helpers moved to `tsl.py`. Same functions, same seeds.
  - In the replicate fraud worlds, the merchant-category medians come from each world's own data.
- **Hero layout** (Benjy): shap's standard waterfall plot of the hook case, with all 43 features in small text and an X through it, then an arrow, then the LLM's 2–3 line customer message. The E1 "same alerts, different features" figure moves into the body.
- **Hero reproducibility.** Haiku 5.5 rejects temperature 0. The hero's text is the cached JSON answer of the pipeline's own glossary+pct call for the hook case. The plot code reads that saved answer, so the figure is byte-reproducible. A live re-call can word things differently, and the P-R repeat check measures how often.
- No prediction, metric or case-selection rule was changed.

## Amendment 2 (2026-10-13), before ANY LLM call
- **Fraud-simulator leak fixed.** Fraud bursts reused one exact location, so "0 km from the last transaction" identified fraud; SHAP on the hook case exposed it.
  - Legit online transactions now come from a stable per-customer IP location.
  - Re-piloted on seed 779 (model side).
  - The main-seed model-side numbers from the earlier run are void. The final run regenerates them.
- No prediction, metric or selection rule changed.

## Amendment 3 (2026-10-13), before the main LLM run
- **One preview call.** It was made on the main-seed hook case (F138763, planted dormant_reactivation), with Claude Haiku 4.5 as a Claude Code subagent, not the pipeline's model or API. Its only purpose was a hero-layout preview.
  - It answered account_takeover. The case has both a brand-new device and a 41-day silence, and the two typology definitions overlapped.
- **Fix.** The fraud-desk taxonomy now has an explicit precedence rule, like a desk's typology manual: dormant_reactivation takes precedence whenever the card had been inactive for weeks, and account_takeover is defined on an active account.
  - The rule applies identically to every condition and was decided before any pipeline call.
- That preview answer is not part of any measured result.
- No prediction, metric or selection rule changed.

## Addendum F (2026-10-14): pre-registered follow-up, written BEFORE its new calls
**Why.** The glossary barely moved fraud accuracy (P2.2 failed: +2 points). Benjy chose a targeted follow-up over reframing (2026-10-14).

**Already observed** (these are the glossary arms; disclosed):

| Condition | Accuracy |
|---|---|
| names+pct, top 10 | 66% |
| glossary+pct, top 10 | 70% |
| glossary+pct, top 10 + channel | 82% |
| glossary+pct, all 43 | 88% |
| legacy names, top 10 | 78% |

The independent review (v1) also showed that the glossary raised confident wrong answers at top 10. On the 50 counterfeit cases, account-takeover answers went from 0 (names+pct) to 19 (glossary+pct). The suspected cause is the "catches: takeovers from far away" text in the distance glossary row.

**Suspected mechanism:** the glossary can't help, and can mislead, when the deciding fact (the channel) never reaches the LLM. Once the evidence is complete, its semantics should help.

**New conditions.** Fraud desk, the same 200 cases, same model, same SHAP values, same cache rules. New calls only for:
- **F-a** `names+pct` with the channel pinned (is_card_present, is_online) next to the top 10;
- **F-b** `names+pct` with all 43 features;
- **F-c** `legacy+glossary+pct`: abbreviated names plus the glossary, top 10;
- **F-d** credit: `eng/names` on the 35 targeted protected-driver denials, which completes the n=50 subgroup (the v1 review found n=15).

**Glossary effect** G(view) = accuracy(glossary+pct) - accuracy(names+pct) on the same cases. The 95% CI comes from a paired bootstrap over cases (2,000 resamples).

**Predictions:**
- **F1** G(top 10 + channel) is at least +5 points, and its paired 95% CI excludes 0.
- **F2** G(all 43) is at least +5 points, and its paired 95% CI excludes 0.
- **F3** G(top 10 + channel) - G(top 10) is at least +3 points: the glossary's value grows once the channel is visible.
- **F4** With the channel pinned, glossary+pct's confident-wrong rate (wrong, not "unclear") is no higher than names+pct's plus 2 points.
- **F5** legacy+glossary+pct lands within 5 points of glossary+pct with readable names (70%).
- **F6** The credit eng/names letters mention a protected attribute 0% of the time on the targeted subgroup.

**Decision rule** for the thesis line. The title "Trees + SHAP + LLMs = ?" stays either way; only the answer to "= ?" changes.
- **F1 and F2 both hold:** "the glossary pays off once SHAP shows the right evidence". The glossary is part of the engine, conditional on context.
- **Neither holds:** "features + context do the explaining. The glossary adds little and can steer confident errors."
- **Exactly one holds:** "mixed". Context first; the glossary helps only in the view where it held.
- The article reports the original P2.2 failure either way.

## Outcomes
_Appended 2026-10-14 after the runs. The predictions above were never edited; `PREREGISTRATION_v1_original.md` reproduces the hash recorded before any result._

- **Original pre-registration:** 11 of 24 predictions held (14 of 27 individual checks).
- **Follow-up (Addendum F):** 3 of 6 held. F1 and F2 failed, so under the decision rule the thesis is "features + context do the explaining; the glossary adds little and can steer confident errors".
- Full scorecard, printed by `code/12_scorecard.py`:

```text
original pre-registration: 11 of 24 predictions held (14 of 27 individual checks)
follow-up (Addendum F): 3 of 6 held
AML glossary gain: +6.5 pts [+3.0, +10.5]
   id                                                                  claim                                                                           measured outcome
 P1.1                       expert PR-AUC beats kitchen and embed by >= 0.05                                           expert 0.897, embed 0.762, kitchen 0.731    HELD
P1.2a                                                    E1 embedding <= 40%                                                                                 0%    HELD
P1.2b                                                 E1 kitchen sink 40-75%                                                                                56%    HELD
P1.2c                                    E1 expert names >= kitchen + 10 pts                                                                         80% vs 56%    HELD
P1.2d                                            E1 expert + glossary >= 80%                                                                                83%    HELD
 P1.3                                         anonymized (same model) <= 45%                                                                                 0%    HELD
 P2.1                                     ladder rises; no inversion > 5 pts                                                   0% < 78% < 68% < 66% < 68% < 70%  FAILED
 P2.2                          glossary+pct - names >= 10 pts, CI excludes 0                                                                     +2% [-2%, +7%]  FAILED
 P2.3                           legacy names within 15 pts of readable names                                                                         78% vs 68%    HELD
 P2.4            hallucinated <= 2% and direction errors <= 5% on every rung                                                      max halluc 0%, max dir_err 1%    HELD
 P2.5           value error: names >= 25%; names+pct and glossary+pct <= 15%                                           names 12%, names+pct 6%, glossary+pct 6%  FAILED
 P3.1       top-3 >= 5 pts below top-10; all-43 within 5 pts at >= 2.5x cost                                    top-3 72%, top-10 70%, all 88%, cost ratio 1.7x  FAILED
 P3.2                                         grouped within 5 pts of top-10                                                                         74% vs 70%    HELD
 P4.1                     low-value error: names >= 30%, glossary+pct <= 15%                                                          names 7%, glossary+pct 8%  FAILED
 P4.2                            glossary gain >= 5 pts larger on trap cases                                                               trap +2%, other +11%  FAILED
 P-A1                                     AML glossary+pct - names >= 10 pts                                                                    +6% [+3%, +10%]  FAILED
 P-A2                             mule_pass_through has the largest AML gain structuring +0%, mule_pass_through +4%, high_risk_corridor +16%, dormant_burst +6%  FAILED
 P5.1                                 credit: eng ROC-AUC within 0.02 of raw                                                               raw 0.777, eng 0.777    HELD
 P5.2                        raw/names: cites protected >= 30%, flags <= 50%                                                               cites 38%, flags 90%  FAILED
 P5.3                               raw/glossary: cites <= 10%, flags >= 80%                                                               cites 40%, flags 94%  FAILED
 P5.4    eng/glossary: cites <= 2%, precision >= 80%, direction errors <= 5%                                                cites 0%, precision 98%, dir_err 0%    HELD
 P5.5                            eng/glossary precision >= eng/names - 5 pts                                                                         98% vs 98%    HELD
 P6.1                            permuted SHAP drops faithful@5 by >= 20 pts                                                                         82% -> 74%  FAILED
 P6.2                                    permuted SHAP accuracy stays >= 60%                                                                                47%  FAILED
 P6.3                                  no-SHAP within 10 pts of glossary+pct                                                                         84% vs 70%  FAILED
 P7.1                            glossary helps both mini desks, claims more                                                             claims +2%, secops +2%    HELD
  P-R                              same prompt twice: typology agrees >= 90%                                                                                94%    HELD
   F1          glossary gain with the channel pinned >= 5 pts, CI excludes 0                                                              -1.0 pts [-5.0, +3.0]  FAILED
   F2             glossary gain with all 43 features >= 5 pts, CI excludes 0                                                              +4.0 pts [+0.5, +7.5]  FAILED
   F3            glossary gain grows by >= 3 pts once the channel is visible                                                                   +4.5 -> -1.0 pts  FAILED
   F4    channel pinned: glossary's confidently-wrong rate <= names' + 2 pts                                                                         14% vs 14%    HELD
   F5 abbreviated names + glossary within 5 pts of readable names + glossary                                                                         68% vs 70%    HELD
   F6               credit eng/names letters mention protected attributes 0%                                                                           0% of 50    HELD

all checks, original + follow-up: 17 of 33 held
narrator spend: 5,536 cached answers, 12,962,404 input + 4,306,016 output tokens, $3.45 at $0.10 / $0.50 per million tokens
```
