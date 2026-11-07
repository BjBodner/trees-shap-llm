# Changelog

## v6 (2026-11-06): better features, better context, 371 cases, a second world, plain words
Benjy's requests: improve the features and AUROC, find out why percentiles seem to hurt, add richer "what larger/smaller means" context, use more cases and a larger variety, write for readers outside the domain (simple terms, domain knowledge shown through edge cases and what works), no "pre-registered" in the main text, bloated feature sets are fine. He approved more cases + a second world and a USD 7 cap (spent USD 5.38 in this round, USD 8.84 for the whole project).

| # | Edit | Source | Status |
|---|---|---|---|
| 1 | Diagnosis before any change: percentiles-hurt was -2.0 points on 200 cases (noise), -4.3 [-7.5, -1.1] on 371. A tree on the 43 inputs alone names the pattern 97.6% of the time, so the narrator's evidence, not the features, is the bottleneck. | Benjy | done |
| 2 | 19 new features (`code/10`), chosen on 4-fold CV over train + val rows (`code/11`); test scored once. PR-AUC 0.872 to 0.934, AUROC 0.9973 to 0.9991, events caught 371 to 390. Selection rule changed at Benjy's request to "best dev-CV PR-AUC, no size penalty" before the test split was scored (Amendment G2). | Benjy | done |
| 3 | New evidence formats (`code/09`): rarity text, fraud rate, past fraud (per row; equal-weight). Cases: every caught event (371) instead of a 200 sample. Original 200 are a subset and reported as a sensitivity check in the repo. | Benjy | done |
| 4 | Second simulated world (`code/13`, seed from SeedSequence(20261106)). | Benjy | done |
| 5 | The article is rewritten for non-specialists: plain terms, a "mess I put in the data" section (edge cases the simulator contains), "what works / what didn't", "Where I was wrong" instead of a scorecard, no "pre-registered" in the main text. | Benjy | done |
| 6 | The dormant-trap analysis (`code/16`): a note I wrote to fix the hero case misfires on 32 of 187 takeover and card-testing alerts. | Finding | done |
| 7 | The v5 article, code and results are frozen in `v5/`. Repo-only v5 scripts (extra conditions, scorecard, figures 5-8) live only there. | Housekeeping | done |
| 8 | Honest framing fixes: the glossary is +5.1 against the percentile version but +0.8 against plain names; rarity alone was not run in world 2; exploratory conditions (A8, A9) are labelled. | Self-review | done |

Review round 1 (review v5, 79.6 weighted, NOT READY because of three over-claims in the intro) fixed: the over-claims, the jargon (15-item pass), disclosures about the simulator and the replication limits, block 9 and 10 clean-ups, figure labels. See REVISION_MEMO.md.

Review round 2 (80.8 weighted, READY AFTER BLOCKERS) fixed: the untested lead sentence, the credit qualifier, simulator facts, stale publish pack; added the measured false-alarm-by-habit table and the dormant-note ablation. See REVISION_MEMO.md.

Review round 3 (81.2 weighted; one citation cap) fixed: the mis-attributed [10], the stale "except three", four over-reaching phrases, ten more plain-language items. Score trajectory 79.6 -> 80.8 -> 81.2. See REVISION_MEMO.md.

Numbers that changed from v5: all fraud-desk numbers (371 cases instead of 200: names 68%, names+pct 63%, glossary 68%, channel pinned 82%). Unchanged: Result 1 (143 matched alerts: 0% / 56% / 80% / 83%), AML, credit, claims and SOC.

## v4 (2026-10-18): simpler, complete code blocks

Benjy's request: "simplify the code blocks a bit and in general simplify the experiments a bit so that the code is preferably complete and it tells the same story. Also the hero plot code should use real data that is discussed in this blog and should be fully reproducible if one follows the code blocks."

His picks (AskUserQuestion): **"Trim, keep numbers"**: same data, models and cached answers, no new LLM calls. **"Keep the 0.97 fraud case"** for the hero, with its code moved up next to the fraud desk.

| # | Edit | Source | Status |
|---|---|---|---|
| 1 | Article code cut from 8 blocks / 539 lines with 9 variant branches to 9 short, complete blocks (about 450 lines) with none. Every table and figure in the main text, hero included, now comes from code shown in the article. | Benjy's request | done |
| 2 | The narrator (`04`) drops the variants only the extra conditions used: abbreviated names, shuffled SHAP, no SHAP, grouped lists. `grade()` keeps the checks the article reports. `summarize()` moves into the block and gains a "confidently wrong" column. | Benjy's request | done |
| 3 | The fraud desk (`05`) runs 8 conditions instead of 17: names hidden, 3 rungs, and the 2 × 2 follow-up grid (top 10 + channel, all 43; with and without the glossary). The other 9 conditions move to repo-only `10_extras.py`. | Benjy's request | done |
| 4 | TreeSHAP is computed only for the alerts being explained, inside `make_cases()`, instead of for all 170,800 test rows. The values are identical (LightGBM's per-row `pred_contrib`). | Simplification | done |
| 5 | The two extra simulated worlds (PR-AUC spread) move to `10_extras.py`. The article quotes the single-world PR-AUC (0.872 / 0.763 / 0.732) and the 3-world means from the repo. | Simplification | done |
| 6 | The hero code (`06_hero.py`) moves from the appendix to right after the hook case, using only the objects the blocks above built. | Benjy's pick | done |
| 7 | Result 1's figure becomes one panel, with PR-AUC in the axis labels (`02_same_alerts.png`). The follow-up figure is made by the article's `08_evidence.py` (`03_glossary_by_view.png`). | Simplification | done |
| 8 | Credit, the two quick desks, abbreviations, list length, the cross-examination, the repeat and the trap become one short section ("Credit, two quick desks and the extra conditions"), with their code in the repo. Figures 3–6 of v3 become repo-only (`04`–`07` now). | Benjy's request | done |
| 9 | `simulate_aml()` moves the AML clean-up lines into `tsl.py`; `QUICK` mode is gone (it needed an API key anyway). | Simplification | done |
| 10 | `10_extras.py` replays the v3 narrator verbatim and **asserts that it writes the article's prompts byte for byte** on every shared case. A synthetic test (300 random cases × 9 conditions) matched before the run. | Integrity | done |
| 11 | The README, image manifest, checklist and memo are updated. The appendix is removed. | Consistency | done |

No numbers change: every LLM answer is replayed from the same cache, with 0 prompt drift and 0 API calls.

### v4 fixes after independent review v3 (80.7 / 81.3 weighted, NOT READY, set by one citation)
| # | Edit | Review factor | Status |
|---|---|---|---|
| R1 | The [15] sentence was backwards since v1. Ichmoukhamedov et al.'s LLMs *followed* most flipped signs and "corrected" a sizeable minority (Table 3: 15–31% sign accuracy against the original signs, against a naive ~4%). Checked on arxiv.org/html/2412.10220 and fixed, along with `DECISIONS.md` and `literature_notes.md`. | F13 (verdict) | done |
| R2 | Hook: "about 5 on AML" became the printed like-for-like 5.0 [+2.0, +8.0], which `09_aml_desk.py` now prints. The intro no longer says "every explanation could be graded against the truth"; that wasn't true for credit. | F14 | done |
| R3 | Recipe step 4 now names the rows that list account takeover, not "one ambiguous row". The scorecard's "+2 points" now says what it's measured against. The held-predictions paragraph lists all 11. | F14 | done |
| R4 | Credit is an H3 again (two short paragraphs). The quick desks get their own H3. The cross-examination bullet is split and shortened. | F4/F5 | done |
| R5 | AML: value errors 6% → 11% with the glossary. A column legend (25th/75th rule) after the desk table. PR-AUC is glossed. | F7/F8 | done |
| R6 | Block 1 imports an explicit list from `tsl.py` (no `import *`). `asset()` raises on anything but a missing cache. Block 9 notes that it continues block 5's random stream. "Before we start" notes the runtime and where figures are saved. | F8/F10 | done |
| R7 | Fig 2's last bar says "+ glossary + percentiles". The hero footer says "their percentiles and the feature glossary" (the LLM saw both). | F11/F12 | done |
| R8 | Blockquotes cut from 10 to 6; two became bold lead-ins. | F4 | done |
| R9 | [10] and [19] year slots cleaned; `citations.md` regenerated from the article. Meta description: "bolder, barely better" (154 chars). Manifest colour claim, checklist heading count, title word count and `env.txt` corrected. | F13/F18/consistency | done |
| — | Repo creation (404), Benjy's edit pass, photo and bio check, title/subtitle and length. | Benjy's call | open |

## v5 (2026-10-20): condensed for teaching (Benjy: "condense and simplify everything, we want it to be for education purposes")
| # | Edit | Status |
|---|---|---|
| 1 | `alert_schema()` and `summarize()` moved to `tsl.py`, and so did two small plot helpers (`bars`, `paired_bars`). Block 1's import line lists every name the article takes from `tsl.py`. | done |
| 2 | The narrator runs cases one after another: no thread pool, no lock. Replays are instant; live calls are slower but simpler to read. The repo scripts do the same. | done |
| 3 | The per-typology pivot table is gone from block 5. Block 8 prints the counterfeit accuracy per condition instead, which is the only part the text used. | done |
| 4 | Article code: 438 → 405 lines. Prose: 3,549 → about 3,000 words. Medium read time: 30 → about 28 min (27 before the review's teaching additions). | done |
| 5 | Prose tightened throughout. "What you'll learn" opens "Before we start", and there's a "how to read a waterfall" note after the hero code. The literature table has 4 rows (the other works are cited in one sentence). The scorecard is grouped by topic, with every failure named. | done |
| 6 | Readability: average sentence 14.1 words, 6.6% over 25 words, no paragraph over 110 words. | done |
| 7 | Companion repo `BjBodner/trees-shap-llm` created, **private** until Benjy says the article is out (`gh repo edit BjBodner/trees-shap-llm --visibility public`). It holds the code, `tsl.py`, `run_all.py`, glossaries, cache, results, images and the records (57 files at the first commit; secret scan clean). The draft and the publish pack stay out (TDS first-publication rule). Running the article's blocks from a fresh clone matched exactly. | done |

### v5 fixes after independent review v4 (84.8 / 85.3 weighted, READY AFTER BLOCKERS)
| # | Edit | Factor | Status |
|---|---|---|---|
| R1 | Block 4 split into three teaching steps: `04_evidence.py`, `04_llm.py`, `04_verify.py`. The `pct_col` lambda became a short function. Block 8 renamed to `08_glossary_effect.py`. | F4/F8/F10 | done |
| R2 | Figures show inline in a notebook (the Agg backend is used only outside Jupyter; `save()` shows then closes), and are still saved to `images/`. | F10/F8 | done |
| R3 | "more decisive, not more correct" → "and only a little more correct". The low-value sentence now gives 4% (trap cases) and 7% (all) against the predicted 30%+. The shorter-list mechanism is hedged, with 19 → 2 false takeover calls on counterfeit cards. | F14 | done |
| R4 | Teaching terms: AML spelled out; the 4 AML typologies defined; `ci95` and `usd_per_1k` added to the column legend; log-odds and ROC-AUC glossed; two real features named. | F8 | done |
| R5 | Credit: internal reasons named a protected attribute 38–40% of the time, the letters never did. [17] "aliases and feature descriptions". [15] notes the reversed order. The runtime is in the closing line. | F7/F13/F10 | done |
| R6 | Pitch note ("17 of 43 rows"), README credit sentence, colour note and checklist pointer; `tsl.py` docstring. | F14/F18 | done |
