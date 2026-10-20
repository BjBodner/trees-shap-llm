# DECISIONS: Trees + SHAP + LLMs (feature glossary framework)

A running log of picks, deviations and prediction outcomes.

## 2026-10-13: Phase 0
- Workspace: `C:\Users\user\OneDrive\Medium\new_posts\trees-shap-llm`.
- Topic (Benjy, chat): a Trees + SHAP + LLM framework. Expert knowledge lives in two places, the **engineered features** and a **feature glossary**. Each glossary entry gives the name, what the feature is, what it is meant to catch, the motivation, and what it means when the value is larger or smaller than expected.
  - A tree model is trained on the engineered features.
  - TreeSHAP gives per-prediction attributions, often a long list.
  - An LLM reads the SHAP scores plus the glossary and writes a readable rationale.
  - Use cases: fraud, AML, credit approve/deny, plus a few others.
  - The article must contrast the framework with the literature on LLM-generated explanations of predictions.
- Related ideas in Benjy's list ("90 Medium Article Ideas" PDF):
  - #4 "SHAP values aren't an explanation until something reads them" [Have]
  - security #25 "From SHAP to technique hypotheses" [Have]
  - #28 "Which explanation format reduces triage time?" [Build]
  - fraud #29 "Drafting suspicious-activity narratives with an LLM, safely" [Build]
  - This article is the general framework piece those ideas hang off.
- Environment:
  - Windows 11, i7-10870H (16 threads), 16 GB RAM, GTX 1650 Ti 4 GB.
  - Python 3.12.10, numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.0, matplotlib 3.11.0, anthropic 1.11.0.
  - Not installed yet: shap, lightgbm, xgboost. No local LLM runtime.
  - **No ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment.** The LLM leg is blocked until Benjy picks a narrator model and provides access (see Phase 1 logistics).
- Repo pattern from past articles: `https://github.com/BjBodner/<slug>`. Suggested: `https://github.com/BjBodner/trees-shap-llm`. Not created; needs Benjy's OK.

## 2026-10-13: Phase 1 recon
- Recon notes are in `research/literature_notes.md` (recon section).
- Nearest neighbour to the framework: Zhang et al. 2026, a LightGBM + TreeSHAP + LLM narration pipeline for money-mule detection in production. Its abstract doesn't specify a glossary or faithfulness metrics. That gap is this article's opening.
- The strongest contrarian source is "Quality Without Usefulness" (2026): good-quality LLM narratives didn't improve task accuracy, inflated confidence, and a placebo text gave the same boost. The article must answer it, not ignore it.
- Awaiting picks (title, hero, hook, arc, venue, repo URL, LLM access).

## 2026-10-13: Phase 1 picks (Benjy)
- **Title:** "Trees + SHAP + LLMs = ?" (23 chars).
  - Benjy's pick. It sits outside the 36–73-char guidance and has no digits.
  - The body must answer the "= ?" plainly.
  - The SEO title in `publish/seo.md` will carry the keywords.
- **Subtitle:** "My recipe for explaining ML model predictions in plain English."
- **Hero:** concept 2, "Same AUC, different story".
  - Two stacked panels from E1: model PR-AUC per feature set, and rationale correctness per feature set.
- **Hook:** "The model says 0.97, the SHAP plot has 43 bars: what do you tell the client?"
  - Both numbers must be real.
  - The fraud desk uses exactly 43 engineered features.
  - The hook case is picked by a pre-registered rule (see PREREGISTRATION).
- **Arc:** 4, "Four desks, one recipe".
  - Desks: fraud, AML and credit, plus a very simple fourth desk with two mini demos (insurance claims, security alert triage).
- **Experiments:** all of E1–E6 approved.
- **Venue:** TDS. **Repo:** `https://github.com/BjBodner/trees-shap-llm`. Benjy said "repo yes". Creation is deferred to the publisher step and needs explicit OK at that point.
- **Narrator LLM:** "the latest Haiku", which is `claude-haiku-4-5-20251001`.
  - This drops the small-vs-large model comparison from E2.
  - The article makes no model-size claims.
- **API key:** Benjy pasted a key in chat. I did not transcribe it, because credentials aren't mine to place.
  - Asked Benjy to put it in `trees-shap-llm/.env` himself.
  - Advised revoking and rotating it afterwards.
  - Scripts read `.env`. It is excluded from the bundle and the repo.
- Data: fraud, AML and the fourth-desk demos are synthetic with planted typologies, disclosed in the article. Credit uses real public data. Theme: dark.

## 2026-10-13: Phase 2–3
- **Literature review** (subagent, 37 sources verified on primary pages): `research/literature_notes.md`. Corrections it found:
  - The CFPB circulars 2022-03 and 2023-03 were **withdrawn on 2025-05-12**. The adverse-action point now rests on Reg B §1002.9(b)(2).
  - SHAPstories is now *Decision Support Systems* 191 (2025).
  - The 2,730 judgments in *Quality Without Usefulness* came from LLM judges, not human participants.
  - The hallucination-taxonomy source from recon couldn't be verified, so it's dropped.
  - Zhang et al. 2026 is closer prior art than recon suggested. Its prompt already includes feature meanings, p50/p90 and the top-10 SHAP values. It lacks larger/smaller semantics, "what it catches", and programmatic checks.
  - Ichmoukhamedov et al. 2024: LLMs followed flipped SHAP signs only 15–31% of the time. This motivates E6.
    - **Correction (2026-10-18, review v3):** this misreads Table 3. The 15–31% is sign accuracy against the *original* signs, where fully following the flips would score about 4%. So the LLMs followed most flips and "corrected" a sizeable minority. The article's sentence is fixed in v4.
- **Environment:**
  - The venv lives outside OneDrive at `C:\Users\user\venvs\trees-shap-llm`. It is built with `--system-site-packages` because Windows Smart App Control blocked a freshly downloaded scikit-learn DLL (`_k_means_common`). The system's already-allowed scikit-learn 1.9.0 is used.
  - `lightgbm 4.7.0` and `shap 0.53.0` installed from PyPI and load fine.
- **Credit data:** OpenML 42477, a mirror of UCI id 350 (CC BY 4.0 per UCI; OpenML lists CC0). Columns x1–x23 are mapped to the UCI names.
- **Pilots** (model side only, no LLM calls, separate seed roots):
  - Fraud seed 777: PR-AUC around 0.99 with 100%-precision alerts, too easy. Added legit look-alikes.
  - Fraud seed 778: expert 0.964 / embed 0.804 / kitchen 0.693. Each typology now shows only some of its tells, and the alert budget moved from 0.5% to 1%.
  - Fraud seed 779: expert 0.941 / embed 0.791 / kitchen 0.737. The kitchen sink now gets the same raw device and profile tables as the expert set (no raw device id, which would leak), plus more confounders. This is the final calibration.
  - AML / claims / secops seeds 900+: PR-AUC 0.924 / 0.811 / 0.994.
  - Credit split seed 902: ROC-AUC raw 0.778 vs eng 0.780. A protected attribute is in the raw top-4 for 8% of joint denials, so a targeted top-up to 50 cases was pre-registered.
- **Hero adaptation:** the pilots show expert features also win on AUC, so "same AUC" doesn't hold across feature sets. To keep Hero 2 honest, an **anonymized** rung was added: the same expert model with names hidden as V01–V43, so AUC is identical by construction. The final hero layout will be decided from results; tell Benjy.
- **PREREGISTRATION.md** was written before any main-seed result or LLM call. SHA-256: `d6c1ff9ebae31f73b3cb118ec9217cf29cc425b7aef90a4eafaa02b9fe24f7e3`.

## 2026-10-13: Benjy's follow-up
- **Narrator** is now `claude-haiku-5-5`, verified on platform.claude.com.
  - Released 2026-10-07, at $0.10 / $0.50 per million tokens.
  - Adaptive thinking; sampling parameters must be omitted.
  - Effort set explicitly to medium; `max_tokens` 4000.
  - Logged as PREREGISTRATION Amendment 1 before any LLM call.
- **Hero redesign** (Benjy): 43 SHAP bars with a big X, then an arrow, then the LLM's one-sentence customer message (real output, labelled with the model and date).
  - The hero is now `images/00_hero.png`, generated in `05_fraud_desk.py`.
  - The E1 figure is `images/02_same_auc.png`, in the body.
  - The separate 43-bar figure is dropped because the hero now shows it.
- **API key:** Benjy asked me to create `.env` with the pasted key. I declined again, since a live credential is not something I place, and gave him a one-line command.
- **Main-seed model-side run** (pre-refactor code, 2026-10-13): PR-AUC over 3 worlds was expert 0.953 ± 0.003, embed 0.764 ± 0.018, kitchen 0.744 ± 0.000.
  - The final full run regenerates all of this.
  - Replicate worlds now use their own merchant-category medians, so the numbers may shift slightly.
- **Code refactor for length:** the plumbing (simulators, feature builder, specs, helpers, the bar-plot helper) is in `tsl.py`. The article's blocks are the files in `code/`, run in one namespace by `run_all.py`. The pre-refactor code is kept in `pilot/code_v1_monolithic/` and is not shipped.

## 2026-10-13: simulator artifact found and fixed (before any LLM call)
- Extracting the hook case (main seed, model side) showed its two largest SHAP drivers were `implied_speed_kmh = 0` (+3.61) and `distance_from_last_txn_km = 0` (+3.59).
- **Cause.** Fraud bursts used one identical location for every transaction (`np.tile`), while each legit transaction got fresh location noise. "Exactly 0 km from the last transaction" therefore marked fraud. That's a simulator leak, not something real data has.
- **Fix in `tsl.simulate_fraud`.** Legit online transactions now come from a stable per-customer IP location (home + a fixed offset); card-present transactions still vary by store. Consecutive legit online transactions now also sit 0 km apart, as they do in real data. The app-store bursts and subscriptions use the same IP.
- This changes the data-generating process. It was found by inspecting SHAP, which is the article's own diagnostic.
- The fix was re-piloted on seed 779 (model side only). It is logged as PREREGISTRATION Amendment 2. No predictions changed.
- **Article beat:** "SHAP caught my own simulator cheating". It's an honest admission and a demonstration of the diagnose step.

## 2026-10-13: hero preview and taxonomy fix
- The new hook (after the location fix) is F138763: score 0.970, planted dormant_reactivation. Top SHAP drivers: device_age_days = 0 (+5.14), cust_txn_count_30d = 0 (+1.95), longest_gap_days_90d = 41.3 (+1.50).
- **Hero preview text.** It came from a Claude Code subagent (`model: haiku`), which reported itself as `claude-haiku-4-5-20251001`, not Haiku 5.5.
  - Its message: "Your card was used on a new device at an unfamiliar merchant during unusual hours. Please verify this transaction immediately."
  - Its typology, account_takeover, was wrong against the planted truth but defensible. That exposed an overlap in the taxonomy, now fixed with a precedence rule (Amendment 3).
  - Preview image: `hero_preview.png` (labelled PREVIEW). The final hero uses the cached `claude-haiku-5-5` pipeline answer.
- **Article beat (honest).** "My first test call got it wrong, and it was my taxonomy's fault." A typology manual with precedence rules is part of the expert knowledge too.

## 2026-10-13: API access and smoke test
- Benjy created `.env` himself. It is UTF-16, written by PowerShell's `>`.
  - Added `tsl.load_env()`, which reads UTF-16/UTF-8 and never prints values. It replaces python-dotenv.
- The key is user-scoped (`sk-ant-usr…`), and the API answered 400: "not scoped to a workspace".
  - Benjy gave his workspace ID (an identifier, not a secret). It is stored in `.env.local`, which is git-ignored.
  - The narrator sends it as the `anthropic-workspace-id` header only when it is set.
- **Smoke test** on pilot seeds, quick mode, 3 calls, separate cache under `pilot/smoke/`:
  - `claude-haiku-5-5` returned valid structured JSON.
  - About 2,800 input and 900–1,050 output tokens per call, thinking included.
  - That is about $0.0008 per explanation.
- Concurrency raised to 12 threads; the SDK retries 429s.

## 2026-10-13/14: first full run (4,701 calls to claude-haiku-5-5; 0 failures, 0 refusals; 71 min)
- **Fraud:**
  - The ladder: anonymized 0% (100% "unclear"), legacy names 78%, readable names 68%, names+pct 66%, glossary 68%, glossary+pct 70%.
  - The SHAP list: top 3 72%, top 5 74%, top 10 70%, all 43 88%, grouped 74%.
  - Cross-examination: permuted SHAP 47%, no SHAP 84%. Repeat agreement 94%.
- **E1** (143 matched cases): embed 0%, kitchen 56%, expert names 80%, expert + glossary 83%.
- **AML:** 89% → 96%. **Claims:** 98% → 100%. **SecOps:** 93% → 95%.
- **Credit:** ROC-AUC 0.777 for both models. Reason precision 98–100%, direction errors 0%.
- **Falsifier 1 fires on the fraud desk** at the top-10 setting: glossary+pct minus names is only 2 points. The thesis changes from "the glossary does the explaining" to "the features do the explaining; the glossary is the seatbelt". It is reported as such.
- **Diagnostic replay** (`pilot/diag/`, cache only, 0 prompt drift over 2,450 answers):
  - Every counterfeit case is card-present, yet `is_card_present` is in the top 10 by |SHAP| for 0% of them; distance_from_home always is.
  - What the LLM said on counterfeit cases:

    | Rung | Counterfeit (correct) | Unclear | Account takeover |
    |---|---|---|---|
    | names | 12 | 29 | 1 |
    | glossary+pct | 13 | 9 | 19 |
    | legacy names | 28 | 17 | 0 |
    | all 43 features | 43 | 1 | 0 |
    | no SHAP | 35 | 7 | 0 |

  - Mechanism: SHAP ranks what moved the score, not what a human needs to tell typologies apart. The model needs distance; a person also needs the channel.
  - Why legacy names beat readable names (an untested hypothesis, labelled as such): evocative readable names (new_merchant, hour_rarity) pull toward account takeover, while abbreviations leave distance as the main cue.
- **Post-hoc treatment (NOT pre-registered, labelled exploratory):** the `top_10+context` condition pins is_card_present and is_online next to the top 10. It adds 200 calls in the second pass.
- **Credit metric fix:**
  - The protected-word regex matched "single" in "a single month", giving 3 false positives in the eng letters. "single" was removed.
  - The metric is now split into `lists_protected` (internal JSON reasons) and `letter_mentions` (the customer-facing letter).
  - The pre-registered combined metric stays reported as defined.
  - True letter mentions of protected attributes are 0 in every condition.
- **Compliance flag** fires on many ordinary denials. It is reported as a false-alarm rate.
- **Fixed:** the "temperature 0" label in the repeat printout now reads "default sampling".

## 2026-10-14: independent review v1 + Benjy chose a pre-registered follow-up
- **Review v1:** 79.0 weighted, NOT READY.
  - Main issue: the "seatbelt" claims overstate the glossary. The value-error drop came from the percentiles; the typology precedence rule lives in the system prompt for every rung; and the glossary raised confident wrong answers (false account-takeover calls).
  - Smaller issues: an 80% vs 68% basis mismatch, E4 4% vs 7%, the numbered-prediction count (11 of 24), the credit eng/names subgroup being only n=15, and the repo-only labelling.
- **Benjy (AskUserQuestion):** a pre-registered follow-up (Addendum F) rather than a reframing.
- **Reader simulation (verify_snippets, v1 draft):** 9 article blocks ran clean in a fresh directory with no key, replaying the cache. Expected flags: 4 repo-only figures, plus 30,000 (dataset size) and 320 (a literature figure).
- **PREREGISTRATION with Addendum F** was written before its new calls; a factual fix to the 'already observed' line was made pre-run. SHA-256: `75f45db9dd3fefa536cfa366ae89d360fe8b86cd1654a4ca078b4fe59484b0f0`.

- **Integrity check (2026-10-14):** removing the inserted Amendment and Addendum sections from PREREGISTRATION.md gives a file whose SHA-256 is `d6c1ff9e...`, which matches the hash recorded before any result. So the amendments were pure additions. The reconstructed file is shipped as `PREREGISTRATION_v1_original.md`.

## 2026-10-14: follow-up results (Addendum F; 635 new calls, 0 prompt drift)
Glossary effect = glossary+pct minus names+pct, on the same 200 fraud cases:

| View | Gain [95% CI] | Confidently wrong (names+pct → +glossary) |
|---|---|---|
| top 10 | +4.5 pts [0.0, +9.0] | 16% → 24% |
| top 10 + channel | -1.0 [-5.0, +3.0] | 14% → 14% |
| all 43 | +4.0 [+0.5, +7.5] | 9% → 8% |

- With the channel pinned, names+pct alone reaches 83%.
- legacy+glossary+pct: 68%, with 40 false account-takeover calls.
- False account-takeover calls: names+pct 6 → glossary+pct 29, and 22 of those 29 cite distance_from_home_km. Its glossary row says it catches "takeovers from far away".
- **F1 and F2 FAILED; F3 FAILED; F4, F5 and F6 HELD.**
- **Decision rule outcome: "neither holds".** The thesis is "features + context do the explaining; the glossary adds little and can steer confident errors". The title is unchanged (a question).
- The credit eng/names targeted subgroup is now n=50, with 0% letter mentions.
- **Totals:** 5,536 cached answers, $3.45.
- **v2 text changes:**
  - The "seatbelt" framing was replaced: "the glossary is a prompt; prompts steer".
  - The value-error drop is now credited to the percentiles.
  - The typology precedence rule is credited to the system prompt.
  - The follow-up section and Fig 7 were added.
  - The full list of failed predictions is in the article.
  - Repo-only material is labelled; the credit excerpt became a feature list; the Desk 4 code block was dropped.
  - Tags: "Machine Learning" → "Model Interpretability".
  - References: [11] wording, [19]/[20] years, [23] split into [23]/[29].
- **Figure polish:** percent ticks, a blue counterfeit series, a labelled chance line, and the Fig 2 callout without an arrow.

## 2026-10-18: v4, simpler and complete code blocks (Benjy's request)
- **Request:** simplify the code blocks and the experiments so the article's code is complete and tells the same story; the hero code must use the blog's own data and reproduce from the blocks.
- **Benjy's picks (AskUserQuestion):** "Trim, keep numbers" over "Rebuild smaller, all inline"; "Keep the 0.97 fraud case" over switching the hero to UCI credit data.
- **What moved:** the article keeps 8 fraud conditions (names hidden, 3 rungs, and the 2 × 2 follow-up grid), Result 1 (E1) and AML (3 rungs). The other 9 fraud conditions, the AML "glossary" rung, E4, the extra simulated worlds, credit, claims, SOC, figures 4–7 and the scorecard are repo-only (`code/10`–`14`).
- **Integrity:** no new LLM calls. `10_extras.py` replays the v3 narrator verbatim and asserts that it produces the article narrator's prompts byte for byte; the full replay ran with the API key blanked (0 calls, 0 prompt drift). The hero and waterfall PNGs are byte-identical to v3.
- **Not a pre-registration change:** same conditions, same cases, same analysis rules. Only where the code lives changed. The single-world PR-AUC is quoted in the article next to the 3-world means from the repo.
- **Hero placement:** the hero code moved from the appendix (Benjy's Oct 13 request) to right after the hook case, as described in the option he picked. If he wants it back at the end, it's one move in `draft_src.md`.
- **Citation fix (review v3, F13):** the article had said that Ichmoukhamedov et al. [15] found LLMs "mostly ignored flipped SHAP signs". Checked on arxiv.org/html/2412.10220: in Table 3, sign accuracy against the original signs is 15–31% after the flip, against a naive about 4% if the model followed every flip. So the LLMs followed most flips and "corrected" a surprisingly large share. The sentence is fixed. P6.1 (shuffled SHAP cuts top-5 citation by 20+ points) was motivated by the misreading. It stays scored as pre-registered (failed), and its failure now agrees with [15] rather than contradicting it.

## 2026-10-20: Benjy's answers on v4
- **Hero placement** right after the hook case: "that's fine".
- **Length:** "let's see how we condense and simplify everything, we want it to be for education purposes". This led to v5 (CHANGELOG): the reviewer's three cuts plus a prose pass, from 30 to about 27 min. The narrator's system prompt and the evidence tables stay word for word, because the cache depends on them.
- **Edit pass:** Benjy will do it tomorrow (2026-10-21).
- **Repo:** "please open a new repo for this project". Created as `BjBodner/trees-shap-llm`, **private** per the publisher rule (private until the article is out). It goes public with `gh repo edit BjBodner/trees-shap-llm --visibility public` when Benjy says so; the article's `asset()` downloads need it public. MIT licence (the publisher default). No gists, no Colab, no Medium draft.
