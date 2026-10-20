# Literature notes: Trees + SHAP + LLMs

## Recon pass (Phase 1, 2026-10-13). Not yet verified unless marked.

- **Martens et al. (2023, v2 later). "Tell Me a Story! Narrative-Driven XAI with Large Language Models."** arXiv:2309.17057.
  - SHAPstories: the LLM gets SHAP values (or counterfactuals) and writes a narrative.
  - Survey: over 90% of a general audience found the narratives convincing.
  - Lens: scenarios / justification. Verify the author list and venue in Phase 2.
- **Zytek, Pido, Alnegheimish, Berti-Equille, Veeramachaneni (2024). "Explingo: Explaining AI Predictions using Large Language Models."** IEEE BigData 2024, arXiv:2412.05145. Verified on the arXiv abstract page.
  - A Narrator writes the narrative and a Grader scores it on accuracy, completeness, fluency and conciseness.
  - Few-shot examples help. Grading is hard in complex domains.
  - The abstract doesn't say whether feature descriptions are inputs; check the paper body.
- **Zytek, Arnaldo, Liu, Berti-Equille, Veeramachaneni (2022). "The Need for Interpretable Features: Motivation and Taxonomy."** SIGKDD Explorations, arXiv:2202.11748.
  - A model is only as interpretable as its features.
  - The paper proposes a taxonomy of feature properties experts need, and transforms that move data from model-ready to interpretable form.
  - This is the intellectual root of the glossary.
- **Zeng (2024?). "Enhancing the Interpretability of SHAP Values Using Large Language Models."** arXiv:2409.00079.
  - The LLM translates SHAP output into plain language for non-technical users. Verify the author.
- **Zhang et al. (2026). "Detection, Attribution, Narration: An End-to-End Pipeline for Explainable Money Mule Identification."** arXiv:2607.17586, submitted 2026-07-20. Verified on the abstract page.
  - LightGBM on 280 engineered features, then TreeSHAP, then an LLM narrator.
  - In production: 89% yield rate vs 61% for rules. Analysts said the narratives cut cognitive load.
  - The abstract specifies no glossary and no faithfulness metrics. This is the **closest prior art**.
- **Co-Investigator AI (2025). Agentic SAR narrative generation.** arXiv:2509.08380.
  - Names hallucination and weak typology alignment as the blockers. Authors not yet checked.
- **"Quality Without Usefulness: LLM-Generated XAI Narratives as Trust Heuristics Rather Than Decision Aids" (2026).** arXiv:2605.26770.
  - High-quality narratives didn't improve accuracy on 5 tasks and inflated confidence.
  - A placebo produced the same confidence boost.
  - Out of distribution, the narratives reduced users' ability to flag bad predictions.
  - **Contrarian anchor.** Authors and venue still to check.
- **"Towards Faithful Agentic XAI: A Verification Method and an Open-World Benchmark" (2026).** arXiv:2605.27879. Faithfulness verification. Details still to check.
- **Hallucination taxonomy for tabular reasoning:** feature-existence, feature-value and comparison hallucination. Source still to pin down (a search snippet). Useful for the automatic checks.
- **CFPB Circular 2022-03:** ECOA and Reg B require accurate, specific principal reasons, even for complex algorithms. Checklist reasons that don't reflect the real reasons don't comply. Pull the primary circular.
- To fetch in Phase 2:
  - TalkToModel (Slack et al. 2023, Nature MI)
  - Kroeger et al. 2023, "Are LLMs post hoc explainers?"
  - Lundberg & Lee 2017 (SHAP)
  - Lundberg et al. 2020 (TreeSHAP, Nature MI)
  - Grinsztajn et al. 2022 (trees on tabular data)
  - Slack et al. 2020 (Fooling LIME/SHAP)
  - Kumar et al. 2020 (problems with Shapley explanations)
  - Rudin 2019
  - x-[plAIn] (Mavrepis et al. 2024)
  - Agarwal et al. 2024 (faithfulness vs plausibility)
  - FinCEN SAR narrative guidance

---

## Deep review (Phase 2)

Done 2026-10-13. Every block below was checked on the primary page named in its **Verified** line: arXiv abstract or HTML, NeurIPS/PMLR proceedings, Crossref DOI record, PMC, Nature, or the regulator's own site or PDF. Quotes are kept under 10 words. Everything else is paraphrased.

### Foundations: SHAP, TreeSHAP, LightGBM, trees on tabular data

### [1] Scott M. Lundberg, Su-In Lee (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems 30 (NIPS 2017). <https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html>
- **Verified:** yes. Title, both authors and venue confirmed on the NeurIPS proceedings abstract page.
- **Key claim (paraphrased):** SHAP gives each feature an importance value for a single prediction. It unifies six earlier attribution methods under one additive framework, with uniqueness guarantees and closer agreement with human intuition.
- **Lens(es):** justification
- **How the article can use it:** Canonical citation for what a SHAP value is: additive, per-prediction, and signed.

### [2] Scott M. Lundberg, Gabriel Erion, Hugh Chen, et al. (10 authors, last author Su-In Lee) (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2(1):56–67. DOI 10.1038/s42256-019-0138-9. <https://www.nature.com/articles/s42256-019-0138-9>
- **Verified:** yes. Title, journal, volume, pages, date (17 Jan 2020) and DOI confirmed on nature.com. The full author list was confirmed on PMC (PMC7326367).
- **Key claim (paraphrased):** TreeExplainer is the first polynomial-time algorithm for exact game-theoretic (SHAP) explanations of tree ensembles, with complexity O(TLD²). It also introduces SHAP interaction values and tools that aggregate many local explanations into a global view, demonstrated on three medical datasets.
- **Lens(es):** justification / treatment
- **How the article can use it:** Explains why trees plus TreeSHAP is the right backbone: attributions are exact for the real model, not a surrogate approximation, and fast enough to run per alert.

### [3] Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS 2017 (NeurIPS 30). <https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html>
- **Verified:** yes. Title, all 8 authors and venue confirmed on the NeurIPS proceedings page.
- **Key claim (paraphrased):** GOSS (gradient-based one-side sampling) and EFB (exclusive feature bundling) make GBDT training up to 20x faster than conventional GBDT at about the same accuracy.
- **Lens(es):** treatment
- **How the article can use it:** One-line citation for the model choice.

### [4] Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35, Datasets and Benchmarks Track. arXiv:2207.08815. <https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html>
- **Verified:** yes. Title, authors and track confirmed on the NeurIPS page. The arXiv title leaves out the word "typical", so cite the NeurIPS title. The rotation finding was read in the paper PDF (Section 5.4, Fig. 5).
- **Key claim (paraphrased):** On 45 tabular datasets, tree models (XGBoost, RF) stay state of the art on medium-sized data (~10K rows). The authors name three inductive biases that neural nets lack:
  - robustness to uninformative features
  - preserving the data's orientation
  - learning irregular functions
- **Rotation finding (Section 5.4, "Finding 3"):**
  - MLPs/ResNets are rotation-invariant learners. Trees are not, because each split looks at one feature, and FT-Transformers are not either.
  - After a random rotation of the (Gaussianized) features, only ResNets keep their performance. The ranking flips: neural nets end up above tree models.
  - The authors read this as evidence that the original basis carries information. Tabular columns carry individual meaning, as their names show.
  - Removing the least important half of the features before rotating shrinks the drop.
- **Lens(es):** causes / justification
- **How the article can use it:** Direct precedent for the planned PCA rotation experiment. Trees degrade when the axes lose their meaning. It also makes the point that meaningful columns are what both trees and the glossary depend on.

### [7] Alexandra Zytek, Ignacio Arnaldo, Dongyu Liu, Laure Berti-Equille, Kalyan Veeramachaneni (2022). The Need for Interpretable Features: Motivation and Taxonomy. ACM SIGKDD Explorations Newsletter 24(1):1–13. DOI 10.1145/3544903.3544905. arXiv:2202.11748. <https://arxiv.org/abs/2202.11748>
- **Verified:** yes. Authors and abstract confirmed on arXiv. The venue, volume, pages and DOI come from the Crossref record, which lists the title as "The Need for Interpretable Features".
- **Key claim (paraphrased):** A model is only as interpretable as its features. The paper makes three arguments:
  - Work should target the "interpretable feature space" that domain experts act on.
  - There should be a taxonomy of the feature properties users need.
  - Transforms that make features interpretable deserve as much attention as transforms that make them model-ready.
- **Lens(es):** causes / justification / gaps
- **How the article can use it:** The intellectual root of the glossary. The glossary turns the interpretable-feature taxonomy into a per-feature artifact the LLM can read.

### LLM explanation of ML predictions (core comparison set)

### [5] David Martens, James Hinns, Camille Dams, Mark Vergouwen, Theodoros Evgeniou (2025). Tell Me a Story! Narrative-Driven XAI with Large Language Models. Decision Support Systems 191:114402. DOI 10.1016/j.dss.2025.114402. arXiv:2309.17057. <https://doi.org/10.1016/j.dss.2025.114402>
- **Verified:** yes.
  - The arXiv abstract page lists the journal ref as DSS 191 (2025) 114402. Crossref confirms the title, the 5 authors, DSS vol. 191, article 114402 and the April 2025 date.
  - The body was read from arXiv HTML v3.
  - Cite as DSS 2025, not arXiv 2023.
- **Key claim (paraphrased):** XAIstories has the LLM narrate SHAP output (SHAPstories) or counterfactuals (CFstories).
  - Over 90% of a general audience found SHAPstories convincing; the body gives 93.2% for lay users and 77.8% for data scientists.
  - 83% of data scientists said they were likely to use them with lay audiences.
  - In a credit-scoring setting, comprehension answers were more accurate with SHAPstories than with SHAP plots alone.
- **Prompt inputs:**
  - The prompt introduces the dataset, its target variable and its features. This is a dataset-level introduction, not a per-feature dictionary with direction meaning.
  - It includes a table of each feature's value and SHAP value, plus the predicted class and whether the prediction was correct.
  - Instructions: focus on the highest absolute SHAP values and keep to 8 sentences or fewer.
  - Not mentioned: per-feature "high/low means" semantics, typical ranges or averages.
  - The main model was GPT-4. The authors found Gemini and Llama less convincing.
- **Evaluation:**
  - Four surveys: 66 lay users and 36 data scientists (qualitative), plus 68 lay users in a comprehension test.
  - No automatic faithfulness check. The authors admit narratives contain assumptions that go beyond the facts and leave assessing them to future work.
- **Domain:** FIFA Man of the Match, Student Performance, German Credit.
- **Lens(es):** scenarios / usefulness / gaps
- **How the article can use it:** The best-known SHAP-to-story precedent. Contrast: dataset-level descriptions only, with convincingness measured but faithfulness unchecked.

### [6] Alexandra Zytek, Sara Pido, Sarah Alnegheimish, Laure Berti-Équille, Kalyan Veeramachaneni (2024). Explingo: Explaining AI Predictions using Large Language Models. 2024 IEEE International Conference on Big Data (BigData), pp. 1197–1208. DOI 10.1109/BigData62323.2024.10825114. arXiv:2412.05145. <https://arxiv.org/abs/2412.05145>
- **Verified:** yes. The arXiv abstract gives authors and the IEEE BigData 2024 note. Crossref confirms the proceedings, pages and DOI. The body was read in arXiv HTML.
- **Key claim (paraphrased):** The Narrator turns SHAP explanations into narratives, and the Grader scores accuracy, completeness, fluency and conciseness. A few hand-written plus bootstrapped exemplars give high scores. Grading is still hard in complex domains.
- **Prompt inputs:**
  - Base instruction.
  - The explanation as `(feature name, feature value, SHAP contribution)` tuples for the top-N features by |SHAP|.
  - A description of the explanation format.
  - A one-line context description, e.g. the model predicts house prices.
  - 1–5 exemplar narratives.
  - No per-feature descriptions, typical ranges, or high/low semantics.
  - The authors say comparative wording such as "larger" can't be graded without distribution information. They suggest passing mean values or feature distributions in future work.
  - Model: GPT-4o for both Narrator and Grader.
- **Evaluation:**
  - LLM Grader. Accuracy checks that features, values and contribution directions match. Completeness, fluency and conciseness are also scored, with conciseness computed automatically from word count.
  - Grader agreement with human labels was 96% on accuracy and 96% on completeness (n=50 validation entries).
- **Domain:** Ames Housing, Student Performance, Mushroom toxicity, PDF Malware.
- **Lens(es):** treatment / diagnosis / gaps
- **How the article can use it:** The closest academic narrator-plus-checker design. Explingo's own limitation (no distribution context for "larger") is exactly what the glossary's larger/smaller-than-expected and typical-range fields supply.

### [8] Xianlong Zeng, Kewen Zhu (2024). Enhancing the Interpretability of SHAP Values Using Large Language Models. arXiv preprint arXiv:2409.00079 (v1 24 Aug 2024; v2 3 Mar 2026). <https://arxiv.org/abs/2409.00079>
- **Verified:** yes. Authors and dates confirmed on the arXiv abstract page. The full text was read from the arXiv PDF (8 pages); the first author is at ohio.edu.
- **Key claim (paraphrased):** A pre-trained LLM rewrites SHAP output as plain-language explanations for non-technical users. The authors claim accuracy is kept and clarity improves, but they show this only through worked examples.
- **Prompt inputs:**
  - Feature names paired with their SHAP values, as tuples, plus an instruction to explain the features' importance.
  - Not mentioned: feature values, feature descriptions, direction semantics, ranges.
  - Model: Mistral 7B, run locally.
- **Evaluation:**
  - None quantitative. The examples are XGBoost on Titanic plus an Adult-income figure.
  - A user study is listed as future work.
  - The limitations section admits incorrect or biased explanations are possible.
- **Domain:** Titanic, Adult income (toy).
- **Lens(es):** scenarios / gaps
- **How the article can use it:** The minimal baseline: names plus SHAP numbers in, prose out, with no check. Use it as the "naive approach" row in the table.

### [9] Dylan Slack, Satyapriya Krishna, Himabindu Lakkaraju, Sameer Singh (2023). Explaining machine learning models with interactive natural language conversations using TalkToModel. Nature Machine Intelligence 5:873–883. DOI 10.1038/s42256-023-00692-8. <https://www.nature.com/articles/s42256-023-00692-8>
- **Verified:** yes. Title, authors, volume, pages, date (27 Jul 2023) and DOI confirmed on nature.com. The arXiv version is 2207.04154.
- **Key claim (paraphrased):** A dialogue system for asking questions about a model.
  - An LLM parses user questions into a grammar of operations: feature importance via LIME/KernelSHAP, counterfactuals, data exploration. It reaches 94.2% parsing accuracy (fine-tuned T5, IID split).
  - The execution engine automatically picks the most faithful explainer.
  - 73% of healthcare workers preferred it over a baseline point-and-click system, and 85% of ML professionals found it easier.
- **Prompt inputs:**
  - The LLM sees only the user's utterance, for parsing.
  - Responses are filled in from templates, not free-form LLM prose.
  - There are operations that describe the dataset and define terms.
- **Evaluation:** parsing accuracy, the human studies above, and an explainer-fidelity selection step.
- **Domain:** tabular healthcare and other tabular datasets.
- **Lens(es):** treatment / usefulness
- **How the article can use it:** A different design point. TalkToModel keeps the LLM out of explanation content and uses templates. The glossary approach gets more fluent prose but has to add checks to win back that safety.

### [10] Nicholas Kroeger, Dan Ley, Satyapriya Krishna, Chirag Agarwal, Himabindu Lakkaraju (2023/2024). In-Context Explainers: Harnessing LLMs for Explaining Black Box Models. arXiv preprint arXiv:2310.05797 (v1 Oct 2023 as "Are Large Language Models Post Hoc Explainers?"; v4 11 Jul 2024 retitled). <https://arxiv.org/abs/2310.05797>
- **Verified:** yes. The current title, the 5 authors and the 4 versions were confirmed on the arXiv abstract page. No journal ref is listed, so cite it as an arXiv preprint. The body was read in HTML v4.
- **Key claim (paraphrased):** LLMs can act as post hoc explainers of other models through in-context learning, using three strategies:
  - Perturb-ICL: 16 perturbed neighbours with model outputs.
  - Perturb+Guide-ICL: the same, plus step-by-step analysis guidelines.
  - Explain-ICL: exemplar input–output–explanation triplets, e.g. from LIME.
  - Explain-ICL comes out roughly on par with LIME on tabular data.
- **Prompt inputs:**
  - Serialized feature values with placeholder labels (e.g. `A: 0.430`), perturbation neighbourhoods and model outputs, or LIME exemplars.
  - No feature semantics.
  - Models: GPT-3.5, GPT-4, GPT-4-0125-preview.
- **Evaluation:** automatic faithfulness metrics against ground truth:
  - Feature Agreement and Rank Agreement on the top-k features.
  - PGI/PGU, the prediction gaps from perturbing important or unimportant features.
- **Domain:** Blood, Recidivism, Adult, Credit, HELOC (tabular); Amazon, IMDb, Yelp (text).
- **Lens(es):** diagnosis / concerns
- **How the article can use it:** The opposite philosophy: the LLM is asked to compute attributions. The article's design deliberately does not. SHAP computes, and the LLM only narrates.

### [11] Yuge Zhang, Yuanxing Zhang, Yichao Jin, et al. (10 authors) (2026). Detection, Attribution, Narration: An End-to-End Pipeline for Explainable Money Mule Identification. arXiv preprint arXiv:2607.17586 (submitted 20 Jul 2026). <https://arxiv.org/abs/2607.17586>
- **Verified:** yes.
  - Title, all 10 authors and the date confirmed on the arXiv abstract page.
  - The prompt (Appendix A, Listing 1; Section 3.5) and the evaluation (Section 4.4, Table 4) were read in arXiv HTML.
  - Full authors: Yuge Zhang, Yuanxing Zhang, Yichao Jin, Khairul Amsyar Mohd Razis, Nicholas Qi An Choo, Kai Yin Anders Wong, Xinyan Tang, Kenneth Zhu Ke, Wee Keong Dennis Lee, Jingyuan Zhao.
- **Key claim (paraphrased):** LightGBM on 280 engineered features (transactions, demographics, network topology, temporal behaviour), then TreeSHAP, then an LLM narrator.
  - In production the yield was 89% vs 61% for the rule-based system.
  - Monthly alerts rose from 211 to 302.
  - Analysts reported a lower cognitive load.
- **Prompt inputs (closest prior art):**
  - Feature names and raw values.
  - Population benchmarks for each feature: p50 and p90 computed across all cases, not peer groups.
  - Per-feature SHAP values for the top 10 features.
  - A `feature_meaning` field explaining what each feature represents.
  - The customer's mule score and its percentile.
  - An explicit sign convention: positive raises mule probability.
  - A fixed JSON schema with `red_flags`, `reasons` and `evidence_references`.
  - An instruction to cite only top-10 SHAP features and not speculate.
  - **Not mentioned:** a per-feature statement of what a high or low value means for risk, which pattern or typology each feature was built to catch, or the motivation behind it. Typology names appear only in the evaluation and discussion, not as prompt content.
  - Models: GPT-OSS-120B (deployed), Qwen3-Next-80B-A3B-Instruct, Gemma-3-27B.
- **Evaluation:**
  - Blinded independent evaluators rated 20 confirmed mule alerts on 1–5 scales. The number and type of raters aren't specified.
  - GPT-OSS-120B scored: factual accuracy 4.20, typology plausibility 3.50, completeness 2.90, actionability 3.35.
  - Qwen3 scored 3.65 / 3.35 / 3.30 / 3.35. Gemma-3 scored 2.55 / 2.75 / 2.80 / 2.60.
  - No programmatic faithfulness check is described. Faithfulness rests on prompt constraints plus periodic human review.
  - Reported failure modes:
    - SHAP magnitudes exaggerated, sometimes by 10x.
    - Credit-to-debit ratio semantics misread, which is a direction-semantics error.
  - The authors warn that even rare hallucinations carry regulatory risk.
- **Domain:** AML / money mules (bank production).
- **Lens(es):** scenarios / gaps / concerns / treatment
- **How the article can use it:** The strongest prior art, and it needs to be acknowledged directly. Its own failure modes (magnitude exaggeration, misread ratio direction) are the errors the glossary's direction fields and the machine-checked structured output target.

### [12] Prathamesh Vasudeo Naik, Naresh Kumar Dintakurthi, Zhanghao Hu, Yue Wang, Robby Qiu (2025). Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives. arXiv preprint arXiv:2509.08380 (v1 10 Sep 2025; v2 17 Sep 2025). <https://arxiv.org/abs/2509.08380>
- **Verified:** yes. Title, 5 authors and dates confirmed on arXiv. The body was read in HTML. No venue is listed. Affiliations aren't stated beyond an acknowledgment of a global fintech company.
- **Key claim (paraphrased):** The authors list three main ways LLMs fail at SAR drafting:
  - factual hallucination
  - limited alignment with crime typologies
  - poor explainability

  Their fix is a multi-agent system: planning, typology detection, intelligence gathering, narrative drafting and compliance validation, with a human investigator in the loop.
- **Prompt inputs:**
  - Structured transactions, account metadata and extracted risk indicators.
  - Typology classifications from tree ensembles (RF/GBM).
  - External intelligence via MCP, regulatory memory and historical SARs.
  - No SHAP attributions are mentioned.
- **Evaluation:**
  - A Compliance Validation agent ("Agent-as-a-Judge") cross-checks claims against the agents' outputs and runs rule-based checks on dates, amounts and jurisdictions. No hallucination rates are reported.
  - 6 AML investigators rated the output:
    - 70% narrative completeness and 61% time savings.
    - Per-typology detection: 100% location anomaly, 93% account integrity, 90% dispute pattern.
- **Domain:** AML SAR narratives.
- **Lens(es):** scenarios / concerns / treatment
- **How the article can use it:** Independent confirmation from AML practice that hallucination and typology misalignment are the blockers. The glossary encodes typology intent for each feature.

### [13] Fabian Lukassen, Jan Herrmann, Christoph Weisser, Alexander Silbersdorff, Benjamin Saefken, Thomas Kneib (2026). Quality Without Usefulness: LLM-Generated XAI Narratives as Trust Heuristics Rather Than Decision Aids. arXiv preprint arXiv:2605.26770 (26 May 2026). <https://arxiv.org/abs/2605.26770>
- **Verified:** yes. Title, 6 authors and date confirmed on arXiv. The design and numbers were read in HTML. No venue is listed.
- **Key claim (paraphrased):** The authors ran five controlled experiments (2,730 judgments over 60 test instances).
  - Narratives judged high-quality did not improve accuracy on any task.
  - They raised self-reported confidence, including on wrong answers.
  - A placebo narrative gave the same confidence boost.
  - In out-of-distribution detection, narratives halved accuracy at flagging unreliable predictions, from 30% to 15% (p=.0002).
  - The authors call this the "Quality-Usefulness Gap".
- **Prompt inputs:**
  - Domain context (French household electricity, 2006–2010), the prediction, raw feature values (lags, ISO week, holidays), SHAP values sorted by |SHAP|, the base value, and model metrics.
  - No per-feature semantics.
- **Evaluation:**
  - **Important caveat: the judges were LLMs (GPT-4o and DeepSeek-R1), not humans.** The authors present the results as behavioural analogues of human judgment.
  - Accuracy odds ratios were not significant: E1 OR 0.83, p=.53; E3 OR 1.37, p=.27; E4 OR 1.86, p=.18.
- **Domain:** time-series energy forecasting (XGBoost + TreeSHAP).
- **Lens(es):** concerns / usefulness
- **How the article can use it:** The contrarian anchor. Fluent, high-quality narratives can raise trust without helping decisions. The article should not claim usefulness from fluency, and should report checks rather than vibes. Say plainly that the judges were LLMs.

### [14] Jaechang Kim, Sunung Mun, Seungjoon Lee, Jaewoong Cho, Jungseul Ok (2026). Towards Faithful Agentic XAI: A Verification Method and an Open-World Benchmark for Better Model Faithfulness. arXiv preprint arXiv:2605.27879 (27 May 2026). <https://arxiv.org/abs/2605.27879>
- **Verified:** yes. Title, 5 authors and date confirmed on arXiv. The method was read in HTML. No venue is listed.
- **Key claim (paraphrased):** FAX breaks a draft explanation into atomic claims. It tests each claim with tools that are faithful by construction (state editing and counterfactual queries to the target model) and keeps only the corroborated ones.
  - On CRAFTER-XAI-Bench (RL policies), simulation faithfulness rose from 0.20 for the best baseline to 0.46, with informativeness, relevance and fluency staying high.
  - On tabular data (TalkToModel protocol) it scored 0.70 vs 0.65 for TalkToModel. The authors warn that tabular scores may partly reflect the LLM's prior.
  - SHAP is treated as noisy post hoc evidence that itself needs verification.
- **Prompt inputs:** agentic tool calls. SHAP is used only to propose candidate claims.
- **Evaluation:** simulation faithfulness. An LLM reads the explanation and predicts the model's decision on unseen states, and the prediction is compared with the actual output.
- **Domain:** RL (Crafter); tabular (diabetes, COMPAS, German credit).
- **Lens(es):** treatment / diagnosis
- **How the article can use it:** Same "decompose, then verify" idea as the article's structured-output checks. The article's version is cheaper: claims are checked against the SHAP list rather than by re-querying the model.

### [33] Timour Ichmoukhamedov, James Hinns, David Martens (2024). How good is my story? Towards quantitative metrics for evaluating LLM-generated XAI narratives. arXiv preprint arXiv:2412.10220 (13 Dec 2024). <https://arxiv.org/abs/2412.10220>
- **Verified:** yes. Title, 3 authors (Univ. Antwerp) and date confirmed on the arXiv HTML v1 page. Table 3 and Section IV.2 were read. No venue is listed.
- **Key claim (paraphrased):** A second "extraction" LLM (GPT-4o) pulls rank, sign and value for each feature out of a narrative and compares them with the SHAP table. That gives three automatic metrics: Rank Accuracy, Sign Accuracy and Value Accuracy.
  - Sign Accuracy was 0.96–0.99 across GPT-4o, Claude 3.5, Llama-3-70B and Mistral-Large; human-written narratives scored 0.97.
  - Rank Accuracy was 0.82–0.93.
  - **Key finding:** ranks and signs in the prompt were inverted while values, averages and descriptions stayed the same. The LLMs followed the manipulated sign only about 15–31% of the time. They mostly reverted to the "plausible" sign, which the authors put down to parametric-knowledge bias.
- **Prompt inputs:**
  - Feature name, SHAP value, feature value, **training-set average**, a **short feature description**, a dataset description, and instructions to stress rank, sign and reasoning.
  - No per-feature high/low semantics.
- **Evaluation:** automatic metrics via an extraction LLM; Random Forest + TreeSHAP; 20 instances per dataset.
- **Domain:** FIFA, German Credit, Student Performance.
- **Lens(es):** diagnosis / concerns / treatment
- **How the article can use it:**
  - The closest precedent for machine-checking direction and rank. Note that the article's structured output removes the need for an extraction LLM.
  - The sign-override result is the strongest warning for a glossary. Rich semantics can make the LLM narrate the prior instead of the model.

### [34] Yifan He, David Martens (2026). An Agentic Approach to Generating XAI-Narratives. arXiv preprint arXiv:2603.20003 (20 Mar 2026). <https://arxiv.org/abs/2603.20003>
- **Verified:** yes. Title, both authors and date confirmed on arXiv. The body was read in HTML. No venue is listed.
- **Key claim (paraphrased):** A Narrator agent revises its narrative using feedback from critic agents, including a rule-based faithfulness evaluator.
  - With Claude Sonnet 4.5, unfaithful narratives fell from 31/100 to 3/100 within a few rounds, roughly a 90% reduction.
  - Sign Accuracy went from 0.958 to 1.000 and Rank Accuracy from 0.905 to 0.990.
  - Ensemble voting helped 4 of the 5 LLMs (GPT-5, Claude Sonnet 4.5, Mistral Medium 3.1, Llama 3.3 70B, DeepSeek-V3.2-Exp).
- **Prompt inputs:** dataset context, task, feature names and values, SHAP values, average feature values, feature descriptions, predicted probability; at most 10 sentences covering the top-4 features.
- **Evaluation:** an LLM extracts rank, sign and value, then rule-based comparison against SHAP. Any mismatch counts as unfaithful.
- **Domain:** Student, FIFA, German Credit, Diabetes, Stroke.
- **Lens(es):** treatment / diagnosis
- **How the article can use it:** Shows that a baseline single-pass narrator gets about 31% of narratives wrong somewhere, and that check-and-retry fixes most of them. This supports the article's checker-plus-retry design and gives a reference rate to compare against.

### [35] Latifa Dwiyanti, Sergio Ryan Wibisono, Hidetaka Nambo (2025). ContextualSHAP: Enhancing SHAP Explanations Through Contextual Language Generation. WSSE '25 (2025 7th World Symposium on Software Engineering), per arXiv journal-ref. arXiv:2512.07178. <https://arxiv.org/abs/2512.07178>
- **Verified:** yes. Title, 3 authors, the WSSE '25 journal-ref and the date (8 Dec 2025) confirmed on arXiv. The body was read in HTML. The ACM page itself was not opened.
- **Key claim (paraphrased):** A Python package that extends SHAP with GPT-generated contextual text. Users supply feature aliases, descriptions and background. In a small healthcare case study, text plus plots were rated more understandable than plots alone.
- **Prompt inputs:**
  - Feature names, aliases and descriptions; semantics of encoded categories and units; background on the model's purpose.
  - A table of per-feature SHAP values, plus bar and waterfall plot images.
  - Guard prompts that explain E[f(X)] vs f(x).
  - No direction semantics or typical ranges are documented.
  - Model: GPT-4o.
- **Evaluation:**
  - 7 users (5 lay, 2 doctors) answered Likert surveys and interviews. Lay understanding rose from 1.55 to 3.28.
  - No faithfulness check.
- **Domain:** healthcare (HCV liver disease, XGBoost).
- **Lens(es):** scenarios / gaps
- **How the article can use it:** The closest "feature dictionary" precedent. It covers descriptions and aliases, but not risk-directional larger/smaller semantics, ranges, or checks.

### [36] Wenxi Geng, Dingyuan Liu, Liya Li, Yiqing Wang (2026). Evaluating Large Language Models as Post Hoc Explainability Interfaces for Credit Risk Models. Expert Systems 43(8): e70351. DOI 10.1111/exsy.70351. arXiv:2602.18895 (arXiv title: "Could Large Language Models work as Post-hoc Explainability Tools in Credit Risk Models?"). <https://doi.org/10.1111/exsy.70351>
- **Verified:** yes. Authors and dates confirmed on arXiv. The DOI is listed there, and Crossref confirms the Expert Systems publication (online 30 Jun 2026) under a different title. Use the journal title.
- **Key claim (paraphrased):** Tested on LendingClub: 39,239 loans, 24 features.
  - When given a ranked attribution list, GPT-4-turbo, Claude Sonnet 4.5 and Gemini 2.5 Flash reproduced it perfectly: top-K overlap 1.0, Kendall τ 1.0.
  - Asked to explain on their own from raw features, they matched SHAP or coefficients poorly: top-4 overlap about 0.06–0.20 zero-shot, about 0.18–0.52 few-shot.
  - Conclusion: LLMs work as narrative interfaces, not as attribution engines.
- **Prompt inputs:**
  - Controlled setting: feature names plus the ranked attribution list.
  - Autonomous setting: raw feature names and values plus the predicted PD (probability of default).
  - Feature descriptions: not mentioned.
- **Evaluation:** top-K overlap, Kendall τ and schema-validity checks against SHAP or coefficients.
- **Domain:** consumer credit.
- **Lens(es):** justification / concerns
- **How the article can use it:** Supports "SHAP computes, the LLM narrates". LLMs copy given attributions faithfully but invent poor ones on their own.

### [37] Laura R. Marusich, Mary Grace Kozuch Dhooghe, Jonathan Z. Bakdash, Murat Kantarcioglu (2026). Human Decision-Making with Persuasive and Narrative LLM Explanations. arXiv preprint arXiv:2605.23867 (22 May 2026). <https://arxiv.org/abs/2605.23867>
- **Verified:** yes. Title, 4 authors and date confirmed on arXiv. The design was read in HTML. No venue is listed.
- **Key claim (paraphrased):** 320 Prolific participants across 4 conditions.
  - Neutral, low-persuasion and extreme-persuasion GPT-4o narratives did not change decision accuracy compared with the prediction alone (F=0.91, p=.44).
  - Narratives raised reliance on the AI for both correct and incorrect predictions (p=.03).
  - Extreme persuasion slowed responses.
- **Prompt inputs:** **no SHAP.** GPT-4o produced the predictions and explanations itself, which makes this LLM self-explanation, not narration of an attribution.
- **Evaluation:** a human behavioural experiment on Census Income and Student Performance.
- **Lens(es):** concerns / usefulness
- **How the article can use it:** A human-subject complement to [13]. Narratives raise reliance without raising accuracy. Its scope differs: the LLM is the predictor here.

### [25] Philip Mavrepis, Georgios Makridis, Georgios Fatouros, Vasileios Koukos, Maria Margarita Separdani, Dimosthenis Kyriazis (2024). XAI for All: Can Large Language Models Simplify Explainable AI? arXiv preprint arXiv:2401.13110 (23 Jan 2024). <https://arxiv.org/abs/2401.13110>
- **Verified:** yes. Title, 6 authors and date confirmed on arXiv. The body was read in HTML. No journal ref.
- **Key claim (paraphrased):** x-[plAIn] is a custom GPT, built with ChatGPT's GPT builder, that summarizes outputs from various XAI methods for audiences at different expertise levels. The authors report that over 80% of study participants preferred its descriptions to those in the original papers.
- **Prompt inputs:** **images** of XAI outputs (SHAP, LIME, Grad-CAM, PDP, Integrated Gradients), sometimes with the research problem, plus the audience level. No structured feature data or descriptions.
- **Evaluation:** a preference survey with the participant count unspecified. No faithfulness check. The authors note the model sometimes explained the image rather than the XAI result.
- **Domain:** boar taint, fake news, BERT sentiment, COVID X-ray, hyperparameter optimization.
- **Lens(es):** scenarios / gaps
- **How the article can use it:** The "explain the plot" end of the spectrum, the weakest grounding. A useful contrast row.

### Critiques of post hoc explanation and LLM explanations

### [21] Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh, Himabindu Lakkaraju (2020). Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods. Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES '20), pp. 180–186. DOI 10.1145/3375627.3375830. arXiv:1911.02508. <https://arxiv.org/abs/1911.02508>
- **Verified:** yes. Authors and abstract confirmed on arXiv. Crossref confirms the AIES proceedings, pages 180–186, the 7 Feb 2020 date and the DOI.
- **Key claim (paraphrased):** A scaffolding attack lets a deliberately biased classifier (e.g. a racist one on COMPAS) show innocuous LIME/SHAP explanations, because perturbation-based explainers can be detected and gamed.
- **Lens(es):** concerns
- **How the article can use it:** A reminder that a faithful narrative of SHAP is only as trustworthy as the SHAP itself. Note that the attack targets perturbation-based KernelSHAP/LIME. Exact TreeSHAP on your own model is a different threat model.

### [22] I. Elizabeth Kumar, Suresh Venkatasubramanian, Carlos Scheidegger, Sorelle Friedler (2020). Problems with Shapley-value-based explanations as feature importance measures. Proceedings of the 37th ICML, PMLR 119:5491–5500. <https://proceedings.mlr.press/v119/kumar20e.html>
- **Verified:** yes. Title, 4 authors and the volume/pages confirmed on the PMLR page.
- **Key claim (paraphrased):** Using Shapley values as feature importance raises mathematical problems. Fixing them needs extra machinery such as causal reasoning, and the explanations don't line up well with human-centred explainability goals.
- **Lens(es):** concerns
- **How the article can use it:** Scope the article's claim: the narratives are faithful to SHAP attributions, not causal accounts.

### [23] Chirag Agarwal, Sree Harsha Tanneru, Himabindu Lakkaraju (2024). Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models. arXiv preprint arXiv:2402.04614 (v1 7 Feb 2024; v3 14 Mar 2024). <https://arxiv.org/abs/2402.04614>
- **Verified:** yes. Title, 3 authors and dates confirmed on arXiv. No venue stated.
- **Key claim (paraphrased):** LLM explanations (chain-of-thought, token importance, counterfactuals) are often plausible, coherent to humans, without being faithful to the model's actual reasoning. In high-stakes domains, faithfulness has to come before user appeal.
- **Lens(es):** concerns / justification
- **How the article can use it:** Vocabulary for the core design choice. Plausibility comes from the glossary, faithfulness from SHAP plus mechanical checks.

### [24] Cynthia Rudin (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1(5):206–215. DOI 10.1038/s42256-019-0048-x. <https://www.nature.com/articles/s42256-019-0048-x>
- **Verified:** yes. Title, author, volume/issue/pages, date (13 May 2019) and DOI confirmed on PMC (PMC9122117).
- **Key claim (paraphrased):** Post hoc explanations of black boxes can be unfaithful or misleading. For high-stakes decisions, build inherently interpretable models instead.
- **Lens(es):** concerns
- **How the article can use it:** The strongest philosophical objection. Respond honestly: GBDT + TreeSHAP is a pragmatic middle ground, and the glossary makes the features themselves interpretable, in line with [7].

### Regulation and reporting standards

### [15] Consumer Financial Protection Bureau (2022). Consumer Financial Protection Circular 2022-03: Adverse action notification requirements in connection with credit decisions based on complex algorithms. Issued 26 May 2022; 87 FR 35864 (14 Jun 2022). <https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/>
- **Verified:** yes, on the consumerfinance.gov page.
- **Key claim (paraphrased):** ECOA/Reg B adverse-action rules apply whatever the technology.
  - Reasons must be specific, give the principal reason(s), and accurately describe the factors actually considered.
  - A creditor's inability to understand its own model is no defence.
  - Post hoc approximations are allowed only if the creditor can validate their accuracy.
- **Status:** **withdrawn 12 May 2025** (see [17]).
- **Lens(es):** justification / concerns
- **How the article can use it:** Motivation for "accurate principal reasons". Cite it with its withdrawn status (see [17]/[18]).

### [16] Consumer Financial Protection Bureau (2023). Consumer Financial Protection Circular 2023-03: Adverse action notification requirements and the proper use of the CFPB's sample forms provided in Regulation B. Issued 19 Sep 2023; 89 FR 27361 (17 Apr 2024). <https://files.consumerfinance.gov/f/documents/cfpb_adverse_action_notice_circular_2023-09.pdf>
- **Verified:** yes. The official CFPB PDF was read. The web page now shows "Content has been archived".
- **Key claim (paraphrased):** The question is whether creditors using AI or complex models can rely on the sample-form checklist when it doesn't reflect the real reasons. The answer is no. Checklist reasons count only if they specifically and accurately state the principal reasons, and overly broad or vague reasons that hide the real ones are also not acceptable. The press release gives an example: a limit cut driven by specific purchase behaviour needs that specificity, not a generic "purchasing history".
- **Status:** **withdrawn 12 May 2025** (see [17]).
- **Lens(es):** justification
- **How the article can use it:** The closest regulatory match to "cited reasons must match the actual SHAP drivers". Cite it as withdrawn guidance that interprets a rule still in force ([18]).

### [17] Consumer Financial Protection Bureau (2025). Interpretive Rules, Policy Statements, and Advisory Opinions; Withdrawal. Federal Register 90 (12 May 2025), FR Doc. 2025-08286. <https://www.govinfo.gov/content/pkg/FR-2025-05-12/html/2025-08286.htm>
- **Verified:** yes. The govinfo.gov Federal Register text lists both Circular 2022-03 (87 FR 35864) and Circular 2023-03 (89 FR 27361) among 67 withdrawn documents. The notice says the withdrawal is not necessarily final while the review is ongoing.
- **Lens(es):** concerns
- **How the article can use it:** **Must cite** if [15]/[16] are mentioned, so the article doesn't present withdrawn guidance as current. Phrase it as: "the CFPB said in 2022–23 (guidance withdrawn in 2025)…" and point to Reg B itself.

### [18] Consumer Financial Protection Bureau. Regulation B, 12 CFR 1002.9(b)(2): Statement of specific reasons, with Official Interpretation. <https://www.consumerfinance.gov/rules-policy/regulations/1002/9/>
- **Verified:** yes, on the consumerfinance.gov regulation page.
- **Key claim (paraphrased):** The reasons must be specific and give the principal reason(s). Saying the applicant failed internal standards or a credit score is not enough. Comment 9(b)(2)-4 says disclosed reasons must relate to factors actually scored, and no principal reason may be left out.
- **Lens(es):** justification
- **How the article can use it:** The durable legal anchor (a regulation, not guidance) for checking that narrated reasons equal the model's actual top drivers.

### [26] Financial Crimes Enforcement Network (2003). Guidance on Preparing a Complete & Sufficient Suspicious Activity Report Narrative. U.S. Department of the Treasury, November 2003. <https://www.fincen.gov/sites/default/files/shared/sarnarrcompletguidfinal_112003.pdf>
- **Verified:** yes. The official FinCEN PDF (33 pp., dated November 2003) was read.
- **Key claim (paraphrased):** A SAR narrative should answer who, what, when, where and why, and should also describe how (the method of operation). The narrative is the SAR's only free-text area, so it has to be clear, concise and thorough. Incomplete or disorganized narratives undermine the report's purpose.
- **Lens(es):** justification / scenarios
- **How the article can use it:** The output spec for the AML scenario. The glossary's "pattern it catches" field feeds the "why", and SHAP plus feature values feed the "what" and "how".

### Datasets

### [19] I-Cheng Yeh, Che-hui Lien (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications 36(2):2473–2480. DOI 10.1016/j.eswa.2007.12.020. <https://doi.org/10.1016/j.eswa.2007.12.020>
- **Verified:** yes. Crossref confirms the title, both authors, journal, volume/issue/pages and March 2009 print date. ScienceDirect blocked the fetch (403), so the abstract wasn't read.
- **Key claim (paraphrased):** The source paper for the Taiwan credit-card default dataset. It compares data-mining classifiers on predicting the probability of default.
- **Lens(es):** scenarios
- **How the article can use it:** Required citation for the credit scenario dataset.

### [20] I-Cheng Yeh (2009, donated 2016). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository, id 350. DOI 10.24432/C55S3H. <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>
- **Verified:** yes, on the UCI dataset page.
  - **License:** CC BY 4.0.
  - **Size:** 30,000 instances, 23 features, no missing values; donated 25 Jan 2016.
  - **Target:** binary default payment (1 = yes, 0 = no).
  - **Variables:**
    - X1 = amount of given credit (NT dollar), LIMIT_BAL
    - X2 = gender (1 male, 2 female)
    - X3 = education (1 graduate school, 2 university, 3 high school, 4 others)
    - X4 = marital status (1 married, 2 single, 3 others)
    - X5 = age
    - X6–X11 = repayment status, Sep 2005 back to Apr 2005
    - X12–X17 = bill statement amounts
    - X18–X23 = previous payment amounts
  - **Repayment coding, as documented:** −1 = pay duly; 1 = one-month delay; 2 = two-month delay; … 8 = eight months; 9 = nine months and above.
  - **Not documented:** the page never mentions −2 or 0, which do occur in the data.
  - **Column names:**
    - The variables table shows ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, then **PAY_0, PAY_2, PAY_3, …**. There is no PAY_1, so the sequence is PAY_0, PAY_2–PAY_6.
    - BILL_AMT1–6 and PAY_AMT1–6 are the data file's column names. The table view was cut off before those rows, so their exact spelling wasn't seen on the page.
- **Lens(es):** scenarios / concerns
- **How the article can use it:**
  - Credit-scenario data.
  - The undocumented −2/0 codes are a good real example of why a glossary entry has to state each code's meaning. If it doesn't, the LLM will guess.
  - Note the PAY_0/PAY_2 naming quirk.

### [27] Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Gianluca Bontempi (2015). Calibrating Probability with Undersampling for Unbalanced Classification. 2015 IEEE Symposium Series on Computational Intelligence (SSCI), pp. 159–166. DOI 10.1109/SSCI.2015.33. <https://doi.org/10.1109/SSCI.2015.33>
- **Verified:** yes. Crossref confirms the title, the 4 authors, SSCI 2015, pages and DOI. The IEEE Xplore page itself wasn't opened.
- **Key claim (paraphrased):** The paper behind the ULB credit-card fraud dataset. It covers probability calibration after undersampling in highly unbalanced classification.
- **Lens(es):** scenarios
- **How the article can use it:** Required citation when the ULB/Kaggle fraud data appears.

### [28] Machine Learning Group, ULB (2016–). Credit Card Fraud Detection [Dataset]. Kaggle, mlg-ulb/creditcardfraud. <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- **Verified:** yes, via the Kaggle dataset API metadata, since the HTML page renders client-side. The same description was cross-checked on OpenML dataset 1597.
  - **Size:** 284,807 transactions by European cardholders over two days in September 2013. 492 are frauds (0.172%).
  - **Features:** V1–V28 are principal components from PCA. Only Time and Amount are left untransformed. The original features are withheld for confidentiality.
  - **Label:** Class (1 = fraud).
  - **Evaluation advice:** the page recommends AUPRC.
  - **License:** ODbL (Database: Open Database; Contents: Database Contents).
- **Lens(es):** scenarios / gaps
- **How the article can use it:** The perfect counterexample. With anonymous PCA features, no glossary can be written and the narratives become meaningless. It also links to [4] on rotations.

### [29] Erik Altman, Jovan Blanuša, Luc von Niederhäusern, Béni Egressy, Andreea Anghel, Kubilay Atasu (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. NeurIPS 36, Datasets and Benchmarks Track. arXiv:2306.16424. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html>
- **Verified:** yes. Title, 6 authors and track confirmed on the NeurIPS page. The patterns were read in arXiv HTML v3.
- **Key claim (paraphrased):** AMLworld is an agent-based generator calibrated to look like real transactions. Synthetic data has complete ground-truth labels, unlike real data, where much laundering is never detected.
  - Eight planted laundering patterns: fan-out, fan-in, gather-scatter, scatter-gather, simple cycle, random, bipartite, stack.
  - Six datasets: HI/LI illicit ratios × Small/Medium/Large.
- **Lens(es):** scenarios / justification
- **How the article can use it:** Justifies synthetic AML data with planted typologies. You can check whether a narrative names the planted typology.

### Practitioner pieces

### [30] Sara Nobrega (2025). The End-to-End Data Scientist's Prompt Playbook. Towards Data Science, 8 Sep 2025. <https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/>
- **Verified:** yes. Title, author and date confirmed on the TDS page.
- **Key claim (paraphrased):** For SHAP narratives, give the LLM a structured table and ask for:
  - a top-5 driver ranking
  - a plain-language narrative
  - actions

  Define the audience, expand acronyms, flag unclear feature names, and add a self-evaluation step.
- **Lens(es):** scenarios / treatment
- **How the article can use it:** Shows current practitioner practice, a prompt-only recipe. Its advice to "flag unclear feature names" is a manual stand-in for a glossary.

### [31] Benjamin Nweke (2026). What SHAP Can't Explain About Agentic AI Fraud. Towards Data Science, 10 Sep 2026. <https://towardsdatascience.com/what-shap-cant-explain-about-agentic-ai-fraud/>
- **Verified:** yes. Title, author and date confirmed on the TDS page.
- **Key claim (paraphrased):** Feature-attribution fraud signals assume a human "pattern of life". Fraud carried out by autonomous agents needs trajectory audits of the agent's decisions, permissions and tool use, not better transaction-level attribution.
- **Lens(es):** concerns / gaps
- **How the article can use it:** Honest limits paragraph. The recipe explains transaction or account-level models; it isn't a tool for auditing agent behaviour.

### [32] Adam Zewe (2024). Enabling AI to explain its predictions in plain language. MIT News, December 2024. <https://news.mit.edu/2024/enabling-ai-explain-predictions-plain-language-1209>
- **Verified:** yes. Headline and author confirmed on news.mit.edu. The page date reads 10 Dec 2024, while the URL slug suggests 9 Dec, so cite "December 2024".
- **Key claim (paraphrased):** A news write-up of Explingo [6].
  - The researchers deliberately limit the LLM to the language part, to reduce the chance of introducing inaccuracies.
  - Even when an LLM makes a mistake doing a task, it often catches the mistake when checking that task.
  - Future work covers comparative language and interactive follow-up questions.
- **Lens(es):** justification / treatment
- **How the article can use it:** An accessible quote source for "LLM narrates, does not compute", and for the narrator-plus-checker split.

### UNVERIFIED / dropped
- **Hallucination taxonomy for tabular reasoning** (feature-existence, feature-value, comparison), from the recon pass: **UNVERIFIED.** Three targeted searches found no paper with that taxonomy. **Do not cite.** Use the rank/sign/value framework of [33]/[34] plus Zhang et al.'s [11] reported failure modes instead. Both are verified and cover the same checks.

---

## Synthesis

### (1) Comparison table: LLM-explanation approaches

| Work | LLM input | Feature semantics in prompt? | Larger/smaller-than-expected semantics? | Faithfulness check? | Domain |
|---|---|---|---|---|---|
| Zeng & Zhu 2024 [8] | (feature name, SHAP value) tuples; Mistral 7B | No | No | None (examples only) | Titanic, Adult (toy) |
| x-[plAIn], Mavrepis et al. 2024 [25] | Image of XAI plot + audience level; custom GPT | No | No | None; preference survey (>80%) | Mixed (vision, NLP, tabular) |
| SHAPstories, Martens et al. 2025 [5] | Dataset/target/feature intro + table of value & SHAP; GPT-4 | Dataset-level intro only | No | None automatic; surveys (93% convincing) | FIFA, student, German credit |
| Explingo, Zytek et al. 2024 [6] | (name, value, SHAP) top-N + format spec + one-line context + exemplars; GPT-4o | One-line context only | No (authors flag "larger" needs distribution info) | LLM Grader: accuracy incl. direction, completeness (96% human agreement, n=50) | Housing, student, mushroom, malware |
| Ichmoukhamedov et al. 2024 [33] | Name, SHAP, value, **training average**, **short description**, dataset description | Yes (short description) | Partial (average only, no risk meaning) | Extraction LLM → rank / sign / value accuracy (sign 0.96–0.99) | FIFA, German credit, student |
| He & Martens 2026 [34] | As [33] + predicted probability; Narrator + critic agents | Yes (short description) | Partial (average only) | Rule-based rank/sign/value check + retry (31% → 3% unfaithful) | 5 tabular incl. credit, health |
| ContextualSHAP, Dwiyanti et al. 2025 [35] | Aliases, descriptions, units/encodings, background, SHAP table + plots; GPT-4o | Yes (user-authored) | No | None; 7-user Likert study | Healthcare (HCV) |
| Zhang et al. 2026 [11] | Names, raw values, **p50/p90 benchmarks**, top-10 SHAP, **feature_meaning**, score percentile, sign convention, JSON with evidence refs; GPT-OSS-120B | Yes (meaning of each feature) | Partial (population percentiles, no per-feature risk direction) | Prompt constraint + blinded human ratings (factual accuracy 4.20/5) + periodic review; no programmatic check | AML money mules (production) |
| Geng et al. 2026 [36] | Controlled: names + ranked attributions. Autonomous: raw values + PD | Not mentioned | No | Top-K overlap, Kendall τ, schema checks vs SHAP | Consumer credit (LendingClub) |
| Kroeger et al. 2024 [10] | Placeholder-labelled feature values, perturbations, model outputs (or LIME exemplars) | No (placeholder labels) | No | FA/RA/PGI/PGU vs ground truth | Tabular + text |
| TalkToModel, Slack et al. 2023 [9] | User utterance only (LLM parses); template responses | Dataset "define terms" ops | No | Picks most-faithful explainer; parsing accuracy; human study | Tabular (health etc.) |
| Lukassen et al. 2026 [13] | Domain context, prediction, values, sorted SHAP, base value, metrics | Domain context only | No | Downstream utility with LLM judges: no accuracy gain; OOD flagging 30% → 15% | Energy forecasting |
| Co-Investigator AI, Naik et al. 2025 [12] | Transactions, risk indicators, typology labels, external intel, past SARs (multi-agent) | Typology knowledge | No | Agent-as-a-Judge + rule checks; 6 investigators (70% completeness) | AML SARs |
| FAX, Kim et al. 2026 [14] | Agentic; SHAP proposes claims, tools verify | n/a | n/a | Claim-level verification via model queries (simulation faithfulness 0.20 → 0.46) | RL + tabular |
| **This article** | Top-k SHAP (value, SHAP, rank) + **structured glossary**: what it is, pattern caught, motivation, larger-than-expected meaning, smaller-than-expected meaning, typical range; Claude Haiku 4.5 structured output | **Yes, expert-authored, per feature** | **Yes, per feature and risk-directional** | **Programmatic:** hallucinated features, direction vs SHAP sign, rank. No extraction LLM is needed because the output is structured. | Fraud, AML, credit, insurance, security |

### (2) Strongest contrarian evidence and how the glossary framework should respond

1. **Narratives raise trust, not accuracy** ([13] Lukassen 2026; [37] Marusich 2026).
   - Fluent narratives left decision accuracy unchanged and raised confidence and reliance. A placebo produced the same boost.
   - Out of distribution, narratives halved the rate of flagging unreliable predictions ([13]; LLM judges). In a human study, reliance rose for wrong AI advice too ([37]; GPT-4o self-explanations, not SHAP).
   - *Response:* don't claim the narratives improve decisions. Claim faithfulness and auditability, which can be measured: hallucinated-feature rate, direction-error rate, rank agreement.
   - Make the narrative surface uncertainty, e.g. "value is outside the glossary's typical range", rather than only reassure. Recommend a human-decision study as future work.
2. **Rich semantics can override the model** ([33] Ichmoukhamedov 2024).
   - When SHAP signs in the prompt were inverted, sign accuracy against the *original* signs was about 15–31%, against a naive about 4% if the LLM followed every flip (Table 3). So the LLMs followed most flips but "corrected" a surprisingly large share back to the plausible direction. *(Corrected 2026-10-18; earlier versions of these notes had this backwards.)*
   - A glossary that says "larger than expected means higher risk" is a strong prior. It could make the LLM state the glossary's direction even when SHAP's sign is the opposite, for example a high value that lowers risk through an interaction.
   - *Response:*
     - Always check direction against the SHAP sign, never against the glossary.
     - Present the glossary's larger/smaller fields as typical meaning, and require the LLM to flag cases that run "contrary to the usual pattern".
     - Include adversarial or counter-intuitive cases in the evaluation: a planted sign conflict, or a sign-flip ablation like [33]'s. Report the direction-error rate with and without the glossary.
     - This is the experiment that would make the article original.
3. **The attribution itself may be wrong** ([24] Rudin 2019; [21] Slack 2020; [22] Kumar 2020; [23] Agarwal 2024).
   - *Response:* scope the claim to "faithful to TreeSHAP attributions of the deployed model". Exact TreeSHAP on your own model avoids the surrogate and perturbation attacks in [21]. It does not give causal claims [22].
   - The glossary addresses Rudin's interpretability concern at the feature level [7], not the model level.
4. **Feature semantics already exist in prior work** ([11], [33], [34], [35]).
   - *Response:* acknowledge it. The novelty is not "add descriptions". It is three things together:
     - (a) risk-directional larger/smaller-than-expected semantics plus the fraud or risk pattern each feature was engineered to catch, written by a domain expert;
     - (b) structured output checked programmatically against SHAP;
     - (c) the same recipe applied across five risk domains.
   - [11]'s own reported errors (10x magnitude, misread ratio direction) are exactly what (a) and (b) target.
5. **Regulatory framing is in flux** ([15]–[18]).
   - The CFPB AI adverse-action circulars were withdrawn in May 2025.
   - *Response:* anchor on Reg B §1002.9(b)(2) itself (specific, accurate principal reasons) and mention the circulars as withdrawn interpretive guidance. FinCEN's who/what/when/where/why/how standard [26] is still the AML anchor.

### (3) The gap this article fills

Existing LLM explainers feed the model SHAP values plus, at most, a short feature description and a population average or percentile ([5], [6], [11], [33], [35]). None tells the LLM, per feature, what a larger-than-expected or smaller-than-expected value means for risk and which fraud or risk pattern the feature was engineered to catch. Even the closest production system [11] still relies on prompt constraints and human review rather than programmatic checks. This article captures that domain knowledge in a structured feature glossary. It then forces the LLM's cited reasons into a schema that is checked automatically against TreeSHAP for hallucinated features, direction and rank. The article tests the recipe across fraud, AML, credit, insurance and security alerts, and reports how often the narrative contradicts the model.

### (4) Reference list (TDS format)

[1] S. M. Lundberg and S.-I. Lee, A Unified Approach to Interpreting Model Predictions (2017), Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

[2] S. M. Lundberg, G. Erion, H. Chen et al., From local explanations to global understanding with explainable AI for trees (2020), Nature Machine Intelligence. https://doi.org/10.1038/s42256-019-0138-9

[3] G. Ke, Q. Meng, T. Finley et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree (2017), Advances in Neural Information Processing Systems 30. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

[4] L. Grinsztajn, E. Oyallon and G. Varoquaux, Why do tree-based models still outperform deep learning on typical tabular data? (2022), NeurIPS Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

[5] D. Martens, J. Hinns, C. Dams, M. Vergouwen and T. Evgeniou, Tell Me a Story! Narrative-Driven XAI with Large Language Models (2025), Decision Support Systems. https://doi.org/10.1016/j.dss.2025.114402

[6] A. Zytek, S. Pido, S. Alnegheimish, L. Berti-Équille and K. Veeramachaneni, Explingo: Explaining AI Predictions using Large Language Models (2024), IEEE International Conference on Big Data. https://arxiv.org/abs/2412.05145

[7] A. Zytek, I. Arnaldo, D. Liu, L. Berti-Equille and K. Veeramachaneni, The Need for Interpretable Features: Motivation and Taxonomy (2022), ACM SIGKDD Explorations. https://arxiv.org/abs/2202.11748

[8] X. Zeng and K. Zhu, Enhancing the Interpretability of SHAP Values Using Large Language Models (2024), arXiv:2409.00079. https://arxiv.org/abs/2409.00079

[9] D. Slack, S. Krishna, H. Lakkaraju and S. Singh, Explaining machine learning models with interactive natural language conversations using TalkToModel (2023), Nature Machine Intelligence. https://doi.org/10.1038/s42256-023-00692-8

[10] N. Kroeger, D. Ley, S. Krishna, C. Agarwal and H. Lakkaraju, In-Context Explainers: Harnessing LLMs for Explaining Black Box Models (2024), arXiv:2310.05797. https://arxiv.org/abs/2310.05797

[11] Y. Zhang, Y. Zhang, Y. Jin et al., Detection, Attribution, Narration: An End-to-End Pipeline for Explainable Money Mule Identification (2026), arXiv:2607.17586. https://arxiv.org/abs/2607.17586

[12] P. V. Naik, N. K. Dintakurthi, Z. Hu, Y. Wang and R. Qiu, Co-Investigator AI: The Rise of Agentic AI for Smarter, Trustworthy AML Compliance Narratives (2025), arXiv:2509.08380. https://arxiv.org/abs/2509.08380

[13] F. Lukassen, J. Herrmann, C. Weisser et al., Quality Without Usefulness: LLM-Generated XAI Narratives as Trust Heuristics Rather Than Decision Aids (2026), arXiv:2605.26770. https://arxiv.org/abs/2605.26770

[14] J. Kim, S. Mun, S. Lee, J. Cho and J. Ok, Towards Faithful Agentic XAI: A Verification Method and an Open-World Benchmark for Better Model Faithfulness (2026), arXiv:2605.27879. https://arxiv.org/abs/2605.27879

[15] Consumer Financial Protection Bureau, Circular 2022-03: Adverse action notification requirements in connection with credit decisions based on complex algorithms (2022; withdrawn 2025), consumerfinance.gov. https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/

[16] Consumer Financial Protection Bureau, Circular 2023-03: Adverse action notification requirements and the proper use of the CFPB's sample forms provided in Regulation B (2023; withdrawn 2025), consumerfinance.gov. https://files.consumerfinance.gov/f/documents/cfpb_adverse_action_notice_circular_2023-09.pdf

[17] Consumer Financial Protection Bureau, Interpretive Rules, Policy Statements, and Advisory Opinions; Withdrawal (2025), Federal Register 90 FR, FR Doc. 2025-08286. https://www.govinfo.gov/content/pkg/FR-2025-05-12/html/2025-08286.htm

[18] Consumer Financial Protection Bureau, Regulation B §1002.9 Notifications, (b)(2) Statement of specific reasons (current), consumerfinance.gov. https://www.consumerfinance.gov/rules-policy/regulations/1002/9/

[19] I.-C. Yeh and C.-h. Lien, The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients (2009), Expert Systems with Applications. https://doi.org/10.1016/j.eswa.2007.12.020

[20] I.-C. Yeh, Default of Credit Card Clients (2016), UCI Machine Learning Repository (CC BY 4.0). https://doi.org/10.24432/C55S3H

[21] D. Slack, S. Hilgard, E. Jia, S. Singh and H. Lakkaraju, Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods (2020), AAAI/ACM Conference on AI, Ethics, and Society. https://doi.org/10.1145/3375627.3375830

[22] I. E. Kumar, S. Venkatasubramanian, C. Scheidegger and S. Friedler, Problems with Shapley-value-based explanations as feature importance measures (2020), ICML (PMLR 119). https://proceedings.mlr.press/v119/kumar20e.html

[23] C. Agarwal, S. H. Tanneru and H. Lakkaraju, Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models (2024), arXiv:2402.04614. https://arxiv.org/abs/2402.04614

[24] C. Rudin, Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead (2019), Nature Machine Intelligence. https://doi.org/10.1038/s42256-019-0048-x

[25] P. Mavrepis, G. Makridis, G. Fatouros et al., XAI for All: Can Large Language Models Simplify Explainable AI? (2024), arXiv:2401.13110. https://arxiv.org/abs/2401.13110

[26] Financial Crimes Enforcement Network, Guidance on Preparing a Complete & Sufficient Suspicious Activity Report Narrative (2003), U.S. Department of the Treasury. https://www.fincen.gov/sites/default/files/shared/sarnarrcompletguidfinal_112003.pdf

[27] A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, Calibrating Probability with Undersampling for Unbalanced Classification (2015), IEEE Symposium Series on Computational Intelligence. https://doi.org/10.1109/SSCI.2015.33

[28] Machine Learning Group, ULB, Credit Card Fraud Detection (2016), Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

[29] E. Altman, J. Blanuša, L. von Niederhäusern et al., Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (2023), NeurIPS Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f38404edff6f3f642d6fa5892479c42-Abstract-Datasets_and_Benchmarks.html

[30] S. Nobrega, The End-to-End Data Scientist's Prompt Playbook (2025), Towards Data Science. https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/

[31] B. Nweke, What SHAP Can't Explain About Agentic AI Fraud (2026), Towards Data Science. https://towardsdatascience.com/what-shap-cant-explain-about-agentic-ai-fraud/

[32] A. Zewe, Enabling AI to explain its predictions in plain language (2024), MIT News. https://news.mit.edu/2024/enabling-ai-explain-predictions-plain-language-1209

[33] T. Ichmoukhamedov, J. Hinns and D. Martens, How good is my story? Towards quantitative metrics for evaluating LLM-generated XAI narratives (2024), arXiv:2412.10220. https://arxiv.org/abs/2412.10220

[34] Y. He and D. Martens, An Agentic Approach to Generating XAI-Narratives (2026), arXiv:2603.20003. https://arxiv.org/abs/2603.20003

[35] L. Dwiyanti, S. R. Wibisono and H. Nambo, ContextualSHAP: Enhancing SHAP Explanations Through Contextual Language Generation (2025), World Symposium on Software Engineering (WSSE '25). https://arxiv.org/abs/2512.07178

[36] W. Geng, D. Liu, L. Li and Y. Wang, Evaluating Large Language Models as Post Hoc Explainability Interfaces for Credit Risk Models (2026), Expert Systems. https://doi.org/10.1111/exsy.70351

[37] L. R. Marusich, M. G. Kozuch Dhooghe, J. Z. Bakdash and M. Kantarcioglu, Human Decision-Making with Persuasive and Narrative LLM Explanations (2026), arXiv:2605.23867. https://arxiv.org/abs/2605.23867
