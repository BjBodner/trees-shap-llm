# file: code/10_extras.py
# Repo only. Every other pre-registered fraud condition, which the article sums up in a
# line each: feature-store abbreviations, the glossary without percentiles, SHAP lists of
# 3 / 5 / grouped by family, shuffled SHAP, no SHAP, the same prompt asked twice,
# abbreviations + glossary; the AML "glossary" rung; the low-value trap (E4); and two
# more simulated fraud worlds for the PR-AUC spread.
# It replays them with the full narrator from the article's first version (the *_full
# functions below). For every condition the article also runs, they write the same
# prompt byte for byte, so all of it replays from the same cache.
from tsl import *                                  # the repo scripts use the rest of tsl.py
LADDER_FULL = {
    "anonymized":   dict(names="anon", pct=False, gloss=False),
    "legacy_names": dict(names="legacy", pct=False, gloss=False),
    "names":        dict(names="readable", pct=False, gloss=False),
    "names+pct":    dict(names="readable", pct=True, gloss=False),
    "glossary":     dict(names="readable", pct=False, gloss=True),
    "glossary+pct": dict(names="readable", pct=True, gloss=True),
}

def system_prompt_full(desk, label, cond):
    t = TYPO[TYPO.desk == desk]
    shap_rule = ("You do NOT get SHAP values; judge each feature's effect yourself."
                 if cond.get("shap") == "none" else
                 "Each feature comes with its SHAP value: positive pushed the score toward "
                 "'suspicious', negative pushed it away (units: log-odds). Follow the SHAP "
                 "sign when you state a feature's effect.")
    anon = ("\nFeature names are anonymized and values are standardized (mean 0, sd 1)."
            if cond["names"] == "anon" else "")
    return (f"You are a senior {label} analyst. A gradient-boosted tree model scored a case "
            f"and you must explain its decision.\n{shap_rule}{anon}\n"
            "Rules:\n1. Use only the evidence given. Refer to features by the exact names shown.\n"
            "2. Choose the typology that best explains the evidence, or 'unclear'.\n"
            "3. List the 2 to 4 reasons that matter most. For each: its effect (raises_risk or "
            "lowers_risk) and whether its value is higher, lower or typical compared with "
            "normal (unknown if you cannot tell).\n"
            "4. analyst_rationale: 3 to 5 plain-English sentences for the investigator.\n"
            "5. customer_message: 1 to 2 plain-English sentences for the customer, at most "
            "35 words; no jargon, no feature names, no scores.\nTypologies:\n"
            + "\n".join(f"- {r.typology}: {r.definition}" for r in t.itertuples()))

def render_case_full(case, cond, gl, anon_names, k=10):
    sv, vals, pcts = case["shap"], case["values"], case["pct"]
    if cond.get("shap") == "permuted":                 # cross-examination: scrambled SHAP
        rng_p = np.random.default_rng(int(case["case_id"][1:]))
        sv = pd.Series(rng_p.permutation(sv.to_numpy()), index=sv.index)
    order = sv.abs().sort_values(ascending=False).index
    if k == "grouped":                                 # family totals + top-2 per family
        fam = gl.family.reindex(sv.index)
        tot = sv.groupby(fam).sum().sort_values(key=abs, ascending=False)
        keep = [f for g in tot.index for f in sv[fam == g].abs().nlargest(2).index]
    else:
        keep = list(order if k == "all" else order[:k])
    keep += [f for f in cond.get("pin", []) if f not in keep]
    name = (anon_names if cond["names"] == "anon" else gl.legacy_name.to_dict()
            if cond["names"] == "legacy" else {f: f for f in sv.index})
    shown = [dict(name=name[f], feature=f, value=vals[f], shap=sv[f], pct=pcts[f],
                  vcat=vcat(pcts[f])) for f in keep]
    head = "feature | value" + (" | SHAP" if cond.get("shap") != "none" else "") + (
        " | percentile" if cond["pct"] else "")
    lines = []
    for s in shown:
        v = case["zvalues"][s["feature"]] if cond["names"] == "anon" else s["value"]
        row = f"{s['name']} | {fmt(v)}"
        row += f" | {s['shap']:+.2f}" if cond.get("shap") != "none" else ""
        row += f" | {ordinal(s['pct'])}" if cond["pct"] and not pd.isna(s["pct"]) else (
            " | n/a" if cond["pct"] else "")
        lines.append(row)
    if k == "grouped":
        lines.insert(0, "family SHAP totals: " + ", ".join(f"{g} {v:+.2f}" for g, v in tot.items()))
    msg = (f"Case {case['case_id']}\n{case['score_line']}\n\n"
           f"Features ({'all' if k == 'all' else 'largest |SHAP| first'}):\n{head}\n"
           + "\n".join(lines))
    if cond["gloss"]:
        msg += "\n\nGlossary for these features:\n" + "\n".join(
            f"- {s['name']}: {gl.at[s['feature'], 'what']}. Catches: {gl.at[s['feature'], 'catches']}. "
            f"Why: {gl.at[s['feature'], 'motivation']}. If higher than expected: "
            f"{gl.at[s['feature'], 'if_higher']}. If lower: {gl.at[s['feature'], 'if_lower']}. "
            f"Typical: {gl.at[s['feature'], 'typical']}." for s in shown)
    return msg, shown

def grade_full(case, shown, ans, shap_given=True):
    by = {s["name"]: s for s in shown}
    rs = ans["reasons"] if ans else []
    known = [r for r in rs if r["feature"] in by]
    risers = sorted([s for s in shown if s["shap"] > 0], key=lambda s: -s["shap"])[:5]
    top5 = {s["name"] for s in risers}
    judged = [r for r in known if r["value_vs_normal"] != "unknown"
              and by[r["feature"]]["vcat"] is not None]
    up = [r for r in known if r["effect"] == "raises_risk"]
    lows = [r for r in judged if by[r["feature"]]["vcat"] == "lower"]
    return dict(
        trap=any(s["vcat"] == "lower" for s in risers[:3]),     # a LOW value raised risk
        low_value_error=np.mean([r["value_vs_normal"] != "lower" for r in lows]) if lows else np.nan,
        case_id=case["case_id"], truth=case["truth"], said=ans["typology"] if ans else None,
        failed=ans is None,
        correct=bool(ans) and ans["typology"] == case["truth"],
        unclear=bool(ans) and ans["typology"] == "unclear",
        hallucinated=any(r["feature"] not in by for r in rs),
        direction_error=shap_given and any((r["effect"] == "raises_risk") != (by[r["feature"]]["shap"] > 0)
                                           for r in known),
        value_error=np.mean([r["value_vs_normal"] != by[r["feature"]]["vcat"] for r in judged])
                    if judged else np.nan,
        faithful=np.mean([r["feature"] in top5 for r in up]) if up else np.nan,
        n_reasons=len(rs))

def run_full(desk, cases, cond_name, cond, gl, schema, label, k=10, anon_names=None):
    system = system_prompt_full(desk, label, cond)
    def one(case):
        user, shown = render_case_full(case, cond, gl, anon_names, k)
        rec = ask(f"{desk}|{cond_name}|{case['case_id']}", system, user, schema)
        return {**grade_full(case, shown, rec["answer"], cond.get("shap") != "none"),
                "cond": cond_name, "inp": rec["inp"], "out": rec["out"],
                "answer": rec["answer"], "model": rec["model"], "date": rec["date"]}
    return pd.DataFrame([one(c) for c in cases])

# the article's prompts and these must agree (checked on every case of every shared condition)
for name, cond in CONDS.items():
    full_cond = dict(LADDER_FULL["anonymized" if name == "anonymized" else
                                 "glossary+pct" if cond.get("gloss") else
                                 "names+pct" if cond.get("pct") else "names"])
    full_cond.update({k_: v for k_, v in cond.items() if k_ == "pin"})
    assert system_prompt_full("fraud", "card-fraud", full_cond) == system_prompt("fraud", "card-fraud", cond)
    for c in CASES.values():
        assert (render_case_full(c, full_cond, G_FRAUD, hide, cond.get("k", 10))[0]
                == render_case(c, cond, G_FRAUD)[0]), (name, c["case_id"])
print("article prompts == full-narrator prompts on all", len(CONDS), "shared conditions")

cases = list(CASES.values())
full = LADDER_FULL["glossary+pct"]
npc = LADDER_FULL["names+pct"]
FRAUD_FULL = pd.concat(
    [run_full("fraud", cases, n, c, G_FRAUD, SCHEMA_F, "card-fraud", anon_names=hide)
     for n, c in LADDER_FULL.items()]
    + [run_full("fraud", cases, f"top_{k}", full, G_FRAUD, SCHEMA_F, "card-fraud", k=k)
       for k in [3, 5, "all", "grouped"]]
    + [run_full("fraud", cases, "top_10+context", dict(full, pin=CHANNEL), G_FRAUD, SCHEMA_F, "card-fraud"),
       run_full("fraud", cases, "permuted_shap", dict(full, shap="permuted"), G_FRAUD, SCHEMA_F, "card-fraud"),
       run_full("fraud", cases, "no_shap", dict(full, shap="none"), G_FRAUD, SCHEMA_F, "card-fraud", k="all"),
       run_full("fraud", cases[:50], "repeat", full, G_FRAUD, SCHEMA_F, "card-fraud"),
       run_full("fraud", cases, "names+pct|top_10+context", dict(npc, pin=CHANNEL), G_FRAUD, SCHEMA_F, "card-fraud"),
       run_full("fraud", cases, "names+pct|top_all", npc, G_FRAUD, SCHEMA_F, "card-fraud", k="all"),
       run_full("fraud", cases, "legacy+glossary+pct", dict(names="legacy", pct=True, gloss=True),
                G_FRAUD, SCHEMA_F, "card-fraud")], ignore_index=True)
print(summarize_full(FRAUD_FULL).to_string())
rep = FRAUD_FULL[FRAUD_FULL.cond == "repeat"].merge(FRAUD_FULL[FRAUD_FULL.cond == "glossary+pct"], on="case_id")
print(f"\nsame prompt asked twice (default sampling): typology agrees on "
      f"{(rep.said_x == rep.said_y).mean():.0%} of {len(rep)} cases")

fcf = FRAUD_FULL[FRAUD_FULL.truth == "counterfeit_card"]
print("\nwhat the LLM answered on the 50 counterfeit cases:")
print(fcf.pivot_table(index="cond", columns="said", values="case_id", aggfunc="count", sort=False)
      .fillna(0).astype(int).to_string())
print("\nhow long a SHAP list? (glossary + percentiles)")
for k in ["top_3", "top_5", "glossary+pct", "top_10+context", "top_all"]:
    g = FRAUD_FULL[FRAUD_FULL.cond == k]
    print(f"{k:15s} all cases {g.correct.mean():.0%} | counterfeit "
          f"{g[g.truth == 'counterfeit_card'].correct.mean():.0%} | "
          f"${(0.10 * g.inp.mean() + 0.50 * g.out.mean()) / 1000:.3f} per 1,000")
ffa = FRAUD_FULL[(FRAUD_FULL.said == "account_takeover") & (FRAUD_FULL.truth != "account_takeover")]
print("\nfalse account-takeover calls by condition:", ffa.groupby("cond", sort=False).size().to_dict())

AML_FULL = pd.concat([run_full("aml", AML_CASES.values(), c, LADDER_FULL[c], G_AML, SCHEMA_A,
                               "anti-money-laundering")
                      for c in ["names", "names+pct", "glossary", "glossary+pct"]], ignore_index=True)
print("\nAML, four rungs:")
print(summarize_full(AML_FULL).to_string())

LAD4 = pd.concat([FRAUD_FULL, AML_FULL], ignore_index=True)
LAD4 = LAD4[LAD4.cond.isin(["names", "names+pct", "glossary", "glossary+pct"])]
print("\nE4: does a LOW value that raises risk trip the narrator? (fraud + AML)")
print(LAD4.groupby(["cond", "trap"], sort=False).agg(
    cases=("correct", "size"), correct=("correct", "mean"), low_value_error=("low_value_error", "mean"))
    .map(lambda v: f"{v:.0%}" if isinstance(v, float) else v).to_string())

# Is the PR-AUC gap stable? Two more simulated fraud worlds (models only, no LLM).
AP = {k: [RUNS[k]["ap"]] for k in RUNS}
for rep_i in [1, 2]:
    seed = int(np.random.SeedSequence([SEEDS["fraud"], rep_i]).generate_state(1)[0])
    tx_r, _, cust_r = simulate_fraud(seed, n_cust=5000)
    sp_r = np.select([tx_r.t < 30 * DAY, tx_r.t < 75 * DAY, tx_r.t < 85 * DAY],
                     ["warmup", "train", "val"], "test")
    med_r = tx_r[tx_r.t < 85 * DAY].groupby("mcc").amount.median().reindex(range(10)).to_numpy()
    ex_r, ki_r = build_features(tx_r, cust_r, med_r)
    fill_r = ex_r[sp_r == "train"].median()
    pca_r = make_pipeline(StandardScaler(), PCA(n_components=43, random_state=0))
    pca_r.fit(ex_r[sp_r == "train"].fillna(fill_r))
    em_r = pd.DataFrame(pca_r.transform(ex_r.fillna(fill_r)), columns=X_embed.columns)
    for k, X in {"embed": em_r, "kitchen": ki_r, "expert": ex_r}.items():
        AP[k].append(train(X, tx_r.is_fraud.to_numpy(), sp_r, seed)["ap"])
for k, v in AP.items():
    print(f"{k:8s} PR-AUC over {len(v)} simulated worlds: {np.mean(v):.3f} +/- {np.std(v):.3f}  "
          f"({', '.join(f'{a:.3f}' for a in v)})")
