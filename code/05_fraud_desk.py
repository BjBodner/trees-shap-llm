# file: code/05_fraud_desk.py
# Desk 1: card fraud. One case per caught fraud event: every event the model alerts on,
# shown by its best-scoring alert and graded against the pattern we planted.
# Then 5 versions of the evidence on the same cases.
G_FRAUD = glossary("fraud")                        # 43 rows, one per expert feature
rng = np.random.default_rng(SEEDS["sample"])

def make_cases(run, X_train, truth, prefix, rows):
    pct = percentiles(X_train)                      # where a value sits among normal ones
    mu, sd = X_train.mean(), X_train.std().replace(0, 1)
    sv = run["explainer"].shap_values(run["X"].iloc[rows])
    cases = {}
    for i, s in zip(rows, sv):
        vals = run["X"].iloc[i]
        cases[i] = dict(case_id=f"{prefix}{i}", truth=truth[i], score=run["p"][i],
                        score_line=f"Model score: {run['p'][i]:.3f} (alert threshold "
                                   f"{run['thr']:.3f}). Decision: ALERT",
                        values=vals, zvalues=(vals - mu) / sd, shap=pd.Series(s, vals.index),
                        pct=pd.Series({c: pct(c, v) for c, v in vals.items()}))
    return cases

ex = RUNS["expert"]
alerted = pd.DataFrame({"i": np.arange(len(test_tx)), "p": ex["p"],
                        "event": test_tx.event_id, "label": test_tx.label})
alerted = alerted[(alerted.p >= ex["thr"]) & (alerted.event >= 0)]
top = alerted.sort_values("p", ascending=False).groupby("event").head(1)   # best alert per event
CASES = make_cases(ex, X_expert[split == "train"], test_tx.label, "F", np.sort(top.i.to_numpy()))
print(f"{len(CASES)} caught fraud events:", test_tx.label[list(CASES)].value_counts().to_dict())

# A 200-case sample of those (50 per pattern) is kept for the hidden-names test in block 7.
S_rows = np.sort(np.concatenate([rng.choice(g.i.to_numpy(), min(50, len(g)), replace=False)
                                 for _, g in top.groupby("label")]))
hook = min((i for i in S_rows if round(ex["p"][i], 2) == 0.97),    # the case in the hero image:
           key=lambda i: test_tx.t[i])                             # earliest alert scored 0.97
hc = CASES[hook]
hide = dict(zip(rng.permutation(X_expert.columns), [f"V{j:02d}" for j in range(1, 44)]))
ANON = dict(hide=hide)                             # names hidden: V01 ... V43, values standardized

CHANNEL = ["is_card_present", "is_online"]         # context a human needs: in store or online?
CONDS = {**LADDER,                                 # names | + percentiles | + glossary (top 10)
         "names+pct|top_10+context": dict(LADDER["names+pct"], pin=CHANNEL),
         "top_10+context": dict(LADDER["glossary+pct"], pin=CHANNEL)}
SCHEMA_F = alert_schema(TYPOLOGIES + ["unclear"])
FRAUD = pd.concat([run("fraud", CASES.values(), name, cond, G_FRAUD, SCHEMA_F, "card-fraud")
                   for name, cond in CONDS.items()], ignore_index=True)
pd.set_option("display.width", 200)
print(summarize(FRAUD).to_string())
said = FRAUD[(FRAUD.cond == "glossary+pct") & (FRAUD.case_id == hc["case_id"])].iloc[0]
a = said.answer                                    # the hook case, glossary + percentiles
print(f"\nhook case {hc['case_id']}: score {hc['score']:.2f}, truth {hc['truth']}. "
      f"{said.model} ({said.date}) answered {a['typology']}, because:")
for r in a["reasons"]:
    print(f"  {r['feature']}: {r['effect']}, value vs normal: {r['value_vs_normal']}")
for who in ["analyst_rationale", "customer_message"]:
    print(textwrap.fill(f"{who}: {a[who]}", 100, subsequent_indent="  "))
