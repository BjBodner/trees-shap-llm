# file: code/09_better_context.py
# Better "what does a big or small value mean?" context, built from TRAIN rows only (Context, in tsl.py).
#   rarity    : how common is this value among normal transactions?
#   fraud rate: how many times more often are rows like this fraud than the average row?
#   past fraud: which kind of fraud did past fraud with such values turn out to be? (uses the known answers)
X_tr, lab_tr = X_expert[split == "train"], tx.label[split == "train"]
CTX = Context(X_tr, lab_tr)
CTX.eq = Context(X_tr, lab_tr, balanced=True)        # the same, but every kind of fraud counts equally

CONTEXT_CONDS = {                                   # name -> what the narrator sees (top-10 SHAP list)
    "A1 names+tail": dict(),                                                  # rarity instead of percentile
    "A2 names+tail+lift": dict(rate=True),                                    # + fraud rate
    "A3 names+tail+lift+mix": dict(rate=True, mix=True),                      # + past fraud (per row)
    "A5 names+tail+lift|chan": dict(rate=True, pin=CHANNEL),                  # + the two channel flags
    "A8 names+tail+lift+balmix": dict(rate=True, mix=True, equal=True),       # past fraud, kinds weighed equally
    "A9 names+tail+lift+balmix|chan": dict(rate=True, mix=True, equal=True, pin=CHANNEL)}
CTXRES = pd.concat([run("fraud", CASES.values(), n, c, G_FRAUD, SCHEMA_F, "card-fraud",
                        render=CTX.render, key="card-fraud-v6") for n, c in CONTEXT_CONDS.items()],
                   ignore_index=True)
ALL = pd.concat([FRAUD, CTXRES], ignore_index=True)
show = ["names", "names+pct", "glossary+pct", "names+pct|top_10+context", "top_10+context", "A1 names+tail",
        "A2 names+tail+lift", "A3 names+tail+lift+mix", "A8 names+tail+lift+balmix",
        "A5 names+tail+lift|chan", "A9 names+tail+lift+balmix|chan"]
print(summarize(ALL[ALL.cond.isin(show)]).loc[show].to_string())
print("\nwhat the narrator sees for the 0.97 case (rarity version, first rows):")
print("\n".join(CTX.render(hc, CONTEXT_CONDS["A1 names+tail"], G_FRAUD)[0].splitlines()[:9]))

print("\npaired differences (points, 95% interval):")
for label, a, b in [("rarity text instead of percentile", "A1 names+tail", "names+pct"),
                    ("rarity text vs names only", "A1 names+tail", "names"),
                    ("+ fraud rate", "A2 names+tail+lift", "A1 names+tail"),
                    ("+ past fraud, counted per row", "A3 names+tail+lift+mix", "A2 names+tail+lift"),
                    ("+ past fraud, kinds equal", "A8 names+tail+lift+balmix", "A2 names+tail+lift"),
                    ("channel pinned, rarity + rate", "A5 names+tail+lift|chan", "names+pct|top_10+context"),
                    ("+ past fraud, equal, pinned", "A9 names+tail+lift+balmix|chan", "A5 names+tail+lift|chan")]:
    d, (lo, hi) = paired_gain(ALL, a, b)
    print(f"{label:34s} {d * 100:+5.1f} [{lo * 100:+5.1f}, {hi * 100:+5.1f}]")
pick = ["A2 names+tail+lift", "A3 names+tail+lift+mix", "A8 names+tail+lift+balmix"]
print("\ncorrect pattern, by true fraud pattern:")
print(ALL[ALL.cond.isin(pick)].pivot_table(index="cond", columns="truth", values="correct", sort=False)
      .loc[pick].map("{:.0%}".format).to_string())
print("LLM calls this session:", STATS["api_calls"], "| replayed:", STATS["replayed"],
      "| prompt drift:", STATS["prompt_drift"])
plot_what_it_sees(ALL, len(CASES))                  # figure 3
