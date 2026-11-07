# file: code/12_aml_desk.py
# Desk 2: AML. Monthly account summaries for 6 legit customer segments and 4 planted
# laundering typologies (the spec is in tsl.py). Same recipe, new glossary.
X_aml, lab_aml, seg_aml, split_aml = simulate_aml(SEEDS["aml"])
print(f"AML accounts: {len(X_aml):,} | features: {X_aml.shape[1]} | segments: "
      f"{len(AML_SEGMENTS)} | typologies: {len(AML_TYPOLOGIES)}")

def desk(name, X, labels, split_, seed, alert_rate, prefix, n_per=50):
    """Train, alert, and take up to n_per caught cases per typology."""
    r = train(X, (labels != "legit").astype(int), split_, seed, alert_rate)
    # rng continues block 5's random stream: run the blocks once, in order
    lab_te = labels[split_ == "test"]
    tp = np.where((r["p"] >= r["thr"]) & (lab_te != "legit"))[0]
    rows = np.sort(np.concatenate([rng.choice(tp[lab_te[tp] == t], min(n_per, (lab_te[tp] == t).sum()),
                                              replace=False) for t in np.unique(lab_te[tp])]))
    print(f"[{name}] PR-AUC {r['ap']:.3f} | caught in test: "
          f"{pd.Series(lab_te[tp]).value_counts().to_dict()}")
    return r, make_cases(r, X[split_ == "train"], lab_te, prefix, rows)

AML_RUN, AML_CASES = desk("aml", X_aml, lab_aml, split_aml, SEEDS["aml"], 0.08, "A")
G_AML, SCHEMA_A = glossary("aml"), alert_schema(list(AML_TYPOLOGIES) + ["unclear"])
AML = pd.concat([run("aml", AML_CASES.values(), c, LADDER[c], G_AML, SCHEMA_A,
                     "anti-money-laundering") for c in LADDER], ignore_index=True)
print(summarize(AML).to_string())
print(AML.pivot_table(index="cond", columns="truth", values="correct", sort=False)
      .map("{:.0%}".format).to_string())
for base in ["names+pct", "names"]:                 # paired over the same 200 cases
    d, (lo, hi) = paired_gain(AML, "glossary+pct", base)
    print(f"glossary+pct minus {base}: {d * 100:+.1f} pts [{lo * 100:+.1f}, {hi * 100:+.1f}]")
print("LLM calls this session:", STATS["api_calls"], "| replayed:", STATS["replayed"],
      "| prompt drift:", STATS["prompt_drift"])
