# file: code/10_better_features.py
# 19 features added after reading the model's mistakes: for each fraud pattern, the tell the
# first 43 features missed (features_v6, in tsl.py). Each uses only the past and the present transaction.
X_v6 = pd.concat([X_expert, features_v6(tx, cust, X_expert, X_kitchen)], axis=1)
G_V6 = glossary("fraud_v6")                         # the 43 old glossary rows + 19 new ones
print(f"{X_expert.shape[1]} features -> {X_v6.shape[1]} features")

# one of the 19, written out by hand: tiny online charges in the last 6 hours
order_key = tx.cust.to_numpy() * 1e7 + tx.t.to_numpy()            # rows are sorted by (customer, time)
window_start = np.searchsorted(order_key, order_key - 360, "right")   # first row inside the 6-hour window
tiny_so_far = np.r_[0, np.cumsum(((tx.amount < 5) & tx.online).to_numpy())]
by_hand = tiny_so_far[np.arange(len(tx)) + 1] - tiny_so_far[window_start]
print("by hand, n_tiny_online_6h matches features_v6:", np.allclose(by_hand, X_v6["n_tiny_online_6h"]))

# the same recipe (block 3), a slower learning rate (0.03, chosen on cross-validation)
y = tx.is_fraud.to_numpy()
RUN6 = train(X_v6, y, split, SEEDS["fraud"], params=dict(PARAMS, learning_rate=0.03))
EV = {}
for name, r in [("v5, 43 features", RUNS["expert"]), ("v6, 62 features", RUN6)]:
    caught = test_tx[(r["p"] >= r["thr"]) & (test_tx.is_fraud == 1)]
    EV[name] = caught.groupby("label").event_id.nunique()
    print(f"{name:16s} AUROC {roc_auc_score(test_tx.is_fraud, r['p']):.4f} | PR-AUC {r['ap']:.3f} | "
          f"fraud events caught {EV[name].sum()} of 400: {EV[name].to_dict()}")

# explain the v6 model's alerts the same way: every caught event, best alert, graded
alerted = pd.DataFrame({"i": np.arange(len(test_tx)), "p": RUN6["p"],
                        "event": test_tx.event_id, "label": test_tx.label})
alerted = alerted[(alerted.p >= RUN6["thr"]) & (alerted.event >= 0)]
top6 = alerted.sort_values("p", ascending=False).groupby("event").head(1)
CASES6 = make_cases(RUN6, X_v6[split == "train"], test_tx.label, "V", np.sort(top6.i.to_numpy()))
CTX = Context(X_v6[split == "train"], tx.label[split == "train"])          # rarity for the 62 features
V6 = pd.concat([run("fraud", CASES6.values(), n, CONDS[n], G_V6, SCHEMA_F, "card-fraud", key="w1v6")
                for n in ["names+pct", "glossary+pct", "names+pct|top_10+context", "top_10+context"]]
               + [run("fraud", CASES6.values(), "A5 names+tail+lift|chan", CONTEXT_CONDS["A5 names+tail+lift|chan"],
                      G_V6, SCHEMA_F, "card-fraud", render=CTX.render, key="w1v6")], ignore_index=True)
print(f"\n{len(CASES6)} caught events explained by the v6 model:")
print(summarize(V6).to_string())
for label, a, b in [("glossary added (top 10)", "glossary+pct", "names+pct"),
                    ("pin channel flags", "names+pct|top_10+context", "names+pct")]:
    d, (lo, hi) = paired_gain(V6, a, b)
    print(f"{label:26s} {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]")
print("by true pattern:", V6[V6.cond == "A5 names+tail+lift|chan"].groupby("truth").correct.mean()
      .map("{:.0%}".format).to_dict())
print("LLM calls this session:", STATS["api_calls"], "| replayed:", STATS["replayed"],
      "| prompt drift:", STATS["prompt_drift"])

plot_better_model(EV, ALL, V6)                    # figure 4
