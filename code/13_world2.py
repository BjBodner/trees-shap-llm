# file: code/13_world2.py
# Repo only. World 2: the whole fraud desk again with a new seed (new customers, new fraud,
# new random numbers), to see whether the findings were a fluke of world 1.
import io, contextlib
w2 = {"__name__": "world2"}
def run_in(space, stem):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(compile(open(f"code/{stem}.py", encoding="utf-8").read(), stem, "exec"), space)
    return out.getvalue()
run_in(w2, "01_setup")
w2["SEEDS"] = dict(zip(["fraud", "aml", "credit", "claims", "secops", "sample"],
                       [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(20261106).spawn(6)]))
print(run_in(w2, "02_fraud_data").strip().splitlines()[0])
print(run_in(w2, "03_models").strip().splitlines()[-1])
ex2 = w2["RUNS"]["expert"]
al2 = pd.DataFrame({"i": np.arange(len(w2["test_tx"])), "p": ex2["p"],
                    "event": w2["test_tx"].event_id, "label": w2["test_tx"].label})
al2 = al2[(al2.p >= ex2["thr"]) & (al2.event >= 0)]
top2 = al2.sort_values("p", ascending=False).groupby("event").head(1)
CASES2 = make_cases(ex2, w2["X_expert"][w2["split"] == "train"], w2["test_tx"].label, "W", np.sort(top2.i.to_numpy()))
print(f"{len(CASES2)} caught fraud events:", w2["test_tx"].label[list(CASES2)].value_counts().to_dict())

CTX_V5 = CTX                                        # keep world 1's rarity tables, use world 2's for now
CTX = Context(w2["X_expert"][w2["split"] == "train"], w2["tx"].label[w2["split"] == "train"])
W2 = pd.concat([run("fraud", CASES2.values(), n, CONDS[n], G_FRAUD, SCHEMA_F, "card-fraud", key="w2")
                for n in ["names+pct", "glossary+pct", "names+pct|top_10+context", "top_10+context"]]
               + [run("fraud", CASES2.values(), n, CONTEXT_CONDS[n], G_FRAUD, SCHEMA_F, "card-fraud",
                      render=CTX.render, key="w2") for n in ["A2 names+tail+lift", "A3 names+tail+lift+mix", "A5 names+tail+lift|chan"]],
               ignore_index=True)
CTX = CTX_V5
print(summarize(W2).to_string())
print("\npaired differences in world 2 (points, 95% interval):")
for label, a, b in [("glossary added (top 10)", "glossary+pct", "names+pct"),
                    ("pin channel flags", "names+pct|top_10+context", "names+pct"),
                    ("+ past fraud, counted per row", "A3 names+tail+lift+mix", "A2 names+tail+lift"),
                    ("rarity + rate vs percentile", "A2 names+tail+lift", "names+pct")]:
    d, (lo, hi) = paired_gain(W2, a, b)
    print(f"{label:32s} {d * 100:+5.1f} [{lo * 100:+5.1f}, {hi * 100:+5.1f}]")
print("LLM calls this session:", STATS["api_calls"], "| replayed:", STATS["replayed"],
      "| prompt drift:", STATS["prompt_drift"])
del w2
