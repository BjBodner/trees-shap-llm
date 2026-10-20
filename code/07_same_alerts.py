# file: code/07_same_alerts.py
# Result 1: same transactions, same alerts. Only the feature set differs.
kit, emb = RUNS["kitchen"], RUNS["embed"]
both = [i for i in S_rows if kit["p"][i] >= kit["thr"] and emb["p"][i] >= emb["thr"]]
print(f"{len(both)} of {len(S_rows)} cases are alerted by all three models:",
      test_tx.label[both].value_counts().to_dict())
E1 = [FRAUD[FRAUD.case_id.isin([f"F{i}" for i in both])
            & FRAUD.cond.isin(["anonymized", "names", "glossary+pct"])]]
for fs in ["kitchen", "embed"]:                    # same rows, other features, names shown
    cs = make_cases(RUNS[fs], FEATURE_SETS[fs][split == "train"], test_tx.label,
                    fs[0].upper(), both)
    E1.append(run("fraud", cs.values(), f"{fs}_names", LADDER["names"], None, SCHEMA_F,
                  "card-fraud"))
E1 = pd.concat(E1, ignore_index=True)
print(summarize(E1).to_string())

order = [("embed_names", "embed", "PCA\nembedding"), ("kitchen_names", "kitchen", "Kitchen\nsink"),
         ("anonymized", "expert", "Expert,\nnames hidden"), ("names", "expert", "Expert\nfeatures"),
         ("glossary+pct", "expert", "Expert + glossary\n+ percentiles")]
acc = np.array([E1[E1.cond == c].correct.mean() for c, _, _ in order])
ci = np.array([boot_ci(E1[E1.cond == c].correct) for c, _, _ in order])
print(f"expert + glossary vs kitchen sink: {acc[-1] / acc[1]:.1f}x more correct rationales")

fig, ax = plt.subplots(figsize=(9, 6))
bars(ax, [f"{lab}\nPR-AUC {RUNS[m]['ap']:.2f}" for _, m, lab in order], acc, ci=ci,
     colors=[GREY, GREY, OKABE_ITO[1], OKABE_ITO[0], OKABE_ITO[2]], ylim=1.1,
     ylabel="Correct fraud typology", fs=11)
ax.axhline(0.25, ls="--", lw=1, color=GREY)
ax.text(4.45, 0.27, "chance (1 in 4)", ha="right", fontsize=11, color=GREY)
ax.set_title("Same transactions, same alerts.\nHow often does the LLM name the real fraud "
             "pattern?", loc="left")
save(fig, "images/02_same_alerts.png")
