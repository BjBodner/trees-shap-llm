# file: code/17_feature_ceiling.py
# Repo only. How much of the answer is already in the inputs? A plain tree classifier that sees
# only the 43 inputs (no LLM, no SHAP), trained on TRAIN fraud rows, names the fraud pattern.
fraud_rows = (split == "train") & (tx.label != "legit")
pattern = pd.Series(tx.label[fraud_rows].to_numpy()).astype("category")
clf = lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=8,
                         min_child_samples=10, verbose=-1, num_threads=4, random_state=1)
clf.fit(X_expert[fraud_rows], pattern.cat.codes.to_numpy())
rows = list(CASES)                                  # the same caught events the narrator explains
said = pattern.cat.categories[clf.predict(RUNS["expert"]["X"].iloc[rows])]
truth = np.array([CASES[i]["truth"] for i in rows])
print(f"a tree on the 43 inputs alone names the fraud pattern for {np.mean(said == truth):.1%} of "
      f"the {len(rows)} caught events")
print(pd.crosstab(truth, said).to_string())
