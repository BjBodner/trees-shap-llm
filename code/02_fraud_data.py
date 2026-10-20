# file: code/02_fraud_data.py
# Desk 1, card fraud: 5,000 simulated customers over 120 days with 4 planted fraud
# typologies, so every fraud case has a KNOWN true reason to grade the LLM against.
tx, events, cust = simulate_fraud(SEEDS["fraud"], n_cust=5000)
print(f"{len(tx):,} transactions, {tx.is_fraud.sum():,} fraud ({tx.is_fraud.mean():.2%}), "
      f"{len(events)} planted fraud events")
split = np.select([tx.t < 30 * DAY, tx.t < 75 * DAY, tx.t < 85 * DAY],
                  ["warmup", "train", "val"], "test")

# The same transactions, three ways to make 43 model inputs:
MCC_MED = tx[tx.t < 85 * DAY].groupby("mcc").amount.median().reindex(range(10)).to_numpy()
X_expert, X_kitchen = build_features(tx, cust, MCC_MED)   # expert: one glossary row each
fill = X_expert[split == "train"].median()                 # kitchen: windows x aggregates
pca = make_pipeline(StandardScaler(), PCA(n_components=43, random_state=0))
pca.fit(X_expert[split == "train"].fillna(fill))           # embed: expert, PCA-rotated
X_embed = pd.DataFrame(pca.transform(X_expert.fillna(fill)),
                       columns=[f"emb_{i:02d}" for i in range(43)])
FEATURE_SETS = {"embed": X_embed, "kitchen": X_kitchen, "expert": X_expert}
print({k: v.shape[1] for k, v in FEATURE_SETS.items()}, "| rows per split:",
      pd.Series(split).value_counts().to_dict())
