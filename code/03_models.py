# file: code/03_models.py
# One LightGBM recipe for every feature set. Analysts review the top 1% of scores.
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=3000, verbose=-1, deterministic=True, force_row_wise=True,
              num_threads=4)

def train(X, y, split, seed, alert_rate=0.01, params=None):
    m = lgb.LGBMClassifier(**(params or PARAMS), random_state=seed)
    m.fit(X[split == "train"], y[split == "train"], eval_X=X[split == "val"],
          eval_y=y[split == "val"], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict_proba(X[split == "test"])[:, 1]
    return dict(p=p, thr=np.quantile(p, 1 - alert_rate),       # alert threshold
                ap=average_precision_score(y[split == "test"], p),
                X=X[split == "test"].reset_index(drop=True),
                explainer=shap.TreeExplainer(m))               # TreeSHAP, in log-odds

y = tx.is_fraud.to_numpy()
test_tx = tx[split == "test"].reset_index(drop=True)
RUNS = {name: train(X, y, split, SEEDS["fraud"]) for name, X in FEATURE_SETS.items()}
for name, r in RUNS.items():
    caught = test_tx[(r["p"] >= r["thr"]) & (test_tx.is_fraud == 1)]
    print(f"{name:8s} PR-AUC {r['ap']:.3f} | fraud events caught: "
          f"{caught.groupby('label').event_id.nunique().to_dict()}")
