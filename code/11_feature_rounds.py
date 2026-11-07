# file: code/11_feature_rounds.py
# Repo only. How the 19 features were chosen. Every choice below is scored by 4-fold
# cross-validation on the TRAIN + VAL rows only (a customer never sits in two folds).
# The test split was scored once, at the end (block 10).
from sklearn.model_selection import GroupKFold
y = tx.is_fraud.to_numpy()
DEV = np.isin(split, ["train", "val"])
GROUPS = {"card testing (4)": ["n_tiny_online_6h", "n_new_merchants_1h", "n_new_merchants_6h", "n_online_txn_1h"],
          "takeover (7)": ["amount_zscore_for_cust", "hour_deviation_hours", "online_ip_shift_km",
                           "distance_from_last_online_km", "cust_vpn_share_90d", "spend_1h_vs_cust_median",
                           "max_amount_24h_vs_cust_median"],
          "dormant (2)": ["expected_txns_missed", "n_new_merchants_24h"],
          "counterfeit (3)": ["n_far_card_present_24h", "n_far_card_present_7d", "amount_vs_cust_mcc_mean"]}
NEW59 = [c for g in GROUPS.values() for c in g]                  # the first 16 new features
BURST3 = [c for c in X_v6.columns if c not in X_expert.columns and c not in NEW59]   # the last 3

def cv_score(X, name, n_trees=500, **params):
    """Out-of-fold AUROC and PR-AUC on the dev rows, and fraud events caught at a 1% alert rate."""
    p = dict(PARAMS, n_estimators=n_trees, **params)
    rows, oof = np.where(DEV)[0], np.full(len(tx), np.nan)
    for a, b in GroupKFold(4).split(rows, groups=tx.cust.to_numpy()[rows]):
        m = lgb.LGBMClassifier(**p, random_state=SEEDS["fraud"]).fit(X.iloc[rows[a]], y[rows[a]])
        oof[rows[b]] = m.predict_proba(X.iloc[rows[b]])[:, 1]
    s, thr = oof[rows], np.quantile(oof[rows], 0.99)
    hit = DEV & (y == 1) & (np.nan_to_num(oof) >= thr)
    print(f"{name:40s} features {X.shape[1]:2d} | AUROC {roc_auc_score(y[rows], s):.4f} | "
          f"PR-AUC {average_precision_score(y[rows], s):.3f} | events caught {tx.event_id[hit].nunique()} of 400")

cv_score(X_expert, "the 43 features of block 2")
for g, cols in GROUPS.items():
    cv_score(pd.concat([X_expert, X_v6[cols]], axis=1), f"+ {g}")
cv_score(X_v6[list(X_expert.columns) + NEW59], "+ all 16")
cv_score(X_v6, "+ all 16 + 3 burst features (62)")
cv_score(X_v6, "62, learning rate 0.03 x 800 trees", n_trees=800, learning_rate=0.03)
gain = pd.Series(lgb.LGBMClassifier(**dict(PARAMS, n_estimators=500), importance_type="gain",
                                    random_state=SEEDS["fraud"]).fit(X_v6[DEV], y[DEV]).feature_importances_,
                 X_v6.columns)
gain = (gain / gain.sum()).sort_values(ascending=False)
print(f"\nthe 5 features with the most gain: { {k: f'{v:.0%}' for k, v in gain.head(5).items()} }")
print(f"features with under 0.1% of the gain: {int((gain < 0.001).sum())} (they stay in: a few extra features are fine)")
cv_score(X_v6[list(gain[gain > 0.004].index)], f"pruned to the {int((gain > 0.004).sum())} best")
mono = {"distance_from_home_km": 1, "implied_speed_kmh": 1, "n_tiny_txn_1h": 1, "n_tiny_online_6h": 1,
        "n_new_merchants_6h": 1, "n_new_merchants_1h": 1, "distance_from_last_txn_km": 1}
cv_score(X_v6, "62 + expert monotone constraints (7)", monotone_constraints=[mono.get(c, 0) for c in X_v6.columns])
