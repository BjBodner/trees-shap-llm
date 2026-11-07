# file: code/08_glossary_effect.py
# What does the glossary add, and what does it steer? Paired over the same cases.
cf = [c for c in CASES.values() if c["truth"] == "counterfeit_card"]
n_cp = sum(any(s["feature"] == "is_card_present" for s in render_case(c, LADDER["names"], None)[1])
           for c in cf)
print(f"is_card_present in the top-10 SHAP list: {n_cp} of {len(cf)} counterfeit cases")
cf_acc = FRAUD[FRAUD.truth == "counterfeit_card"].groupby("cond", sort=False).correct.mean()
print("counterfeit cards named correctly:", cf_acc.map("{:.0%}".format).to_dict())

def paired_gain(df, a, b):
    """Accuracy of condition a minus condition b, paired over the same cases (+ 95% CI)."""
    m = df[df.cond == a][["case_id", "correct"]].merge(
        df[df.cond == b][["case_id", "correct"]], on="case_id")
    d = m.correct_x.astype(float).to_numpy() - m.correct_y.astype(float).to_numpy()
    return d.mean(), boot_ci(d)

acc = lambda c: FRAUD[FRAUD.cond == c].correct.mean()
sure_wrong = lambda c: (~FRAUD[FRAUD.cond == c].correct & ~FRAUD[FRAUD.cond == c].unclear).mean()
print("\npaired differences between evidence formats (points, 95% interval):")
for label, a, b in [("percentiles vs names only", "names+pct", "names"),
                    ("glossary added (top 10)", "glossary+pct", "names+pct"),
                    ("glossary vs names only", "glossary+pct", "names"),
                    ("pin channel flags (names+pct)", "names+pct|top_10+context", "names+pct"),
                    ("glossary added (channel pinned)", "top_10+context", "names+pct|top_10+context")]:
    d, (lo, hi) = paired_gain(FRAUD, a, b)
    print(f"{label:33s} {acc(b):.0%} -> {acc(a):.0%} | {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]"
          f" | confidently wrong {sure_wrong(b):.0%} -> {sure_wrong(a):.0%}")

fa = FRAUD[(FRAUD.said == "account_takeover") & (FRAUD.truth != "account_takeover")].copy()
fa["cite_distance"] = fa.answer.apply(lambda a: any(r["feature"] == "distance_from_home_km"
                                                    for r in a["reasons"]))
print("\nfalse account-takeover calls (and how many cite distance_from_home_km):")
print(fa.groupby("cond", sort=False).agg(calls=("case_id", "size"),
                                         cite_distance=("cite_distance", "sum")).to_string())
print("glossary row distance_from_home_km catches:", G_FRAUD.at["distance_from_home_km", "catches"],
      f"| rows that mention takeover: {G_FRAUD.catches.str.contains('takeover').sum()} of {len(G_FRAUD)}")
