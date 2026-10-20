# file: code/08_evidence.py
# Results 2 and 3: what does the narrator need to see, and what does the glossary add?
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
VIEWS = {"top 10": ("names+pct", "glossary+pct"),           # (without, with) the glossary
         "top 10 + channel": ("names+pct|top_10+context", "top_10+context"),
         "all 43": ("names+pct|top_all", "top_all")}
print("\nglossary effect = (names + percentiles + glossary) minus (names + percentiles):")
for view, (no_g, with_g) in VIEWS.items():
    d, (lo, hi) = paired_gain(FRAUD, with_g, no_g)
    print(f"{view:17s} {acc(no_g):.0%} -> {acc(with_g):.0%} | gain {d * 100:+.1f} pts "
          f"[{lo * 100:+.1f}, {hi * 100:+.1f}] | confidently wrong {sure_wrong(no_g):.0%} -> "
          f"{sure_wrong(with_g):.0%}")

fa = FRAUD[(FRAUD.said == "account_takeover") & (FRAUD.truth != "account_takeover")].copy()
fa["cite_distance"] = fa.answer.apply(lambda a: any(r["feature"] == "distance_from_home_km"
                                                    for r in a["reasons"]))
print("\nfalse account-takeover calls (and how many cite distance_from_home_km):")
print(fa.groupby("cond", sort=False).agg(calls=("case_id", "size"),
                                         cite_distance=("cite_distance", "sum")).to_string())
print("glossary row distance_from_home_km catches:", G_FRAUD.at["distance_from_home_km", "catches"],
      f"| rows that mention takeover: {G_FRAUD.catches.str.contains('takeover').sum()} of {len(G_FRAUD)}")

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
for ax, metric, top, lab in [(a1, acc, 1.12, "Correct typology"), (a2, sure_wrong, 0.5, "Confidently wrong")]:
    paired_bars(ax, list(VIEWS), {"names + percentiles": [metric(n) for n, _ in VIEWS.values()],
                                  "+ glossary": [metric(g) for _, g in VIEWS.values()]},
                [GREY, OKABE_ITO[2]], ylim=top, ylabel=lab)
a1.legend(frameon=False, ncol=2)
a1.set_title("Does the glossary pay off once SHAP shows the right evidence?", loc="left")
save(fig, "images/03_glossary_by_view.png")
