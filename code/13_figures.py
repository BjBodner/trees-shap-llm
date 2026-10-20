# file: code/13_figures.py
# Repo only. Figures for the conditions the article sums up: the glossary ladder,
# SHAP list length, the cross-examination, and the credit desk.
RUNGS = ["names", "names+pct", "glossary", "glossary+pct"]
ink = plt.rcParams["text.color"]
# Figure 3: the ladder, fraud and AML
fig, ax = plt.subplots(figsize=(9, 5.5))
lab = ["anonymized", "legacy_names"] + RUNGS
for j, (d, df, col) in enumerate([("Fraud", FRAUD_FULL, OKABE_ITO[0]), ("AML", AML_FULL, OKABE_ITO[2])]):
    xs = [i for i, c in enumerate(lab) if c in set(df.cond)]
    acc = [df[df.cond == lab[i]].correct.mean() for i in xs]
    ci = np.array([boot_ci(df[df.cond == lab[i]].correct) for i in xs])
    ax.plot(xs, acc, "o-", color=col, lw=2.5, ms=9, label=d)
    ax.fill_between(xs, ci[:, 0], ci[:, 1], color=col, alpha=0.2)
    for x, v in zip(xs, acc):
        ax.text(x, v + (0.05 if j == 0 else -0.09), f"{v:.0%}", ha="center", color=col,
                fontsize=12, fontweight="bold")
ax.set_xticks(range(len(lab)), ["names\nhidden", "feature-store\nabbrev.", "readable\nnames",
                                "+ percentiles", "+ glossary", "+ glossary\n+ percentiles"], fontsize=11)
ax.set_ylim(0, 1.08); ax.set_ylabel("Correct typology")
ax.axhline(0.25, ls="--", lw=1, color=GREY)
ax.text(5.4, 0.265, "chance (1 in 4)", ha="right", fontsize=11, color=GREY)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(frameon=False, loc="lower right")
ax.set_title("Same model, same SHAP values. Only the words change.", loc="left")
save(fig, "images/04_ladder.png")

# Figure 4: how long should the SHAP list be? (all cases vs counterfeit cases)
ks = ["top_3", "top_5", "glossary+pct", "top_10+context", "top_all"]
klab = ["top 3", "top 5", "top 10", "top 10\n+ channel", "all 43"]
cf = FRAUD_FULL.truth == "counterfeit_card"
acc_all = [FRAUD_FULL[FRAUD_FULL.cond == k].correct.mean() for k in ks]
acc_cf = [FRAUD_FULL[(FRAUD_FULL.cond == k) & cf].correct.mean() for k in ks]
usd = [(0.10 * FRAUD_FULL[FRAUD_FULL.cond == k].inp.mean() + 0.50 * FRAUD_FULL[FRAUD_FULL.cond == k].out.mean()) / 1000
       for k in ks]                                  # Haiku 5.5: $0.10 / $0.50 per M tokens
for k, a, c, u in zip(ks, acc_all, acc_cf, usd):
    print(f"{k:15s} all cases {a:.0%} | counterfeit {c:.0%} | ${u:.3f} per 1,000")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
xs, w = np.arange(len(ks)), 0.38
for off, vals, col, lbl in [(-w / 2, acc_all, OKABE_ITO[0], "all 200 cases"),
                            (w / 2, acc_cf, OKABE_ITO[4], "counterfeit cards (50)")]:
    a1.bar(xs + off, vals, w, color=col, label=lbl)
    for x, v in zip(xs + off, vals):
        a1.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=12, fontweight="bold")
a1.set_ylim(0, 1.15); a1.set_ylabel("Correct typology")
a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
a1.legend(frameon=False, loc="upper left", fontsize=11, ncol=2)
a1.set_title("The top-10 SHAP list hides the one fact a human needs", loc="left")
a2.bar(xs, usd, color=GREY)
for x, v in zip(xs, usd):
    a2.text(x, v * 1.04, f"${v:.2f}", ha="center", fontsize=12, fontweight="bold")
a2.set_xticks(xs, klab, fontsize=12)
a2.set_ylabel("USD per 1,000\nexplanations"); a2.set_ylim(0, max(usd) * 1.3)
save(fig, "images/05_topk.png")

# Figure 5: cross-examination
cx = ["glossary+pct", "permuted_shap", "no_shap"]
cxl = ["real SHAP", "SHAP shuffled\nacross features", "no SHAP\n(all 43 values)"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
bars(a1, cxl[:2], [FRAUD_FULL[FRAUD_FULL.cond == c].faithful.mean() for c in cx[:2]],
     colors=[OKABE_ITO[2], OKABE_ITO[5]], ylabel="Share", fs=11)
a1.set_title("Cites the features SHAP\nranked highest", loc="left", fontsize=14)
bars(a2, cxl, [FRAUD_FULL[FRAUD_FULL.cond == c].correct.mean() for c in cx],
     colors=[OKABE_ITO[2], OKABE_ITO[5], GREY], fs=11)
a2.set_title("Names the true\nfraud typology", loc="left", fontsize=14)
save(fig, "images/06_cross_exam.png")

# Figure 6: credit. What happens when SEX, MARRIAGE or AGE drives a denial?
sub = CREDIT[CREDIT.has_prot]
clean = CREDIT[(CREDIT.decision == "deny") & ~CREDIT.has_prot & ~CREDIT.targeted]
cc = ["raw/names", "raw/glossary+pct", "eng/glossary+pct"]
ccl = ["raw model,\nnames only", "raw model,\n+ glossary", "engineered model\n(no demographics)"]
mets = [("lists_protected", "listed as an internal reason", OKABE_ITO[0]),
        ("letter_mentions", "mentioned in the customer letter", OKABE_ITO[5]),
        ("flagged", "flagged for compliance review", OKABE_ITO[1])]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9.5), gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.6})
xs, w = np.arange(3), 0.27
for j, (m, lbl, col) in enumerate(mets):
    v = [sub[sub.cond == c][m].mean() for c in cc]
    a1.bar(xs + (j - 1) * w, v, w, color=col, label=lbl)
    for x, vv in zip(xs + (j - 1) * w, v):
        a1.text(x, vv + 0.03, f"{vv:.0%}", ha="center", fontsize=12, fontweight="bold")
a1.set_xticks(xs, ccl, fontsize=11); a1.set_ylim(0, 1.3)
a1.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
a1.set_title(f"{sub.case_id.str[2:].nunique()} denials where SEX, MARRIAGE or AGE was a top-4 driver",
             loc="left", fontsize=14)
fa = ["raw/names", "raw/glossary+pct", "eng/names", "eng/glossary+pct"]
fv = [clean[clean.cond == c].flagged.mean() for c in fa]
a2.bar(range(4), fv, color=GREY)
for x, v in enumerate(fv):
    a2.text(x, v + 0.03, f"{v:.0%}", ha="center", fontsize=12, fontweight="bold")
a2.set_xticks(range(4), ["raw,\nnames", "raw,\n+ glossary", "engineered,\nnames", "engineered,\n+ glossary"],
              fontsize=11)
a2.set_ylim(0, 1.15); a2.set_title("False alarms: flags on ordinary denials", loc="left", fontsize=14)
save(fig, "images/07_credit_protected.png")
print("flag rate on ordinary denials:", ", ".join(f"{c} {v:.0%}" for c, v in zip(fa, fv)))

print("figures saved | API calls this run:", STATS["api_calls"], "| replayed:", STATS["replayed"],
      "| prompt drift:", STATS["prompt_drift"])
