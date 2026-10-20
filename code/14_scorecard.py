# file: code/14_scorecard.py
# Repo only. Score every pre-registered prediction (PREREGISTRATION.md) against the runs above.
def acc(df, c):
    return df[df.cond == c].correct.mean()

def paired_gain_s(df, a, b, seed=0):
    m = df[df.cond == a][["case_id", "correct"]].merge(df[df.cond == b][["case_id", "correct"]],
                                                     on="case_id")
    d = (m.correct_x.astype(float) - m.correct_y.astype(float)).to_numpy()
    return d.mean(), boot_ci(d, seed=seed)

def usd(df, c):
    g = df[df.cond == c]
    return (0.10 * g.inp.mean() + 0.50 * g.out.mean()) / 1000

S_ = FRAUD_FULL
rows = []
def check(pid, claim, value, held):
    rows.append(dict(id=pid, claim=claim, measured=value, outcome="HELD" if held else "FAILED"))

ap = {k: np.mean(v) for k, v in AP.items()}
check("P1.1", "expert PR-AUC beats kitchen and embed by >= 0.05",
      f"expert {ap['expert']:.3f}, embed {ap['embed']:.3f}, kitchen {ap['kitchen']:.3f}",
      ap["expert"] - max(ap["embed"], ap["kitchen"]) >= 0.05)
e1 = {c: E1[E1.cond == c].correct.mean() for c in E1.cond.unique()}
check("P1.2a", "E1 embedding <= 40%", f"{e1['embed_names']:.0%}", e1["embed_names"] <= 0.40)
check("P1.2b", "E1 kitchen sink 40-75%", f"{e1['kitchen_names']:.0%}", 0.40 <= e1["kitchen_names"] <= 0.75)
check("P1.2c", "E1 expert names >= kitchen + 10 pts", f"{e1['names']:.0%} vs {e1['kitchen_names']:.0%}",
      e1["names"] >= e1["kitchen_names"] + 0.10)
check("P1.2d", "E1 expert + glossary >= 80%", f"{e1['glossary+pct']:.0%}", e1["glossary+pct"] >= 0.80)
check("P1.3", "anonymized (same model) <= 45%", f"{acc(S_, 'anonymized'):.0%}", acc(S_, "anonymized") <= 0.45)
lad = [acc(S_, c) for c in LADDER_FULL]
inv = max([lad[i] - lad[i + 1] for i in range(len(lad) - 1)])
check("P2.1", "ladder rises; no inversion > 5 pts", " < ".join(f"{v:.0%}" for v in lad), inv <= 0.05)
g, ci = paired_gain_s(S_, "glossary+pct", "names")
check("P2.2", "glossary+pct - names >= 10 pts, CI excludes 0", f"{g:+.0%} [{ci[0]:+.0%}, {ci[1]:+.0%}]",
      g >= 0.10 and ci[0] > 0)
check("P2.3", "legacy names within 15 pts of readable names",
      f"{acc(S_, 'legacy_names'):.0%} vs {acc(S_, 'names'):.0%}",
      abs(acc(S_, "legacy_names") - acc(S_, "names")) <= 0.15)
lad_df = S_[S_.cond.isin(LADDER_FULL)]
h = lad_df.groupby("cond").hallucinated.mean(); de = lad_df.groupby("cond").direction_error.mean()
check("P2.4", "hallucinated <= 2% and direction errors <= 5% on every rung",
      f"max halluc {h.max():.0%}, max dir_err {de.max():.0%}", h.max() <= 0.02 and de.max() <= 0.05)
ve = lad_df.groupby("cond").value_error.mean()
check("P2.5", "value error: names >= 25%; names+pct and glossary+pct <= 15%",
      f"names {ve['names']:.0%}, names+pct {ve['names+pct']:.0%}, glossary+pct {ve['glossary+pct']:.0%}",
      ve["names"] >= 0.25 and ve["names+pct"] <= 0.15 and ve["glossary+pct"] <= 0.15)
k10, k3, kall, kgr = (acc(S_, c) for c in ["glossary+pct", "top_3", "top_all", "top_grouped"])
check("P3.1", "top-3 >= 5 pts below top-10; all-43 within 5 pts at >= 2.5x cost",
      f"top-3 {k3:.0%}, top-10 {k10:.0%}, all {kall:.0%}, cost ratio "
      f"{usd(S_, 'top_all') / usd(S_, 'glossary+pct'):.1f}x",
      k3 <= k10 - 0.05 and abs(kall - k10) <= 0.05 and usd(S_, "top_all") >= 2.5 * usd(S_, "glossary+pct"))
check("P3.2", "grouped within 5 pts of top-10", f"{kgr:.0%} vs {k10:.0%}", abs(kgr - k10) <= 0.05)
lv = LAD4.groupby("cond").low_value_error.mean()
check("P4.1", "low-value error: names >= 30%, glossary+pct <= 15%",
      f"names {lv['names']:.0%}, glossary+pct {lv['glossary+pct']:.0%}",
      lv["names"] >= 0.30 and lv["glossary+pct"] <= 0.15)
gT = (LAD4[LAD4.trap & (LAD4.cond == "glossary+pct")].correct.mean()
      - LAD4[LAD4.trap & (LAD4.cond == "names")].correct.mean())
gN = (LAD4[~LAD4.trap & (LAD4.cond == "glossary+pct")].correct.mean()
      - LAD4[~LAD4.trap & (LAD4.cond == "names")].correct.mean())
check("P4.2", "glossary gain >= 5 pts larger on trap cases", f"trap {gT:+.0%}, other {gN:+.0%}",
      gT - gN >= 0.05)
ga, cia = paired_gain_s(AML_FULL, "glossary+pct", "names")
check("P-A1", "AML glossary+pct - names >= 10 pts", f"{ga:+.0%} [{cia[0]:+.0%}, {cia[1]:+.0%}]", ga >= 0.10)
gains = {t: AML_FULL[(AML_FULL.cond == "glossary+pct") & (AML_FULL.truth == t)].correct.mean()
         - AML_FULL[(AML_FULL.cond == "names") & (AML_FULL.truth == t)].correct.mean() for t in AML_TYPOLOGIES}
check("P-A2", "mule_pass_through has the largest AML gain",
      ", ".join(f"{t} {v:+.0%}" for t, v in gains.items()), max(gains, key=gains.get) == "mule_pass_through")
te_cr = split_cr == "test"
auc = {k: roc_auc_score(y_cr[te_cr], CR[k]["p"]) for k in CR}
check("P5.1", "credit: eng ROC-AUC within 0.02 of raw", f"raw {auc['raw']:.3f}, eng {auc['eng']:.3f}",
      abs(auc["raw"] - auc["eng"]) <= 0.02)
sp = CREDIT[CREDIT.has_prot]
c_rn, f_rn = sp[sp.cond == "raw/names"].cites_protected.mean(), sp[sp.cond == "raw/names"].flagged.mean()
c_rg, f_rg = sp[sp.cond == "raw/glossary+pct"].cites_protected.mean(), sp[sp.cond == "raw/glossary+pct"].flagged.mean()
check("P5.2", "raw/names: cites protected >= 30%, flags <= 50%", f"cites {c_rn:.0%}, flags {f_rn:.0%}",
      c_rn >= 0.30 and f_rn <= 0.50)
check("P5.3", "raw/glossary: cites <= 10%, flags >= 80%", f"cites {c_rg:.0%}, flags {f_rg:.0%}",
      c_rg <= 0.10 and f_rg >= 0.80)
eg = CREDIT[(CREDIT.cond == "eng/glossary+pct") & (CREDIT.decision == "deny")]
egn = eg[~eg.targeted]
check("P5.4", "eng/glossary: cites <= 2%, precision >= 80%, direction errors <= 5%",
      f"cites {eg.cites_protected.mean():.0%}, precision {egn.precision.mean():.0%}, "
      f"dir_err {egn.direction_error.mean():.0%}",
      eg.cites_protected.mean() <= 0.02 and egn.precision.mean() >= 0.80 and egn.direction_error.mean() <= 0.05)
en = CREDIT[(CREDIT.cond == "eng/names") & ~CREDIT.targeted]
check("P5.5", "eng/glossary precision >= eng/names - 5 pts",
      f"{egn.precision.mean():.0%} vs {en.precision.mean():.0%}", egn.precision.mean() >= en.precision.mean() - 0.05)
fr, fp = S_[S_.cond == "glossary+pct"].faithful.mean(), S_[S_.cond == "permuted_shap"].faithful.mean()
check("P6.1", "permuted SHAP drops faithful@5 by >= 20 pts", f"{fr:.0%} -> {fp:.0%}", fr - fp >= 0.20)
check("P6.2", "permuted SHAP accuracy stays >= 60%", f"{acc(S_, 'permuted_shap'):.0%}",
      acc(S_, "permuted_shap") >= 0.60)
check("P6.3", "no-SHAP within 10 pts of glossary+pct", f"{acc(S_, 'no_shap'):.0%} vs {k10:.0%}",
      abs(acc(S_, "no_shap") - k10) <= 0.10)
mg = {d: MINI[(MINI.desk == d) & (MINI.cond == "glossary+pct")].correct.mean()
      - MINI[(MINI.desk == d) & (MINI.cond == "names")].correct.mean() for d in ["claims", "secops"]}
check("P7.1", "glossary helps both mini desks, claims more", f"claims {mg['claims']:+.0%}, secops {mg['secops']:+.0%}",
      mg["claims"] >= 0 and mg["secops"] >= 0 and mg["claims"] > mg["secops"])
rp = S_[S_.cond == "repeat"].merge(S_[S_.cond == "glossary+pct"], on="case_id")
agree = (rp.said_x == rp.said_y).mean()
check("P-R", "same prompt twice: typology agrees >= 90%", f"{agree:.0%}", agree >= 0.90)
# Addendum F: the pre-registered follow-up
gF = {v: paired_gain_s(S_, g_, n_) for v, g_, n_ in [("top10", "glossary+pct", "names+pct"),
      ("context", "top_10+context", "names+pct|top_10+context"), ("all", "top_all", "names+pct|top_all")]}
fmt_g = lambda v: f"{gF[v][0] * 100:+.1f} pts [{gF[v][1][0] * 100:+.1f}, {gF[v][1][1] * 100:+.1f}]"
check("F1", "glossary gain with the channel pinned >= 5 pts, CI excludes 0", fmt_g("context"),
      gF["context"][0] >= 0.05 and gF["context"][1][0] > 0)
check("F2", "glossary gain with all 43 features >= 5 pts, CI excludes 0", fmt_g("all"),
      gF["all"][0] >= 0.05 and gF["all"][1][0] > 0)
check("F3", "glossary gain grows by >= 3 pts once the channel is visible",
      f"{gF['top10'][0] * 100:+.1f} -> {gF['context'][0] * 100:+.1f} pts",
      gF["context"][0] - gF["top10"][0] >= 0.03)
ws_ = lambda c: (~S_[S_.cond == c].correct & ~S_[S_.cond == c].unclear).mean()
check("F4", "channel pinned: glossary's confidently-wrong rate <= names' + 2 pts",
      f"{ws_('names+pct|top_10+context'):.0%} vs {ws_('top_10+context'):.0%}",
      ws_("top_10+context") <= ws_("names+pct|top_10+context") + 0.02)
check("F5", "abbreviated names + glossary within 5 pts of readable names + glossary",
      f"{acc(S_, 'legacy+glossary+pct'):.0%} vs {k10:.0%}", abs(acc(S_, "legacy+glossary+pct") - k10) <= 0.05)
en_t = CREDIT[(CREDIT.cond == "eng/names") & CREDIT.has_prot]
check("F6", "credit eng/names letters mention protected attributes 0%",
      f"{en_t.letter_mentions.mean():.0%} of {en_t.case_id.str[2:].nunique()}", en_t.letter_mentions.mean() == 0)
SCORE = pd.DataFrame(rows)
SCORE["pred"] = SCORE.id.str.replace(r"[a-d]$", "", regex=True)
by_pred = SCORE.groupby("pred", sort=False).outcome.apply(lambda o: "HELD" if (o == "HELD").all() else "FAILED")
orig = by_pred[~by_pred.index.str.startswith("F")]
print(f"original pre-registration: {(orig == 'HELD').sum()} of {len(orig)} predictions held "
      f"({(SCORE[~SCORE.pred.str.startswith('F')].outcome == 'HELD').sum()} of "
      f"{(~SCORE.pred.str.startswith('F')).sum()} individual checks)")
fu = by_pred[by_pred.index.str.startswith("F")]
print(f"follow-up (Addendum F): {(fu == 'HELD').sum()} of {len(fu)} held")
print(f"AML glossary gain: {ga * 100:+.1f} pts [{cia[0] * 100:+.1f}, {cia[1] * 100:+.1f}]")
pd.set_option("display.max_colwidth", 90)
print(SCORE.drop(columns="pred").to_string(index=False))
print(f"\nall checks, original + follow-up: {(SCORE.outcome == 'HELD').sum()} of {len(SCORE)} held")
recs = list(CACHE.values())
tin, tout = sum(r["inp"] for r in recs), sum(r["out"] for r in recs)
print(f"narrator spend: {len(recs):,} cached answers, {tin:,} input + {tout:,} output tokens, "
      f"${tin * 0.10 / 1e6 + tout * 0.50 / 1e6:.2f} at $0.10 / $0.50 per million tokens")
