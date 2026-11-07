# file: code/19_dormant_ablation.py
# Repo only. Is my "inactive for weeks" note really what sends takeover and card-testing alerts to
# "dormant card"? Ask the same 371 alerts again (names + percentiles, channel flags pinned) with both
# sentences of the note removed from the fraud-pattern list, and compare.
NOTE = [" This takes precedence over account_takeover whenever the card had been inactive for weeks.",
        " If the card had been inactive for weeks before this, choose dormant_reactivation instead."]
TYPO_WITH_NOTE = TYPO
TYPO = TYPO.assign(definition=TYPO.definition.map(lambda d: d.replace(NOTE[0], "").replace(NOTE[1], "")))
print("definitions changed:", int((TYPO.definition != TYPO_WITH_NOTE.definition).sum()))
cond = "names+pct|top_10+context"
NO_NOTE = run("fraud", CASES.values(), cond, CONDS[cond], G_FRAUD, SCHEMA_F, "card-fraud", key="no-note")
TYPO = TYPO_WITH_NOTE                              # put the note back
WITH_NOTE = FRAUD[FRAUD.cond == cond]
for name, d in [("with the note", WITH_NOTE), ("without the note", NO_NOTE)]:
    mine = d[d.truth.isin(["account_takeover", "card_testing"])]
    print(f"{name:17s} correct overall {d.correct.mean():.0%} | dormant-card cases correct "
          f"{d[d.truth == 'dormant_reactivation'].correct.mean():.0%} | takeover and card-testing alerts "
          f"called 'dormant card': {int((mine.said == 'dormant_reactivation').sum())} of {len(mine)}")
print("the 0.97 hero case, without the note, was called:", NO_NOTE[NO_NOTE.case_id == hc["case_id"]].said.iloc[0])
