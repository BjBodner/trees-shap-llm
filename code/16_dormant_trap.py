# file: code/16_dormant_trap.py
# Repo only. The mistakes that are left: why does the narrator call some takeover and
# card-testing alerts "dormant card"? (channel flags pinned, names + percentiles)
c = FRAUD[FRAUD.cond == "names+pct|top_10+context"]
mine = c[c.truth.isin(["account_takeover", "card_testing"])]
bad, good = mine[mine.said == "dormant_reactivation"], mine[mine.said == mine.truth]
cites = lambda d, f: np.mean([any(r["feature"] == f for r in a["reasons"]) for a in d.answer])
print(f"takeover and card-testing alerts called 'dormant card': {len(bad)} of {len(mine)}")
print(f"  ...that cite longest_gap_days_90d: {cites(bad, 'longest_gap_days_90d'):.0%} "
      f"(the right answers cite it {cites(good, 'longest_gap_days_90d'):.0%} of the time)")
med = lambda d, col, k=1: np.median([CASES[int(i[1:])]["values"][col] / k for i in d.case_id])
print(f"  longest quiet spell in the last 90 days, median: {med(bad, 'longest_gap_days_90d'):.0f} days "
      f"(right answers: {med(good, 'longest_gap_days_90d'):.0f})")
print(f"  time since the purchase before this one, median: {med(bad, 'minutes_since_last_txn', 60):.0f} hours "
      f"(right answers: {med(good, 'minutes_since_last_txn', 60):.0f})")
print("my typology note for takeover ends with:",
      TYPO[TYPO.typology == "account_takeover"].definition.iloc[0].split(". ")[-1])
