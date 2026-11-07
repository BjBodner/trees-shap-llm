# file: code/18_cost.py
# Repo only. What did all of this cost? Every answer in the cache records its tokens.
recs = list(CACHE.values())
usd = sum(r["inp"] * 0.10 + r["out"] * 0.50 for r in recs) / 1e6      # Haiku 5.5: $0.10 / $0.50 per million tokens
print(f"{len(recs):,} saved answers, ${usd:.2f} in total at Haiku 5.5 prices")
