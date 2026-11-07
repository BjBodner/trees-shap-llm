# file: code/20_habits.py
# Repo only. Which honest habits set off the better model's false alarms? (62-feature model, test rows,
# alerts = the top 1% of scores). Each purchase in the simulator carries the habit that produced it.
alert = RUN6["p"] >= RUN6["thr"]
honest = test_tx.label.to_numpy() == "legit"
h = pd.DataFrame({"habit": test_tx.habit.to_numpy(), "alert": alert})[honest]
g = h.groupby("habit").agg(purchases=("alert", "size"), false_alarms=("alert", "sum"))
g["per_1000_purchases"] = (1000 * g.false_alarms / g.purchases).round(2)
g["share_of_false_alarms"] = (g.false_alarms / g.false_alarms.sum()).map("{:.0%}".format)
print(g.sort_values("per_1000_purchases", ascending=False).to_string())
print(f"false alarms in total: {int(g.false_alarms.sum())} among {int(g.purchases.sum()):,} honest purchases")
