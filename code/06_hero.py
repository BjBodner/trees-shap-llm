# file: code/06_hero.py
# The two plots at the top of this article, made from the hook case above.
# Plot 1: shap's standard waterfall, all 43 features. Plot 2: the same plot crossed
# out, next to the customer message in the LLM answer just printed (replayed from
# llm_cache/responses.jsonl, so both images come out the same on every run).
from matplotlib.patches import FancyArrowPatch
expl = shap.Explanation(values=hc["shap"].to_numpy(), data=hc["values"].round(2).to_numpy(),
                        base_values=float(np.ravel(ex["explainer"].expected_value)[-1]),
                        feature_names=list(hc["shap"].index))
shap.plots.waterfall(expl, max_display=43, show=False)                 # plot 1
fig = plt.gcf()
fig.set_size_inches(8, 10)
for a in fig.axes:
    a.tick_params(labelsize=9)                      # 43 features: small text on purpose
save(fig, "images/01_shap_waterfall.png")

fig = plt.figure(figsize=(16, 8))                   # plot 2
ax = fig.add_axes([0.0, 0.0, 0.42, 0.90])
ax.imshow(plt.imread("images/01_shap_waterfall.png")); ax.axis("off")
for ys in [(0.03, 0.97), (0.97, 0.03)]:             # the X
    ax.plot([0.03, 0.97], ys, transform=ax.transAxes, color="#FFFFFF", lw=10, alpha=0.85)
fig.text(0.01, 0.93, f"Fraud score {hc['score']:.2f}. {len(expl.values)} SHAP values.",
         fontsize=26, fontweight="bold")
fig.add_artist(FancyArrowPatch((0.44, 0.47), (0.53, 0.47), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=55, lw=6, color=OKABE_ITO[2]))
msg = said.answer["customer_message"]
fig.text(0.56, 0.74, "What the LLM tells the client:", fontsize=21, color=GREY)
fig.text(0.56, 0.47, textwrap.fill(msg, len(msg) // 3 + 2), fontsize=25, fontweight="bold",
         va="center", linespacing=1.4)
fig.text(0.56, 0.20, f"{said.model}, {said.date}\nreading the top-10 SHAP values, their "
         "percentiles\nand the feature glossary", fontsize=15, color=GREY, va="top")
save(fig, "images/00_hero.png")
print(f"hero: case {hc['case_id']} ({hc['truth']}), score {hc['score']:.2f}, base value "
      f"{expl.base_values:.2f}, f(x) {expl.base_values + expl.values.sum():.2f}")
print("LLM to the client:", msg)
