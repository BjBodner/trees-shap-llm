# file: code/04_evidence.py
# Narrator step 1: the evidence. What the LLM sees for one case: the top-k features
# by |SHAP| (+ pinned context), as a small table, plus glossary rows if asked for.
def glossary(desk):
    return pd.read_csv(asset(f"glossary/{desk}.csv"), sep="|",
                       keep_default_na=False).set_index("feature")

# Same model, same SHAP values. Only the words around them change.
LADDER = {"names": dict(), "names+pct": dict(pct=True), "glossary+pct": dict(pct=True, gloss=True)}

def pct_cell(s):
    return " | n/a" if pd.isna(s["pct"]) else f" | {ordinal(s['pct'])}"   # "| 87th"

def render_case(case, cond, gl):
    k, hide = cond.get("k", 10), cond.get("hide")             # hide = {feature: "V01", ...}
    order = case["shap"].abs().sort_values(ascending=False).index
    keep = list(order if k == "all" else order[:k])
    keep += [f for f in cond.get("pin", []) if f not in keep]  # context a human always needs
    shown = [dict(feature=f, name=hide[f] if hide else f, shap=case["shap"][f],
                  value=case["zvalues" if hide else "values"][f], pct=case["pct"][f],
                  vcat=vcat(case["pct"][f])) for f in keep]    # vcat: higher/typical/lower
    msg = (f"Case {case['case_id']}\n{case['score_line']}\n\n"
           f"Features ({'all' if k == 'all' else 'largest |SHAP| first'}):\n"
           f"feature | value | SHAP{' | percentile' if cond.get('pct') else ''}\n"
           + "\n".join(f"{s['name']} | {fmt(s['value'])} | {s['shap']:+.2f}"
                       + (pct_cell(s) if cond.get("pct") else "") for s in shown))
    if cond.get("gloss"):
        rows = [(s["name"], gl.loc[s["feature"]]) for s in shown]
        msg += "\n\nGlossary for these features:\n" + "\n".join(
            f"- {n}: {g.what}. Catches: {g.catches}. Why: {g.motivation}. If higher than "
            f"expected: {g.if_higher}. If lower: {g.if_lower}. Typical: {g.typical}."
            for n, g in rows)
    return msg, shown
