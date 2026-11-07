# file: code/04_verify.py
# Narrator step 3: check the answer. Every cited reason is checked against the evidence
# in plain Python (no judge LLM), and run() ties the three steps together.
def grade(case, shown, ans):
    by = {s["name"]: s for s in shown}
    reasons = ans["reasons"] if ans else []
    known = [r for r in reasons if r["feature"] in by]
    judged = [r for r in known if r["value_vs_normal"] != "unknown"
              and by[r["feature"]]["vcat"] is not None]
    return dict(case_id=case["case_id"], truth=case["truth"], said=ans and ans["typology"],
                correct=bool(ans) and ans["typology"] == case["truth"],
                unclear=bool(ans) and ans["typology"] == "unclear",
                hallucinated=any(r["feature"] not in by for r in reasons),
                direction_error=any((r["effect"] == "raises_risk") != (by[r["feature"]]["shap"] > 0)
                                    for r in known),
                value_error=np.mean([r["value_vs_normal"] != by[r["feature"]]["vcat"]
                                     for r in judged]) if judged else np.nan)

def run(desk, cases, name, cond, gl, schema, label, render=render_case, key=None):
    """One condition on a set of cases: evidence -> ask (or replay) -> grade."""
    system, rows = system_prompt(desk, label, cond), []
    for case in cases:
        user, shown = render(case, cond, gl)
        rec = ask(f"{key or desk}|{name}|{case['case_id']}", system, user, schema)
        rows.append(dict(grade(case, shown, rec["answer"]), cond=name, inp=rec["inp"],
                         out=rec["out"], answer=rec["answer"], model=rec["model"], date=rec["date"]))
    return pd.DataFrame(rows)
