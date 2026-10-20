# file: code/04_narrator.py
# The narrator: top SHAP values (+ percentiles, + glossary rows) -> LLM -> JSON
# whose every claim we can check against the model.
import anthropic
load_env(); load_env(".env.local")        # optional: ANTHROPIC_API_KEY (+ workspace id)
ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
client = (anthropic.Anthropic(max_retries=8,
                              default_headers={"anthropic-workspace-id": ws} if ws else None)
          if os.environ.get("ANTHROPIC_API_KEY") else None)
CACHE_PATH, CACHE = asset("llm_cache/responses.jsonl"), {}
if CACHE_PATH.exists():                   # every answer so far: replays need no key
    CACHE = {r["key"]: r for r in map(json.loads, open(CACHE_PATH, encoding="utf-8"))}
STATS = {"api_calls": 0, "replayed": 0, "prompt_drift": 0}
TYPO = pd.read_csv(asset("glossary/typologies.csv"), sep="|")

def glossary(desk):
    return pd.read_csv(asset(f"glossary/{desk}.csv"), sep="|",
                       keep_default_na=False).set_index("feature")

# Same model, same SHAP values. Only the words around them change.
LADDER = {"names": dict(), "names+pct": dict(pct=True), "glossary+pct": dict(pct=True, gloss=True)}

def system_prompt(desk, label, cond):
    hidden = ("\nFeature names are anonymized and values are standardized (mean 0, sd 1)."
              if cond.get("hide") else "")
    return (f"You are a senior {label} analyst. A gradient-boosted tree model scored a case "
            "and you must explain its decision.\nEach feature comes with its SHAP value: "
            "positive pushed the score toward 'suspicious', negative pushed it away (units: "
            f"log-odds). Follow the SHAP sign when you state a feature's effect.{hidden}\n"
            "Rules:\n1. Use only the evidence given. Refer to features by the exact names shown.\n"
            "2. Choose the typology that best explains the evidence, or 'unclear'.\n"
            "3. List the 2 to 4 reasons that matter most. For each: its effect (raises_risk or "
            "lowers_risk) and whether its value is higher, lower or typical compared with "
            "normal (unknown if you cannot tell).\n"
            "4. analyst_rationale: 3 to 5 plain-English sentences for the investigator.\n"
            "5. customer_message: 1 to 2 plain-English sentences for the customer, at most "
            "35 words; no jargon, no feature names, no scores.\nTypologies:\n"
            + "\n".join(f"- {r.typology}: {r.definition}"
                        for r in TYPO[TYPO.desk == desk].itertuples()))

def render_case(case, cond, gl):
    """The evidence: top-k features by |SHAP| (+ pinned context), as a table + glossary rows."""
    k, hide = cond.get("k", 10), cond.get("hide")             # hide = {feature: "V01", ...}
    order = case["shap"].abs().sort_values(ascending=False).index
    keep = list(order if k == "all" else order[:k])
    keep += [f for f in cond.get("pin", []) if f not in keep]  # context a human always needs
    shown = [dict(feature=f, name=hide[f] if hide else f, shap=case["shap"][f],
                  value=case["zvalues" if hide else "values"][f], pct=case["pct"][f],
                  vcat=vcat(case["pct"][f])) for f in keep]    # vcat: higher/typical/lower
    pct_col = lambda s: (" | n/a" if pd.isna(s["pct"]) else f" | {ordinal(s['pct'])}") \
        if cond.get("pct") else ""
    msg = (f"Case {case['case_id']}\n{case['score_line']}\n\n"
           f"Features ({'all' if k == 'all' else 'largest |SHAP| first'}):\n"
           f"feature | value | SHAP{' | percentile' if cond.get('pct') else ''}\n"
           + "\n".join(f"{s['name']} | {fmt(s['value'])} | {s['shap']:+.2f}{pct_col(s)}"
                       for s in shown))
    if cond.get("gloss"):
        rows = [(s["name"], gl.loc[s["feature"]]) for s in shown]
        msg += "\n\nGlossary for these features:\n" + "\n".join(
            f"- {n}: {g.what}. Catches: {g.catches}. Why: {g.motivation}. If higher than "
            f"expected: {g.if_higher}. If lower: {g.if_lower}. Typical: {g.typical}."
            for n, g in rows)
    return msg, shown

def ask(key, system, user, schema):
    """Replay the saved answer to this exact prompt, or ask Claude once and save it."""
    sha = hashlib.sha256((MODEL + system + user + json.dumps(schema, sort_keys=True))
                         .encode()).hexdigest()[:16]
    rec = CACHE.get(key)
    if rec and (rec["prompt_sha"] == sha or client is None):
        STATS["replayed"] += 1
        STATS["prompt_drift"] += rec["prompt_sha"] != sha
        return rec
    if client is None:
        raise RuntimeError(f"no cached answer for {key}: set ANTHROPIC_API_KEY")
    r = client.messages.create(                 # Haiku 5.5: adaptive thinking, no temperature
        model=MODEL, max_tokens=4000, system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}})
    text = next((b.text for b in r.content if b.type == "text"), "")   # skip thinking
    rec = dict(key=key, prompt_sha=sha, model=r.model, date=time.strftime("%Y-%m-%d"),
               stop=r.stop_reason, inp=r.usage.input_tokens, out=r.usage.output_tokens,
               answer=json.loads(text) if r.stop_reason == "end_turn" else None)
    STATS["api_calls"] += 1
    CACHE[key] = rec
    with open(CACHE_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec

def grade(case, shown, ans):
    """Check every cited reason against the evidence. Plain Python, no judge LLM."""
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

def run(desk, cases, name, cond, gl, schema, label):
    """One condition on a set of cases: prompt, ask (or replay), grade."""
    system, rows = system_prompt(desk, label, cond), []
    for case in cases:
        user, shown = render_case(case, cond, gl)
        rec = ask(f"{desk}|{name}|{case['case_id']}", system, user, schema)
        rows.append(dict(grade(case, shown, rec["answer"]), cond=name, inp=rec["inp"],
                         out=rec["out"], answer=rec["answer"], model=rec["model"], date=rec["date"]))
    return pd.DataFrame(rows)

print("narrator ready | cached answers:", len(CACHE), "| live API:", client is not None)
