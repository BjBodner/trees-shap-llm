# file: code/04_llm.py
# Narrator step 2: the LLM call. A system prompt with the rules and the typologies,
# one structured-output request per case, and a cache, so every answer replays exactly.
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

print("narrator ready | cached answers:", len(CACHE), "| live API:", client is not None)
