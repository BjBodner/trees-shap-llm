# file: code/12_more_desks.py
# Repo only. Desk 4, two quick ones: insurance claims and security alerts.
# Same recipe, new glossary: 8 features each, 3 planted typologies (specs in tsl.py).
MINI = []
for dname, segs, spec, typ, label in [
        ("claims", {"standard": .85, "high_value": .15}, CLAIMS_SPEC, CLAIMS_TYP, "insurance-claims fraud"),
        ("secops", {"workstation": .8, "server": .2}, SECOPS_SPEC, SECOPS_TYP, "security operations (SOC)")]:
    X_d, lab_d, _, sp_d = simulate_desk(SEEDS[dname], 12000, segs, spec, typ, 150)
    _, cases_d = desk(dname, X_d, lab_d, sp_d, SEEDS[dname], 0.05, dname[0].upper(), n_per=20)
    g_d = glossary(dname)
    assert list(g_d.index) == list(X_d.columns)
    sch = alert_schema(list(typ) + ["unclear"])
    MINI += [run(dname, cases_d.values(), c, dict(LADDER[c], k=8), g_d, sch, label)
             for c in ["names", "glossary+pct"]]
MINI = pd.concat(MINI, ignore_index=True)
MINI["desk"] = np.where(MINI.case_id.str[0] == "C", "claims", "secops")
print(MINI.groupby(["desk", "cond"], sort=False).agg(
    n=("correct", "size"), correct=("correct", "mean"), dir_err=("direction_error", "mean"),
    value_err=("value_error", "mean")).map(lambda v: f"{v:.0%}" if isinstance(v, float) else v)
    .to_string())
