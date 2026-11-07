"""Run the article's scripts in order, in ONE shared namespace (exactly what a reader gets by pasting
the code blocks into one notebook), and save each script's stdout to results/<script stem>.txt.

    python run_all.py              # the article's blocks (01-10), then the repo-only scripts (11-15)
    python run_all.py --until 10   # just the article's blocks
    python run_all.py --skip 11_feature_rounds,13_world2   # leave out slow scripts
    python run_all.py --only 19_dormant_ablation           # blocks 1-5, then just this script
"""
import contextlib, io, os, sys, time

ORDER = ["01_setup", "02_fraud_data", "03_models", "04_evidence", "04_llm", "04_verify", "05_fraud_desk",
         "06_hero", "07_same_alerts", "08_glossary_effect", "09_better_context", "10_better_features",
         "11_feature_rounds", "12_aml_desk", "13_world2", "14_credit_desk", "15_more_desks", "16_dormant_trap", "17_feature_ceiling", "18_cost", "19_dormant_ablation", "20_habits"]
until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else "99"
skip = sys.argv[sys.argv.index("--skip") + 1].split(",") if "--skip" in sys.argv else []
only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None   # blocks 1-5 plus these
os.makedirs("results", exist_ok=True)


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


ns = {"__name__": "__main__"}
t_all = time.time()
for stem in ORDER:
    if stem[:2] > until or stem in skip or (only and stem not in only and stem[:2] > "05") or not os.path.exists(f"code/{stem}.py"):
        continue
    t0 = time.time()
    with open(f"results/{stem}.txt", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, fh)):
        exec(compile(open(f"code/{stem}.py", encoding="utf-8").read(), f"code/{stem}.py", "exec"), ns)
    print(f"[run_all] {stem} done in {time.time() - t0:.0f}s", file=sys.__stdout__, flush=True)
print(f"[run_all] total {time.time() - t_all:.0f}s", file=sys.__stdout__)
