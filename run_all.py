"""Run the article's scripts in order, in ONE shared namespace (exactly what a reader
gets by pasting the code blocks into one notebook), and save each script's stdout to
results/<script stem>.txt.

    python run_all.py              # everything: the article's blocks (01-09), then the repo-only scripts (10-14)
    python run_all.py --until 09   # just the article's blocks
"""
import contextlib, glob, io, os, sys, time

until = sys.argv[sys.argv.index("--until") + 1] if "--until" in sys.argv else "99"
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
for path in sorted(glob.glob("code/[0-9][0-9]_*.py")):
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem[:2] > until:
        break
    t0 = time.time()
    with open(f"results/{stem}.txt", "w", encoding="utf-8") as fh, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, fh)):
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    print(f"[run_all] {stem} done in {time.time() - t0:.0f}s", file=sys.__stdout__, flush=True)
print(f"[run_all] total {time.time() - t_all:.0f}s", file=sys.__stdout__)
