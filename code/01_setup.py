# file: code/01_setup.py
# pip install numpy==2.4.6 pandas==3.0.6 scikit-learn==1.9.0 lightgbm==4.7.0 shap==0.53.0 matplotlib==3.11.0 anthropic==1.11.0
import os, sys, json, hashlib, shutil, textwrap, time, urllib.request, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib
if "ipykernel" not in sys.modules:                # a script: draw off-screen
    matplotlib.use("Agg")                         # (a notebook also shows each figure)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ss = np.random.SeedSequence(20261013)            # one root seed for everything
SEEDS = dict(zip(["fraud", "aml", "credit", "claims", "secops", "sample"],
                 [int(s.generate_state(1)[0]) for s in ss.spawn(6)]))
MODEL = "claude-haiku-5-5"                        # the narrator ($0.10 / $0.50 per M tokens)
REPO = "https://raw.githubusercontent.com/BjBodner/trees-shap-llm/main"
for d in ("images", "glossary", "llm_cache"):
    Path(d).mkdir(exist_ok=True)

def asset(rel):
    """Glossaries, cached LLM answers and tsl.py live in the repo. Copy them from a
    local clone (TSL_HOME=path/to/clone) or download them once."""
    p = Path(rel)
    if not p.exists():
        try:
            if os.environ.get("TSL_HOME"):
                shutil.copy(Path(os.environ["TSL_HOME"]) / rel, p)
            else:
                urllib.request.urlretrieve(f"{REPO}/{rel}", p)
        except (OSError, ValueError):
            if not rel.startswith("llm_cache"):   # only the cache may be missing
                raise
    return p

asset("tsl.py")                                   # simulators, plot theme, small helpers
from tsl import (simulate_fraud, build_features, DAY, TYPOLOGIES,      # fraud desk
                 simulate_aml, AML_SEGMENTS, AML_TYPOLOGIES,           # AML desk
                 alert_schema, percentiles, vcat, fmt, ordinal,        # narrator helpers
                 boot_ci, summarize, load_env,                         # stats, reporting
                 apply_theme, bars, paired_bars, save, OKABE_ITO, GREY)  # plots
apply_theme("dark")                               # white on #121212, Okabe-Ito colours
warnings.filterwarnings("ignore", category=UserWarning)   # shap's output-format notice
print("seeds:", SEEDS, "| narrator:", MODEL)
