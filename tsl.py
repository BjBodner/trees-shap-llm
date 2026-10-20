"""tsl.py: the plumbing behind "Trees + SHAP + LLMs = ?".

Simulators with planted typologies (card fraud, AML, insurance claims, security
alerts), the 43-feature fraud builder, the credit loader, and small helpers.
The article's code blocks import the names they use from this file (block 1 lists
them): the simulators and feature builder, the narrator's JSON schema, and small
stats, formatting and plotting helpers. The models, SHAP, prompts, checks and
experiments are in the article itself.
"""
import numpy as np
import pandas as pd
import sys
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- card fraud
MCC = ["grocery", "restaurant", "fuel", "retail", "online_services",
       "digital_goods", "electronics", "jewelry", "travel", "pharmacy"]
MCC_OFFSET = np.array([0.2, -0.2, 0.1, 0.4, -0.5, -0.3, 1.5, 1.8, 1.6, -0.3])
MCC_RISK = np.array([0, 0, 0, 0, 1, 3, 3, 3, 1, 0])
TYPOLOGIES = ["card_testing", "account_takeover", "counterfeit_card",
              "dormant_reactivation"]
DAY = 1440.0                                   # minutes

def _rows(t, amt, online, mcc, merch, xy, label="legit"):
    return pd.DataFrame(dict(t=t, amount=np.round(amt, 2), online=online, mcc=mcc,
                             merchant=merch, x=xy[:, 0], y=xy[:, 1], label=label))

def simulate_fraud(seed, n_cust=5000, days=120):
    r = np.random.default_rng(seed)
    n_ev = int(0.02 * n_cust)                           # events per typology per split
    home = r.uniform(0, 2000, (n_cust, 2))
    rate = r.gamma(2.0, 0.5, n_cust)                    # txns per day
    mu = r.normal(3.4, 0.5, n_cust)                     # log-amount level
    peak, hsd = r.uniform(8, 21, n_cust), r.uniform(1.5, 4, n_cust)
    p_online = r.beta(2, 5, n_cust)
    pref = r.dirichlet([5, 4, 3, 4, 2, 0.3, 0.6, 0.2, 0.5, 1.5], n_cust)
    age0 = r.uniform(30, 3000, n_cust)                  # account age at day 0
    favs = r.integers(0, 300, (n_cust, 12))             # favourite merchants
    gamer = r.random(n_cust) < 0.10                     # app-store micro-bursts

    victims = r.permutation(n_cust)[: 8 * n_ev]
    ev = pd.DataFrame({"cust": victims, "typology": np.repeat(TYPOLOGIES, 2 * n_ev),
                       "split": np.tile(["train"] * n_ev + ["test"] * n_ev, 4)})
    ev["t0"] = np.where(ev.split == "train", r.uniform(35, 82, len(ev)),
                        r.uniform(86, days - 2, len(ev))) * DAY
    ev["event_id"] = np.arange(len(ev))
    ev_of = dict(zip(ev.cust, ev.index))
    quiet = r.random(n_cust) < 0.12                     # legit dormancy (holidays)
    rows, profile_changes = [], {}
    for c in range(n_cust):
        n = r.poisson(rate[c] * days)
        t = np.sort(r.uniform(0, days, n)).astype(int) * DAY + (
            (r.normal(peak[c], hsd[c], n) % 24) * 60)
        online = r.random(n) < p_online[c]
        mcc = r.choice(10, n, p=pref[c])
        amt = np.exp(r.normal(mu[c] + MCC_OFFSET[mcc], 0.7))
        rnd = (mcc == 2) & (r.random(n) < 0.4)              # round fuel amounts
        amt[rnd] = np.maximum(10, np.round(amt[rnd], -1))
        merch = np.where(r.random(n) < 0.75, r.choice(favs[c], n),
                         r.integers(0, 1000, n)) + 1000 * mcc
        ip = home[c] + r.normal(0, 8, 2)                   # the customer's usual IP location
        xy = np.where(online[:, None], ip, home[c] + r.normal(0, 8, (n, 2)))  # stores vary, IPs don't
        vpn = online & (r.random(n) < 0.2)                 # IP far from home
        xy[vpn] = r.uniform(0, 2000, (vpn.sum(), 2))
        f = [_rows(t, amt, online, mcc, merch, xy)]
        for _ in range(r.poisson(1.5)):                     # legit trips, often by air
            s, d = r.uniform(0, days) * DAY, r.uniform(2, 7) * DAY
            ang, km = r.uniform(0, 2 * np.pi), r.uniform(100, 1500)
            dest = home[c] + km * np.array([np.cos(ang), np.sin(ang)])
            on_trip = (f[0].t >= s) & (f[0].t < s + d) & ~f[0].online
            f[0].loc[on_trip, ["x", "y"]] = dest + r.normal(0, 10, (on_trip.sum(), 2))
            if r.random() < 0.5:                            # big trip purchases
                k = r.integers(1, 4)
                f.append(_rows(s + r.uniform(0, d, k), np.exp(mu[c] + r.uniform(1, 2.5, k)),
                               np.zeros(k, bool), r.choice([3, 6, 7, 8], k),
                               r.integers(0, 1000, k) + 3000, dest + r.normal(0, 10, (k, 2))))
        for _ in range(r.poisson(0.5)):                     # legit shopping sprees
            k, s = r.integers(3, 7), r.uniform(0, days) * DAY
            f.append(_rows(s + np.cumsum(r.uniform(3, 20, k)),
                           np.exp(mu[c] + r.uniform(0, 1.5, k)), np.zeros(k, bool),
                           r.choice([1, 3, 6], k), r.integers(0, 1000, k) + 3000,
                           home[c] + r.normal(0, 3, (k, 2))))
        if gamer[c]:                                        # tiny digital bursts
            for _ in range(r.poisson(3)):
                k, s = r.integers(2, 7), r.uniform(0, days) * DAY
                f.append(_rows(s + np.cumsum(r.uniform(1, 10, k)), r.uniform(0.99, 4.99, k),
                               np.ones(k, bool), np.full(k, 5),
                               r.integers(0, 1000, k) + 5000, np.tile(ip, (k, 1))))
        if r.random() < 0.3:                                # small subscriptions
            m = r.integers(0, 1000)
            k = days // 30
            f.append(_rows(np.arange(k) * 30 * DAY + r.uniform(0, DAY), np.full(k, r.uniform(0.99, 9.99)),
                           np.ones(k, bool), np.full(k, 4), np.full(k, 4000 + m),
                           np.tile(ip, (k, 1))))
        L = pd.concat(f, ignore_index=True)
        if quiet[c]:                                        # legit dormancy
            g0 = r.uniform(10, 70) * DAY
            L = L[~((L.t >= g0) & (L.t < g0 + r.uniform(30, 80) * DAY))]
        pc = [r.uniform(0, days) * DAY] if r.random() < 0.15 else []
        new_phone = r.uniform(10, days) * DAY if r.random() < 0.35 else np.inf
        if c in ev_of:
            e = ev.loc[ev_of[c]]
            F = _fraud_burst(r, e.typology, t0=e.t0, mu=mu[c], home=home[c])
            t0 = F.t.min()                                  # first fraud charge
            if e.typology == "dormant_reactivation":
                L = L[~((L.t >= t0 - r.uniform(30, 80) * DAY) & (L.t < t0))]
            if e.typology == "account_takeover" and r.random() < 0.5:
                pc.append(t0 - r.uniform(0.2, 48) * 60)     # credentials changed first
            new_dev = r.random() < {"card_testing": 0.6, "account_takeover": 0.7,
                                    "dormant_reactivation": 0.6}.get(e.typology, 0)
            F["device"] = np.where(F.online, 10 * c + (7 if new_dev else 0), -1)
            F["event_id"] = e.event_id
            rows.append(F.assign(cust=c))
        L = L.assign(device=np.where(L.online, 10 * c + (L.t > new_phone), -1),
                     event_id=-1, cust=c)
        rows.append(L)
        profile_changes[c] = np.array(sorted(pc))
    tx = pd.concat(rows, ignore_index=True).sort_values(["cust", "t"], kind="stable")
    tx = tx[tx.t < days * DAY].reset_index(drop=True)
    tx["is_fraud"] = (tx.label != "legit").astype(int)
    cust = dict(home=home, age0=age0, profile_changes=profile_changes)
    return tx, ev, cust

def _fraud_burst(r, typ, t0, mu, home):
    # each typology shows only SOME of its tells, like real fraudsters do
    near = lambda: home + r.normal(0, 8, 2)                 # residential proxy
    far = lambda: r.uniform(0, 2000, 2)
    if typ == "card_testing":            # tiny online charges at new merchants
        k = r.integers(3, 9)
        t = t0 + np.cumsum(r.uniform(1, 20, k))
        amt = r.uniform(0.5, 4.99, k)
        mcc = r.choice([4, 5], k)
        if r.random() < 0.6:             # then the cash-out
            m = r.integers(1, 3)
            t = np.r_[t, t[-1] + np.cumsum(r.uniform(30, 180, m))]
            amt = np.r_[amt, r.uniform(60, 300, m)]
            mcc = np.r_[mcc, r.choice([3, 5, 6], m)]
        online = np.ones(len(t), bool)
        xy = np.tile(far() if r.random() < 0.5 else near(), (len(t), 1))
    elif typ == "account_takeover":      # new device, larger online buys, odd hours
        k = r.integers(1, 4)
        start = (t0 // DAY) * DAY + (r.uniform(0, 5) if r.random() < 0.3
                                     else r.uniform(0, 24)) * 60
        t = start + np.cumsum(r.uniform(10, 240, k))
        amt = np.exp(mu) * r.uniform(1.5, 6, k)
        mcc = r.choice([3, 5, 6, 7], k)
        online = np.ones(k, bool)
        xy = np.tile(near() if r.random() < 0.6 else far(), (k, 1))
    elif typ == "counterfeit_card":      # card-present far from home
        k = r.integers(1, 5)
        t = t0 + np.cumsum(r.uniform(15, 180, k))
        amt = np.exp(mu) * r.uniform(1, 5, k)
        mcc = r.choice([0, 2, 3, 6, 7], k)
        online = np.zeros(k, bool)
        ang, km = r.uniform(0, 2 * np.pi), r.uniform(100, 900)
        xy = home + km * np.array([np.cos(ang), np.sin(ang)]) + r.normal(0, 5, (k, 2))
    else:                                # dormant card wakes up online
        k = r.integers(1, 5)
        t = t0 + np.sort(r.uniform(0, 48 * 60, k))
        amt = np.exp(mu) * r.uniform(1, 4, k)
        mcc = r.choice([3, 4, 5, 6], k)
        online = np.ones(k, bool)
        xy = np.tile(far() if r.random() < 0.4 else near(), (k, 1))
    return _rows(t, amt, online, mcc, r.integers(0, 1000, len(t)) + 1000 * mcc, xy, typ)


# ---------------------------------------------------- fraud features
H, D = 60.0, DAY

def build_features(tx, cust, mcc_med):
    t, c = tx.t.to_numpy(), tx.cust.to_numpy()
    amt, mcc, merch = tx.amount.to_numpy(), tx.mcc.to_numpy(), tx.merchant.to_numpy()
    x, y, onl = tx.x.to_numpy(), tx.y.to_numpy(), tx.online.to_numpy()
    dev = tx.device.to_numpy()
    n, idx = len(tx), np.arange(len(tx))
    key = c * 1e7 + t                       # sorted by (customer, time)
    lo = lambda w: np.searchsorted(key, key - w, "right")   # window (t-w, t]
    lop = lambda w: np.searchsorted(key, key - w, "left")   # prior [t-w, t)
    cs = lambda a: np.r_[0, np.cumsum(a)]
    win_sum = lambda a, w: cs(a)[idx + 1] - cs(a)[lo(w)]
    prior_sum = lambda a, w: cs(a)[idx] - cs(a)[lop(w)]
    first = np.r_[True, c[1:] != c[:-1]]
    prev = np.where(first, -1, idx - 1)
    home = cust["home"][c]
    d_home = np.hypot(x - home[:, 0], y - home[:, 1])
    gap_min = np.where(first, np.nan, t - t[prev])
    d_last = np.where(first, np.nan, np.hypot(x - x[prev], y - y[prev]))
    hour = (t % D) // 60
    ones = np.ones(n)
    n90, n30 = prior_sum(ones, 90 * D), prior_sum(ones, 30 * D)
    s90 = prior_sum(amt, 90 * D)
    far_cp = (~onl) & (d_home > 200)
    # per-row loop for the order statistics and distinct counts
    lo90, lo30, lo1h, lo24, lo7 = lop(90 * D), lop(30 * D), lo(H), lo(D), lo(7 * D)
    lo_w = {w: lo(w) for w in (10, H, D, 7 * D, 30 * D)}
    cell = (x // 100) * 100 + (y // 100)
    day_i = (t // D).astype(int)
    E = {k: np.full(n, np.nan) for k in
         ["p50", "p95", "rare", "dm1h", "reg24", "act90", "gap90", "dev30"]}
    K = {f"amount_max_{k}": np.full(n, np.nan) for k in ["10m", "1h", "24h", "7d", "30d"]}
    K.update({f"merchants_nunique_{k}": np.zeros(n) for k in ["1h", "24h", "7d"]})
    K["devices_nunique_24h"] = np.zeros(n)
    for i in range(n):
        a, b = lo90[i], i
        if b - a >= 3:
            E["p50"][i], E["p95"][i] = np.percentile(amt[a:b], [50, 95])
        if b - a >= 5:
            dh = np.abs(hour[a:b] - hour[i]); dh = np.minimum(dh, 24 - dh)
            E["rare"][i] = 1 - (dh <= 2).mean()
        E["dm1h"][i] = len(set(merch[lo1h[i]:i + 1]))
        E["reg24"][i] = len(set(cell[lo24[i]:i + 1]))
        E["act90"][i] = len(set(day_i[a:b]))
        if i - a >= 1:
            E["gap90"][i] = np.diff(t[a:i + 1]).max() / D
        d30 = dev[lo30[i]:i + 1]
        E["dev30"][i] = len(set(d30[d30 >= 0]))
        for k, w in zip(["10m", "1h", "24h", "7d", "30d"], (10, H, D, 7 * D, 30 * D)):
            K[f"amount_max_{k}"][i] = amt[lo_w[w][i]:i + 1].max()
        K["merchants_nunique_1h"][i] = E["dm1h"][i]
        K["merchants_nunique_24h"][i] = len(set(merch[lo24[i]:i + 1]))
        K["merchants_nunique_7d"][i] = len(set(merch[lo7[i]:i + 1]))
        d24 = dev[lo24[i]:i + 1]
        K["devices_nunique_24h"][i] = len(set(d24[d24 >= 0]))
    # first-seen flags and profile changes
    g = tx[["cust", "device", "merchant", "mcc"]]
    seen_dev_t = tx.groupby(["cust", "device"]).t.transform("min").to_numpy()
    new_device = onl & ~g.duplicated(["cust", "device"]).to_numpy()
    pc_last = np.full(n, np.nan)
    for cc, times in cust["profile_changes"].items():
        if len(times):
            rows = np.where(c == cc)[0]
            j = np.searchsorted(times, t[rows], "right") - 1
            pc_last[rows] = np.where(j >= 0, times[np.maximum(j, 0)], np.nan)
    days_pc = np.where(np.isnan(pc_last), 999.0, (t - pc_last) / D)
    mkey = merch * 1e7 + t
    order = np.argsort(mkey, kind="stable")
    ms = mkey[order]
    pop = np.empty(n); pop[order] = np.arange(n) - np.searchsorted(ms, ms - 30 * D, "left")
    ckey = (c * 10 + mcc) * 1e7 + t
    o2 = np.argsort(ckey, kind="stable"); s2 = ckey[o2]
    same_mcc = np.empty(n); same_mcc[o2] = np.arange(n) - np.searchsorted(s2, s2 - 90 * D, "left")
    safe = lambda a, b: np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)

    expert = pd.DataFrame({
        # amount
        "amount_usd": amt,
        "amount_vs_cust_median": safe(amt, E["p50"]),
        "amount_vs_cust_p95": safe(amt, E["p95"]),
        "amount_is_tiny": (amt < 5).astype(int),
        "amount_is_round": (amt % 10 == 0).astype(int),
        "amount_vs_mcc_median": amt / mcc_med[mcc],
        # velocity
        "n_txn_10m": idx - lo(10) + 1,
        "n_txn_1h": idx - lo(H) + 1,
        "n_txn_24h": idx - lo(D) + 1,
        "n_distinct_merchants_1h": E["dm1h"],
        "n_tiny_txn_1h": win_sum((amt < 5).astype(float), H),
        "spend_24h_vs_cust_daily": safe(win_sum(amt, D), s90 / 90),
        "minutes_since_last_txn": gap_min,
        "n_txn_7d_vs_cust_weekly": safe(idx - lo(7 * D) + 1, n90 / 90 * 7),
        # geography
        "distance_from_home_km": d_home,
        "distance_from_last_txn_km": d_last,
        "implied_speed_kmh": d_last / np.maximum(gap_min / 60, 1 / 60),
        "is_card_present": (~onl).astype(int),
        "n_regions_24h": E["reg24"],
        "cust_travel_share_90d": safe(prior_sum(far_cp.astype(float), 90 * D),
                                      prior_sum((~onl).astype(float), 90 * D)),
        # time
        "hour_of_day": hour,
        "is_night": (hour < 5).astype(int),
        "hour_rarity_for_cust": E["rare"],
        "is_weekend": ((t // D) % 7 >= 5).astype(int),
        "longest_gap_days_90d": E["gap90"],
        # device and account
        "is_online": onl.astype(int),
        "new_device": new_device.astype(int),
        "device_age_days": np.where(onl, (t - seen_dev_t) / D, np.nan),
        "n_devices_30d": E["dev30"],
        "days_since_profile_change": np.minimum(days_pc, 999),
        "account_age_days": cust["age0"][c] + t / D,
        "profile_change_last_7d": (days_pc <= 7).astype(int),
        # merchant
        "mcc_risk_level": MCC_RISK[mcc],
        "new_merchant_for_cust": (~g.duplicated(["cust", "merchant"])).astype(int).to_numpy(),
        "new_mcc_for_cust": (~g.duplicated(["cust", "mcc"])).astype(int).to_numpy(),
        "merchant_popularity_30d": pop,
        "is_digital_goods": (mcc == 5).astype(int),
        "mcc_share_for_cust": safe(same_mcc, n90),
        # customer baseline
        "cust_txn_count_30d": n30,
        "cust_active_days_90d": E["act90"],
        "cust_median_amount_90d": E["p50"],
        "cust_online_share_90d": safe(prior_sum(onl.astype(float), 90 * D), n90),
        "cust_tenure_txn_count": idx - np.maximum.accumulate(np.where(first, idx, 0)),
    })
    # kitchen sink: the same raw tables (transactions, devices, profile events,
    # home address), aggregated blindly over fixed windows
    pc_cnt = lambda w: np.array([np.sum((pt > tt - w) & (pt <= tt)) if len(pt) else 0
                                 for pt, tt in zip(map(cust["profile_changes"].get, c), t)])
    kitchen = {"amount": amt, "hour": hour, "x": x, "y": y, "home_x": home[:, 0],
               "home_y": home[:, 1], "is_online": onl.astype(int), "mcc": mcc,
               "account_age_days": cust["age0"][c] + t / D}
    for k, w in zip(["10m", "1h", "24h", "7d", "30d"], (10, H, D, 7 * D, 30 * D)):
        cnt, sm = idx - lo(w) + 1, win_sum(amt, w)
        kitchen[f"amount_count_{k}"] = cnt
        kitchen[f"amount_sum_{k}"] = sm
        kitchen[f"amount_mean_{k}"] = sm / cnt
        kitchen[f"amount_max_{k}"] = K[f"amount_max_{k}"]
    c30 = idx - lo(30 * D) + 1
    kitchen.update({"x_mean_30d": win_sum(x, 30 * D) / c30,
                    "y_mean_30d": win_sum(y, 30 * D) / c30,
                    "merchants_nunique_1h": K["merchants_nunique_1h"],
                    "merchants_nunique_24h": K["merchants_nunique_24h"],
                    "merchants_nunique_7d": K["merchants_nunique_7d"],
                    "devices_nunique_24h": K["devices_nunique_24h"],
                    "devices_nunique_30d": E["dev30"],
                    "profile_changes_1d": pc_cnt(D), "profile_changes_7d": pc_cnt(7 * D),
                    "profile_changes_30d": pc_cnt(30 * D),
                    "hour_mean_30d": win_sum(hour, 30 * D) / c30,
                    "minutes_since_last_txn": gap_min,
                    "online_mean_30d": win_sum(onl.astype(float), 30 * D) / c30,
                    "amount_std_30d": np.sqrt(np.maximum(win_sum(amt ** 2, 30 * D) / c30
                                                         - (win_sum(amt, 30 * D) / c30) ** 2, 0))})
    return expert, pd.DataFrame(kitchen)


# ------------------------------------------- summary-level desks (AML, ...)
def draw(r, d, k):
    kind, *a = d
    f = {"poisson": lambda: r.poisson(a[0], k), "poisson1": lambda: 1 + r.poisson(a[0], k),
         "lognorm": lambda: np.exp(r.normal(a[0], a[1], k)), "beta": lambda: r.beta(*a, k),
         "uniform": lambda: r.uniform(*a, k), "int": lambda: r.integers(a[0], a[1] + 1, k),
         "bern": lambda: r.random(k) < a[0]}
    return f[kind]().astype(float)

def simulate_desk(seed, n, segments, spec, typologies, n_typ):
    r = np.random.default_rng(seed)
    seg = r.choice(list(segments), n, p=list(segments.values()))
    X = pd.DataFrame(index=range(n))
    for feat, by in spec.items():
        X[feat] = 0.0
        for s in np.unique(seg):
            m = seg == s
            X.loc[m, feat] = draw(r, by.get(s, by["*"]), m.sum())
    label = np.array(["legit"] * n, dtype=object)
    bad = r.permutation(n)[: n_typ * len(typologies)]
    for j, (typ, effects) in enumerate(typologies.items()):
        rows = bad[j * n_typ:(j + 1) * n_typ]
        label[rows] = typ
        for feat, op, d, p in effects:
            hit = rows[r.random(len(rows)) < p]
            v, new = X.loc[hit, feat].to_numpy(), draw(r, d, len(hit))
            X.loc[hit, feat] = {"set": new, "add": v + new, "mul": v * new}[op]
    split = r.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])
    return X, label, seg, split

AML_SEGMENTS = {"retail": .70, "cash_business": .10, "payroll": .05, "gig_worker": .08,
                "remitter": .05, "seasonal": .02}
AML_SPEC = {
    "cash_deposits_30d": {"*": ("poisson", .6), "cash_business": ("poisson", 12), "seasonal": ("poisson", 3)},
    "cash_deposit_total_vs_income": {"*": ("lognorm", -3, 1), "cash_business": ("lognorm", 0, .5),
                                     "seasonal": ("lognorm", -1, .7)},
    "share_deposits_9k_10k": {"*": ("beta", .3, 40), "cash_business": ("beta", .8, 12)},
    "n_deposit_branches_30d": {"*": ("poisson1", .3), "cash_business": ("poisson1", .8)},
    "inbound_vs_12m_avg": {"*": ("lognorm", 0, .3), "seasonal": ("lognorm", .9, .5)},
    "outbound_vs_12m_avg": {"*": ("lognorm", 0, .3), "seasonal": ("lognorm", .8, .5)},
    "balance_retention_ratio": {"*": ("beta", 3, 7), "gig_worker": ("beta", 1.2, 9), "payroll": ("beta", 1, 25),
                                "cash_business": ("beta", 2, 6), "remitter": ("beta", 1.5, 6)},
    "median_hold_time_hours": {"*": ("lognorm", 5, .8), "gig_worker": ("lognorm", 3.4, .8),
                               "payroll": ("lognorm", 3, .6), "remitter": ("lognorm", 4, .8)},
    "n_distinct_senders_30d": {"*": ("poisson", 2), "gig_worker": ("poisson", 14),
                               "cash_business": ("poisson", 4), "seasonal": ("poisson", 3)},
    "n_distinct_receivers_30d": {"*": ("poisson", 6), "payroll": ("poisson", 40), "cash_business": ("poisson", 10)},
    "new_counterparty_share": {"*": ("beta", 2, 8), "gig_worker": ("beta", 3, 4)},
    "p2p_inbound_share": {"*": ("beta", 1, 6), "gig_worker": ("beta", 5, 2)},
    "wire_out_high_risk_share": {"*": ("beta", .2, 40), "remitter": ("beta", 3, 4)},
    "round_amount_share": {"*": ("beta", 2, 10), "remitter": ("beta", 4, 6), "cash_business": ("beta", 3, 8)},
    "months_active_last_12": {"*": ("int", 10, 12), "seasonal": ("int", 2, 5), "gig_worker": ("int", 4, 12)},
    "activity_vs_12m_avg": {"*": ("lognorm", 0, .3), "seasonal": ("lognorm", 1.1, .4)},
    "account_age_days": {"*": ("uniform", 60, 5000), "gig_worker": ("uniform", 20, 900)},
    "n_accounts_sharing_device": {"*": ("poisson1", .05), "gig_worker": ("poisson1", .4)},
    "cash_withdrawal_share_outflow": {"*": ("beta", 1, 9), "cash_business": ("beta", 1.5, 6)},
    "avg_txn_amount": {"*": ("lognorm", 4, .7), "cash_business": ("lognorm", 5, .6), "payroll": ("lognorm", 6.5, .5)},
    "txn_count_30d": {"*": ("poisson", 40), "cash_business": ("poisson", 120), "payroll": ("poisson", 60),
                      "gig_worker": ("poisson", 55), "remitter": ("poisson", 25)},
    "intl_wire_count_30d": {"*": ("poisson", .05), "remitter": ("poisson", 3)},
    "night_activity_share": {"*": ("beta", 1, 12), "gig_worker": ("beta", 2, 8)},
    "declared_income_band": {"*": ("int", 1, 5), "gig_worker": ("int", 1, 2), "payroll": ("int", 4, 5),
                             "remitter": ("int", 1, 3)},
}
AML_TYPOLOGIES = {
    "structuring": [("cash_deposits_30d", "add", ("poisson", 6), .8),
                    ("share_deposits_9k_10k", "set", ("beta", 5, 4), .75),
                    ("n_deposit_branches_30d", "add", ("poisson", 2), .6),
                    ("cash_deposit_total_vs_income", "mul", ("uniform", 2, 8), .7)],
    "mule_pass_through": [("balance_retention_ratio", "set", ("beta", 1, 30), .8),
                          ("median_hold_time_hours", "set", ("lognorm", 1.5, .8), .75),
                          ("n_distinct_senders_30d", "add", ("poisson", 8), .6),
                          ("new_counterparty_share", "set", ("beta", 6, 3), .6),
                          ("account_age_days", "set", ("uniform", 15, 250), .5),
                          ("n_accounts_sharing_device", "add", ("poisson", 1.5), .4)],
    "high_risk_corridor": [("wire_out_high_risk_share", "set", ("beta", 5, 4), .8),
                           ("round_amount_share", "set", ("beta", 6, 4), .6),
                           ("intl_wire_count_30d", "add", ("poisson", 4), .7),
                           ("outbound_vs_12m_avg", "mul", ("uniform", 1.5, 5), .5)],
    "dormant_burst": [("months_active_last_12", "set", ("int", 1, 3), .8),
                      ("activity_vs_12m_avg", "mul", ("uniform", 3, 12), .75),
                      ("inbound_vs_12m_avg", "mul", ("uniform", 2, 8), .6),
                      ("outbound_vs_12m_avg", "mul", ("uniform", 2, 8), .6)],
}

def simulate_aml(seed, n=25000, n_typ=400):
    """The AML desk: 25,000 accounts, 6 legit segments, 4 laundering typologies."""
    X, label, seg, split = simulate_desk(seed, n, AML_SEGMENTS, AML_SPEC, AML_TYPOLOGIES, n_typ)
    X.loc[X.cash_deposits_30d == 0, "share_deposits_9k_10k"] = 0.0   # no deposits, no share
    X["share_deposits_9k_10k"] = X.share_deposits_9k_10k.clip(0, 1)
    X["months_active_last_12"] = X.months_active_last_12.clip(1, 12)
    return X, label, seg, split

CLAIMS_SPEC = {
    "claim_amount_vs_peer_median": {"*": ("lognorm", 0, .5), "high_value": ("lognorm", .5, .5)},
    "days_policy_start_to_claim": {"*": ("uniform", 30, 1500)},
    "report_delay_days": {"*": ("lognorm", 1, .8)},
    "prior_claims_3y": {"*": ("poisson", .3)},
    "parties_in_other_claims": {"*": ("poisson", .05)},
    "soft_tissue_injury": {"*": ("bern", .1)},
    "police_report_filed": {"*": ("bern", .7)},
    "shop_flagged_claim_rate": {"*": ("beta", 1, 30)},
}
CLAIMS_TYP = {
    "early_claim": [("days_policy_start_to_claim", "set", ("uniform", 1, 30), .9),
                    ("claim_amount_vs_peer_median", "mul", ("uniform", 1, 2), .5)],
    "staged_accident": [("parties_in_other_claims", "add", ("poisson", 2), .8),
                        ("soft_tissue_injury", "set", ("bern", 1), .8),
                        ("police_report_filed", "set", ("bern", 0), .7)],
    "inflated_claim": [("claim_amount_vs_peer_median", "mul", ("uniform", 2.5, 6), .9),
                       ("shop_flagged_claim_rate", "set", ("beta", 4, 10), .7)],
}
SECOPS_SPEC = {
    "failed_logins_1h": {"*": ("poisson", .3), "server": ("poisson", 1)},
    "distinct_usernames_tried_1h": {"*": ("poisson1", .05)},
    "bytes_out_vs_baseline": {"*": ("lognorm", 0, .4)},
    "new_external_destination": {"*": ("bern", .05)},
    "beacon_regularity": {"*": ("beta", 1, 8), "server": ("beta", 2, 6)},
    "off_hours_share": {"*": ("beta", 1, 8), "server": ("beta", 3, 3)},
    "process_rarity": {"*": ("beta", 1, 10)},
    "dest_domain_age_days": {"*": ("lognorm", 7, 1)},
}
SECOPS_TYP = {
    "password_spraying": [("failed_logins_1h", "add", ("poisson", 25), .9),
                          ("distinct_usernames_tried_1h", "add", ("poisson", 15), .85)],
    "data_exfiltration": [("bytes_out_vs_baseline", "mul", ("uniform", 8, 40), .9),
                          ("new_external_destination", "set", ("bern", 1), .8),
                          ("off_hours_share", "set", ("beta", 5, 2), .6)],
    "c2_beaconing": [("beacon_regularity", "set", ("beta", 12, 1.5), .9),
                     ("dest_domain_age_days", "set", ("uniform", 1, 30), .8),
                     ("process_rarity", "set", ("beta", 5, 3), .6)],
}

# --------------------------------------------------------------- credit
def load_credit():
    """UCI 'Default of Credit Card Clients' (30,000 rows, CC BY 4.0) via OpenML."""
    from sklearn.datasets import fetch_openml
    raw = fetch_openml(data_id=42477, as_frame=True).frame
    PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY
            + [f"BILL_AMT{k}" for k in range(1, 7)] + [f"PAY_AMT{k}" for k in range(1, 7)])
    raw = raw.rename(columns=dict(zip([f"x{k}" for k in range(1, 24)], COLS))).astype(float)
    y = raw.pop("y").astype(int).to_numpy()            # 1 = defaulted next month
    return raw[COLS], y

# -------------------------------------------------------------- helpers
def percentiles(X_ref):
    """Mid-rank percentile of a value inside the training population."""
    ref = {c: np.sort(X_ref[c].dropna().to_numpy()) for c in X_ref}
    def pct(c, v):
        a = ref[c]
        if pd.isna(v) or len(a) == 0:
            return np.nan
        return 50 * (np.searchsorted(a, v, "left") + np.searchsorted(a, v, "right")) / len(a)
    return pct

def vcat(p):                                       # ground truth for "higher/lower than normal"
    return None if pd.isna(p) else "higher" if p > 75 else "lower" if p < 25 else "typical"

def fmt(v):
    if pd.isna(v):
        return "missing"
    if abs(v) >= 1000 or float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:.3g}"

def ordinal(p):
    n = int(round(p))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])

def alert_schema(typologies):
    """The JSON the narrator must return (structured output): typology, cited reasons, two texts."""
    reason = {"type": "object", "additionalProperties": False,
              "required": ["feature", "effect", "value_vs_normal"],
              "properties": {"feature": {"type": "string"},
                             "effect": {"type": "string", "enum": ["raises_risk", "lowers_risk"]},
                             "value_vs_normal": {"type": "string", "enum":
                                                 ["higher", "lower", "typical", "unknown"]}}}
    return {"type": "object", "additionalProperties": False,
            "required": ["typology", "reasons", "analyst_rationale", "customer_message"],
            "properties": {"typology": {"type": "string", "enum": typologies},
                           "reasons": {"type": "array", "items": reason},
                           "analyst_rationale": {"type": "string"},
                           "customer_message": {"type": "string"}}}

def summarize(df):
    """Per condition: accuracy with a 95% bootstrap CI, how it fails, and the cost."""
    out = {}
    for cond, g in df.groupby("cond", sort=False):
        lo, hi = boot_ci(g.correct)
        out[cond] = {"n": len(g), "correct": f"{g.correct.mean():.0%}",
                     "ci95": f"[{lo:.0%}, {hi:.0%}]", "unclear": f"{g.unclear.mean():.0%}",
                     "sure_wrong": f"{(~g.correct & ~g.unclear).mean():.0%}",
                     "halluc": f"{g.hallucinated.mean():.0%}",
                     "dir_err": f"{g.direction_error.mean():.0%}",
                     "value_err": f"{g.value_error.mean():.0%}",
                     "usd_per_1k": f"{(0.10 * g.inp.mean() + 0.50 * g.out.mean()) / 1000:.3f}"}
    return pd.DataFrame(out).T

def summarize_full(df):            # v1 table, used by the repo-only scripts
    rows = []
    for cond, g in df.groupby("cond", sort=False):
        lo_, hi_ = boot_ci(g.correct)
        rows.append(dict(cond=cond, n=len(g), failed=int(g.failed.sum()),
                         correct=f"{g.correct.mean():.0%}",
                         ci95=f"[{lo_:.0%}, {hi_:.0%}]", unclear=f"{g.unclear.mean():.0%}",
                         halluc=f"{g.hallucinated.mean():.0%}",
                         dir_err=f"{g.direction_error.mean():.0%}",
                         value_err=f"{g.value_error.mean():.0%}",
                         faithful=f"{g.faithful.mean():.0%}",
                         usd_per_1k=f"{(g.inp.mean() * 0.10 + g.out.mean() * 0.50) / 1000:.3f}"))
    return pd.DataFrame(rows).set_index("cond")


# -------------------------------------------------------------- plotting
def bars(ax, labels, vals, ci=None, colors=None, pct=True, ylim=1.12, ylabel="", fs=12):
    """Labelled bar chart: values on top (above the 95% CI whiskers, if given)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    xs, vals = np.arange(len(labels)), np.asarray(vals, float)
    ax.bar(xs, vals, color=colors)
    tops = vals
    if ci is not None:
        ci = np.asarray(ci, float)
        ax.errorbar(xs, vals, yerr=[vals - ci[:, 0], ci[:, 1] - vals], fmt="none",
                    ecolor=plt.rcParams["text.color"], capsize=4, lw=1)
        tops = ci[:, 1]
    for x, v, t in zip(xs, vals, tops):
        ax.text(x, t + 0.03 * ylim, f"{v:.0%}" if pct else f"{v:.2f}", ha="center",
                fontsize=fs + 2, fontweight="bold")
    ax.set_xticks(xs, labels, fontsize=fs)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    if pct:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    return xs

def paired_bars(ax, groups, series, colors, ylim=1.12, ylabel="", fs=12):
    """Side-by-side bars: series = {label: one value per group}, % axis, values on top."""
    from matplotlib.ticker import PercentFormatter
    xs, w = np.arange(len(groups)), 0.8 / len(series)
    for j, ((lab, vals), col) in enumerate(zip(series.items(), colors)):
        x = xs + (j - (len(series) - 1) / 2) * w
        ax.bar(x, vals, w, color=col, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", fontsize=fs, fontweight="bold")
    ax.set_xticks(xs, groups, fontsize=fs)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

def load_env(name=".env"):
    """Read KEY=VALUE lines from .env in this folder or a parent. Handles the UTF-16
    files that PowerShell's `>` writes. Never prints values."""
    import os
    from pathlib import Path
    for d in [Path.cwd(), *Path.cwd().parents][:4]:
        p = d / name
        if p.exists():
            b = p.read_bytes()
            enc = "utf-16" if b[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            for line in b.decode(enc).splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(p)

# ------------------------------------------------- plot theme (dark)
THEME = "dark"
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7"]
GREY = "#BBBBBB"

def apply_theme(theme=THEME):
    fg, bg, grid = (("#FFFFFF", "#121212", "#3a3a3a") if theme == "dark"
                    else ("#111111", "#FFFFFF", "#dddddd"))
    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg,
        "text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": fg,
        "xtick.color": fg, "ytick.color": fg, "grid.color": grid,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "font.size": 13, "axes.titlesize": 15, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
    })

def save(fig, path):
    """Save to images/; in a notebook, also show the figure inline."""
    fig.savefig(path, bbox_inches="tight")
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)

