import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ADMIN_DIR = DATA_DIR / "admin"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_PATH = MODELS_DIR / "artifacts.json"

CV_SAMPLES = 12000
DL_SAMPLES = 12000
NLP_SAMPLES = 8000


def _ensure_dirs():
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _fit_scaler(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _apply_scaler(x, mean, std):
    return (x - mean) / std


def _admin_csv(name):
    return ADMIN_DIR / f"{name}.csv"


def load_admin_rows(name):
    import pandas as pd
    path = _admin_csv(name)
    if not path.exists():
        return None
    return pd.read_csv(path)


def save_admin_rows(name, df):
    import pandas as pd
    _ensure_dirs()
    df.to_csv(_admin_csv(name), index=False)


def append_admin_row(name, row_dict, labeled=True):
    import pandas as pd
    suffix = "labeled" if labeled else "unlabeled"
    df = load_admin_rows(f"{name}_{suffix}")
    if df is None:
        df = pd.DataFrame([row_dict])
    else:
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
    save_admin_rows(f"{name}_{suffix}", df)
    return df


def remove_admin_rows(name, indices, labeled=False):
    import pandas as pd
    suffix = "labeled" if labeled else "unlabeled"
    df = load_admin_rows(f"{name}_{suffix}")
    if df is None:
        return None
    df = df.drop(index=list(indices)).reset_index(drop=True)
    save_admin_rows(f"{name}_{suffix}", df)
    return df


def encode_text(text, word_index):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokens = text.lower().split()
    ids = [word_index.get(w, 2) + 3 for w in tokens]
    ids = [i if i < 5000 else 2 for i in ids]
    return pad_sequences([ids], maxlen=200, padding="pre", truncating="pre",
                         value=0).astype(np.int32)


def load_datasets(include_admin=True):
    """Return (data, artifacts). data[head] = dict(train=(x,y), test=(x,y))."""
    from tensorflow.keras.datasets import fashion_mnist, mnist, imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    _ensure_dirs()

    (xf_tr, yf_tr), (xf_te, yf_te) = fashion_mnist.load_data()
    (xm_tr, ym_tr), (xm_te, ym_te) = mnist.load_data()
    rng = np.random.RandomState(0)
    fi = rng.permutation(len(xf_tr))[:CV_SAMPLES]
    mi = rng.permutation(len(xm_tr))[:DL_SAMPLES]
    fti = rng.permutation(len(xf_te))[:3000]
    mti = rng.permutation(len(xm_te))[:3000]

    def img(x, y, idx):
        x = x[idx].astype(np.float32)[..., None] / 255.0
        return x, y[idx].astype(np.int32)

    data = {
        "cv": {"train": img(xf_tr, yf_tr, fi), "test": img(xf_te, yf_te, fti)},
        "dl": {"train": img(xm_tr, ym_tr, mi), "test": img(xm_te, ym_te, mti)},
    }

    (xt_tr, yt_tr), (xt_te, yt_te) = imdb.load_data(num_words=5000)
    xt_tr = pad_sequences(xt_tr, maxlen=200, padding="pre", truncating="pre")
    xt_te = pad_sequences(xt_te, maxlen=200, padding="pre", truncating="pre")
    ni = rng.permutation(len(xt_tr))[:NLP_SAMPLES]
    nti = rng.permutation(len(xt_te))[:2000]
    data["nlp"] = {
        "train": (xt_tr[ni].astype(np.int32), yt_tr[ni].astype(np.int32)),
        "test": (xt_te[nti].astype(np.int32), yt_te[nti].astype(np.int32)),
    }

    cancer = load_breast_cancer()
    xi = rng.permutation(len(cancer.data))[:400]
    xti = rng.permutation(len(cancer.data))[400:]
    xtab, ytab = cancer.data, cancer.target.astype(np.int32)
    mean, std = _fit_scaler(xtab[xi])
    data["ml"] = {
        "train": (_apply_scaler(xtab[xi], mean, std).astype(np.float32), ytab[xi]),
        "test": (_apply_scaler(xtab[xti], mean, std).astype(np.float32), ytab[xti]),
    }

    tips = _load_tips()
    day_map = {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}
    time_map = {"Lunch": 0, "Dinner": 1}
    feats = np.column_stack([
        tips["total_bill"].values,
        tips["size"].values,
        tips["day"].map(day_map).fillna(0).values,
        tips["time"].map(time_map).fillna(0).values,
    ]).astype(np.float32)
    target = tips["tip"].values.astype(np.float32)
    n = len(feats)
    di = rng.permutation(n)
    split = int(n * 0.75)
    tr, te = di[:split], di[split:]
    dsm, dss = _fit_scaler(feats[tr])
    data["ds"] = {
        "train": (_apply_scaler(feats[tr], dsm, dss), target[tr]),
        "test": (_apply_scaler(feats[te], dsm, dss), target[te]),
    }

    artifacts = {
        "tab_mean": mean.tolist(),
        "tab_std": std.tolist(),
        "ds_mean": dsm.tolist(),
        "ds_std": dss.tolist(),
        "word_index": _top_word_index(),
        "cancer_feature_names": list(cancer.feature_names),
        "tips_day_map": day_map,
        "tips_time_map": time_map,
    }

    if include_admin:
        _merge_admin(data, artifacts)

    return data, artifacts


def _top_word_index():
    from tensorflow.keras.datasets import imdb
    wi = imdb.get_word_index()
    return {w: i for w, i in wi.items() if i + 3 < 5000}


def _merge_admin(data, artifacts):
    """Merge admin LABELED rows from the database into training data.
    Unlabeled rows are for inference/testing only and never enter training."""
    import json as _json
    import sys
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import db as adb
    except Exception:
        return

    def rows(domain):
        return adb.fetch_samples(domain, labeled=True)

    nlp_rows = rows("nlp")
    if nlp_rows:
        xs = np.vstack([encode_text(_json.loads(r.payload)["text"],
                                    artifacts["word_index"]) for r in nlp_rows])
        ys = np.array([int(r.label) for r in nlp_rows], np.int32)
        data["nlp"]["train"] = (
            np.concatenate([data["nlp"]["train"][0], xs]),
            np.concatenate([data["nlp"]["train"][1], ys]),
        )

    ml_rows = rows("ml")
    if ml_rows:
        xs = _apply_scaler(
            np.array([_json.loads(r.payload)["features"] for r in ml_rows],
                     dtype=np.float32),
            np.array(artifacts["tab_mean"]), np.array(artifacts["tab_std"]))
        ys = np.array([int(r.label) for r in ml_rows], np.int32)
        data["ml"]["train"] = (
            np.concatenate([data["ml"]["train"][0], xs]),
            np.concatenate([data["ml"]["train"][1], ys]),
        )

    for head in ("cv", "dl"):
        img_rows = rows(head)
        if img_rows:
            xs = np.array([_json.loads(r.payload)["pixels"] for r in img_rows],
                          dtype=np.float32).reshape(-1, 28, 28, 1)
            ys = np.array([int(r.label) for r in img_rows], np.int32)
            data[head]["train"] = (
                np.concatenate([data[head]["train"][0], xs]),
                np.concatenate([data[head]["train"][1], ys]),
            )

    ds_rows = rows("ds")
    if ds_rows:
        feats, targets = [], []
        for r in ds_rows:
            p = _json.loads(r.payload)
            feats.append([p["total_bill"], p["size"], p["day"], p["time"]])
            targets.append(float(r.target if r.target is not None else r.label))
        xs = _apply_scaler(np.array(feats, dtype=np.float32),
                           np.array(artifacts["ds_mean"]),
                           np.array(artifacts["ds_std"]))
        data["ds"]["train"] = (
            np.concatenate([data["ds"]["train"][0], xs]),
            np.concatenate([data["ds"]["train"][1],
                            np.array(targets, np.float32)]),
        )


def _load_tips():
    import pandas as pd
    path = DATA_DIR / "tips.csv"
    if not path.exists():
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            rng = np.random.RandomState(1)
            n = 244
            total = rng.uniform(10, 50, n)
            size = rng.randint(1, 6, n)
            day = rng.choice(["Thur", "Fri", "Sat", "Sun"], n)
            time = rng.choice(["Lunch", "Dinner"], n)
            tip = total * rng.uniform(0.12, 0.25, n)
            return pd.DataFrame({"total_bill": total, "tip": tip,
                                 "size": size, "day": day, "time": time})
    return pd.read_csv(path)


def load_artifacts():
    if ARTIFACTS_PATH.exists():
        with open(ARTIFACTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_artifacts(artifacts):
    _ensure_dirs()
    with open(ARTIFACTS_PATH, "w") as f:
        json.dump(artifacts, f)
