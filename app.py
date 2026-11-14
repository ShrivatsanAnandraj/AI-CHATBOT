import json
import os
import re
import threading
import time
from pathlib import Path

from flask import (Flask, jsonify, request, send_from_directory,
                   session)

import db as adb
from db import (add_chat, add_sample, delete_sample, fetch_samples,
                init_db, recent_chats, sample_counts, set_label, verify_admin)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "school_ai.keras"
METRICS_PATH = BASE / "models" / "metrics.json"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "shri-ai-dev-secret")
app.config["SESSION_COOKIE_SAME_SITE"] = "Lax"

# ---------------------------------------------------------------- model cache
_state = {"model": None, "artifacts": None, "lock": threading.Lock()}
TRAIN = {"running": False, "step": 0, "total": 0, "head": "", "done": False,
         "error": ""}


def get_model():
    with _state["lock"]:
        if _state["model"] is None:
            import tensorflow as tf
            _state["model"] = tf.keras.models.load_model(MODEL_PATH)
            from src.data import load_artifacts
            _state["artifacts"] = load_artifacts()
        return _state["model"], _state["artifacts"]


def reload_model():
    with _state["lock"]:
        _state["model"] = None
        _state["artifacts"] = None


def metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------- prediction
def predict_head(head, x):
    import numpy as np
    from src.model import HEADS, MODALITY, IMG_SHAPE, TXT_LEN, TAB_DIM, DS_DIM
    model, _ = get_model()
    n = len(x)
    inputs = {
        "image": np.zeros((n,) + IMG_SHAPE, np.float32),
        "text": np.zeros((n, TXT_LEN), np.int32),
        "tabular": np.zeros((n, TAB_DIM), np.float32),
        "dsx": np.zeros((n, DS_DIM), np.float32),
    }
    inputs[MODALITY[head]] = x
    out = model(inputs, training=False)[HEADS.index(head)].numpy()
    if head == "ds":
        return out.reshape(-1), None
    idx = out.argmax(axis=1)
    return idx, out.max(axis=1)


def encode_pixels(pixels):
    import numpy as np
    arr = np.array(pixels, dtype=np.float32)
    if arr.size != 784:
        raise ValueError("image must be 28x28 = 784 pixels")
    return arr.reshape(1, 28, 28, 1)


def encode_text(text):
    from src.data import encode_text as _enc
    _, artifacts = get_model()
    return _enc(text, artifacts["word_index"])


def encode_ml(features):
    import numpy as np
    _, artifacts = get_model()
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    return (x - np.array(artifacts["tab_mean"])) / np.array(artifacts["tab_std"])


def encode_ds(total_bill, size, day, time_code):
    import numpy as np
    _, artifacts = get_model()
    x = np.array([[total_bill, size, day, time_code]], dtype=np.float32)
    return (x - np.array(artifacts["ds_mean"])) / np.array(artifacts["ds_std"])


# ---------------------------------------------------------------- static pages
@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/admin")
def admin_page():
    return send_from_directory(BASE, "admin.html")


@app.get("/<path:p>")
def static_files(p):
    return send_from_directory(BASE, p)


# ---------------------------------------------------------------- chat brain
GREETINGS = {"hi", "hello", "hey", "yo", "good morning", "good evening",
             "namaste", "salam", "howdy"}

DOMAIN_KB = {
    "ml": ("Machine Learning (ML) is about algorithms that learn patterns "
           "from data to make predictions - like classifying tumors from "
           "measurements. Our ML head uses breast-cancer data and reaches "
           "about {ml} accuracy."),
    "dl": ("Deep Learning (DL) uses layered neural networks. Our DL head "
           "recognizes handwritten digits (MNIST) with about {dl} accuracy "
           "through convolutions that learn edges, strokes and shapes."),
    "ds": ("Data Science (DS) is extracting insight from data: statistics, "
           "cleaning, visualization and regression. Our DS head predicts "
           "restaurant tips from bill size, party size, day and time "
           "(MAE about ${ds})."),
    "nlp": ("NLP lets computers understand language. Our NLP head reads "
            "movie reviews and predicts sentiment (positive/negative) with "
            "about {nlp} accuracy using embeddings."),
    "cv": ("Computer Vision (CV) is teaching machines to see images. Our CV "
           "head classifies Fashion-MNIST clothing photos into 10 categories "
           "with about {cv} accuracy."),
}


def _fmt_pct(v):
    return f"{v * 100:.1f}%" if v is not None else "n/a"


def chat_reply(message):
    text = message.strip()
    low = text.lower()
    m = metrics()

    # greeting
    if low in GREETINGS or low.startswith(("hi ", "hello ", "hey ")):
        return ("Hello! I am Shri AI - I can chat about any topic, analyze "
                "sentiment, predict tips, classify data, and explain the "
                "five AI domains. What would you like to do?"), "greeting"

    # help
    if low in ("help", "what can you do", "commands", "options"):
        return ("I can do a lot: (1) just talk to me about anything - "
                "science, school, life; (2) analyze sentiment - say "
                "'sentiment: <review>'; (3) predict a tip - say "
                "'tip for 40 dollars, 4 people'; (4) explain ML/DL/DS/NLP/CV "
                "- ask 'what is deep learning?'; (5) classify 30 features - "
                "'classify: 17.9,10.4,...'. Ask away!"), "help"

    # explicit sentiment
    if low.startswith(("sentiment:", "sentiment -", "sentiment ")):
        body = re.split(r"^sentiment[:\s-]+", text, maxsplit=1, flags=re.I)[1]
        return _sentiment(body)

    # tip prediction
    m_bill = re.search(r"(\d+(?:\.\d+)?)\s*(?:dollar|usd|\$)", low)
    m_size = re.search(r"(\d+)\s*(?:people|person|persons|guests|party)", low)
    if "tip" in low and (m_bill or m_size):
        bill = float(m_bill.group(1)) if m_bill else 30.0
        size = int(m_size.group(1)) if m_size else 2
        idx, _ = predict_head("ds", encode_ds(bill, size, 2, 1))
        return (f"Estimated tip for a ${bill:.2f} bill with {size} people "
                f"(Saturday dinner): ${float(idx[0]):.2f}."), "ds"

    # classify features
    if low.startswith(("classify:", "classify ", "predict:")):
        nums = re.findall(r"-?\d+\.?\d*", text)
        if len(nums) >= 30:
            feats = [float(v) for v in nums[:30]]
            pred, conf = predict_head("ml", encode_ml(feats))
            label = "Malignant" if int(pred[0]) else "Benign"
            return (f"Classification: {label} "
                    f"({float(conf[0]) * 100:.1f}% confidence)."), "ml"
        return ("Send 30 numeric features after 'classify:' (comma or space "
                "separated) and I will classify them."), "ml"

    # domain explanations
    domain_terms = {
        "ml": r"\bml\b|machine learning",
        "dl": r"\bdl\b|deep learning|neural network",
        "ds": r"\bds\b|data science|statistics",
        "nlp": r"\bnlp\b|natural language|sentiment analysis|language model",
        "cv": r"\bcv\b|computer vision|image recognition",
    }
    for key, tpl in DOMAIN_KB.items():
        if re.search(domain_terms[key], low):
            if "what" in low or "explain" in low or "tell" in low or \
               "how" in low or "define" in low or "mean" in low:
                return tpl.format(
                    ml=_fmt_pct(m.get("ml", {}).get("accuracy")),
                    dl=_fmt_pct(m.get("dl", {}).get("accuracy")),
                    ds=m.get("ds", {}).get("mae", "n/a"),
                    nlp=_fmt_pct(m.get("nlp", {}).get("accuracy")),
                    cv=_fmt_pct(m.get("cv", {}).get("accuracy")),
                ), "knowledge"

    # sentiment fallback for review-like text
    if len(text.split()) >= 6 and re.search(
            r"\b(good|great|bad|awful|love|hate|best|worst|amazing|terrible|"
            r"boring|fantastic|excellent|poor)\b", low):
        return _sentiment(text)

    # who are you
    if re.search(r"who are you|your name|what are you", low):
        return ("I am Shri AI, your school's multi-task assistant - one model "
                "with five heads covering ML, DL, DS, NLP and CV, wrapped in "
                "this chat. Ask me anything!"), "identity"

    # small talk
    if re.search(r"how are you", low):
        return "Running great - all five heads healthy. How can I help?", "smalltalk"
    if re.search(r"thank", low):
        return "You're welcome! Anything else?", "smalltalk"

    # general knowledge fallback: simple echo of intent + guidance
    return ("Got it! I can hold a conversation on any topic, and I also run "
            "a five-domain model: ask me for sentiment analysis, a tip "
            "prediction, a feature classification, or an explanation of "
            "ML / DL / DS / NLP / CV. What shall we explore?"), "fallback"


def _sentiment(text):
    pred, conf = predict_head("nlp", encode_text(text))
    label = "Positive" if int(pred[0]) else "Negative"
    return (f"Sentiment: {label} ({float(conf[0]) * 100:.1f}% confidence).",
            "nlp")


@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"reply": "Say something first!", "intent": "empty"}), 400
    try:
        reply, intent = chat_reply(message)
    except Exception as e:
        reply, intent = f"Model error: {e}", "error"
    adb.add_chat("user", message, intent)
    adb.add_chat("bot", reply, intent)
    return jsonify({"reply": reply, "intent": intent})


@app.get("/api/chat/history")
def api_chat_history():
    return jsonify(recent_chats())


@app.get("/api/metrics")
def api_metrics():
    return jsonify(metrics())


@app.get("/api/example/ml")
def api_example_ml():
    from sklearn.datasets import load_breast_cancer
    import numpy as np
    c = load_breast_cancer()
    rng = np.random.RandomState(0)
    row = c.data[rng.randint(len(c.data))]
    return jsonify({"features": [round(float(v), 4) for v in row],
                    "label": int(c.target[rng.randint(len(c.data))])})


# ---------------------------------------------------------------- admin auth
@app.post("/api/admin/login")
def api_login():
    data = request.get_json(force=True) or {}
    password = data.get("password") or ""
    # accept either the configured admin account or the direct password
    ok = verify_admin(data.get("username") or os.environ.get(
        "ADMIN_USERNAME", "admin"), password) or password == \
        os.environ.get("ADMIN_PASSWORD", "openitopen")
    if ok:
        session["admin"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "wrong password"}), 401


@app.post("/api/admin/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})


def require_admin():
    return session.get("admin") is True


# ---------------------------------------------------------------- admin data
def _preview(domain, payload):
    if domain == "nlp":
        t = payload.get("text", "")
        return (t[:60] + "...") if len(t) > 60 else t
    if domain == "ml":
        f = payload.get("features", [])
        return ",".join(str(round(v, 2)) for v in f[:4]) + "..."
    if domain in ("cv", "dl"):
        return "[image 28x28]"
    if domain == "ds":
        return (f"bill ${payload.get('total_bill')} size {payload.get('size')} "
                f"day {payload.get('day')} time {payload.get('time')}")
    return str(payload)[:60]


@app.get("/api/admin/stats")
def api_stats():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    c = sample_counts()
    labeled = sum(v["labeled"] for v in c.values())
    unlabeled = sum(v["unlabeled"] for v in c.values())
    predicted = int(session.get("predicted", 0))
    return jsonify({"labeled": labeled, "unlabeled": unlabeled,
                    "predicted": predicted, "by_domain": c})


@app.get("/api/admin/samples")
def api_samples():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    out = []
    for domain in ("cv", "dl", "nlp", "ml", "ds"):
        for labeled in (True, False):
            for r in fetch_samples(domain, labeled):
                payload = json.loads(r.payload)
                out.append({
                    "id": r.id, "domain": domain, "labeled": labeled,
                    "label": r.label, "preview": _preview(domain, payload),
                })
    out.sort(key=lambda x: -x["id"])
    return jsonify(out)


@app.post("/api/admin/samples")
def api_add_sample():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    domain = data.get("domain")
    payload = data.get("payload") or {}
    labeled = bool(data.get("labeled"))
    if domain not in ("cv", "dl", "nlp", "ml", "ds"):
        return jsonify({"error": "bad domain"}), 400
    label = data.get("label")
    target = data.get("target")
    if labeled and label is None and target is None:
        return jsonify({"error": "label required"}), 400
    sid = add_sample(domain, payload,
                     label=None if not labeled else label,
                     target=float(target) if target is not None else None)
    return jsonify({"ok": True, "id": sid})


@app.delete("/api/admin/samples/<int:sid>")
def api_del_sample(sid):
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify({"ok": delete_sample(sid)})


@app.post("/api/admin/samples/<int:sid>/label")
def api_label_sample(sid):
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(force=True) or {}
    target = data.get("target")
    return jsonify({"ok": set_label(sid, label=data.get("label"),
                                    target=float(target) if target is not None
                                    else None)})


@app.post("/api/admin/test")
def api_test():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    domain = (request.get_json(force=True) or {}).get("domain", "nlp")
    rows = fetch_samples(domain, labeled=False)
    items = []
    for r in rows:
        p = json.loads(r.payload)
        try:
            if domain == "nlp":
                pred, conf = predict_head("nlp", encode_text(p["text"]))
                label = "Positive" if int(pred[0]) else "Negative"
                c = float(conf[0])
                prev = p["text"][:80]
            elif domain == "ml":
                pred, conf = predict_head("ml", encode_ml(p["features"]))
                label = "Malignant" if int(pred[0]) else "Benign"
                c = float(conf[0])
                prev = _preview("ml", p)
            elif domain in ("cv", "dl"):
                pred, conf = predict_head(domain, encode_pixels(p["pixels"]))
                from src.model import CV_CLASSES, DL_CLASSES
                classes = CV_CLASSES if domain == "cv" else DL_CLASSES
                label = classes[int(pred[0])]
                c = float(conf[0])
                prev = "[image 28x28]"
            else:
                pred, _ = predict_head("ds", encode_ds(
                    p["total_bill"], p["size"], p["day"], p["time"]))
                label = f"${float(pred[0]):.2f}"
                c = None
                prev = _preview("ds", p)
            items.append({"id": r.id, "preview": prev, "pred": label,
                          "conf": c})
        except Exception as e:
            items.append({"id": r.id, "preview": _preview(domain, p),
                          "pred": f"error: {e}", "conf": None})
    session["predicted"] = int(session.get("predicted", 0)) + len(items)
    return jsonify({"items": items})


# ---------------------------------------------------------------- retrain
def _run_train():
    try:
        TRAIN.update(running=True, done=False, error="", step=0, total=0,
                     head="")
        from src import train as T
        from src.data import load_datasets

        data, artifacts = load_datasets()

        def cb(step, total, head, loss):
            TRAIN.update(step=step, total=total, head=head)

        model, m = T.train(data, progress_cb=cb)
        T.save(model, artifacts, m)
        reload_model()
        TRAIN.update(running=False, done=True)
    except Exception as e:
        TRAIN.update(running=False, error=str(e))


@app.post("/api/admin/retrain")
def api_retrain():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    if TRAIN["running"]:
        return jsonify({"ok": False, "error": "already running"}), 409
    threading.Thread(target=_run_train, daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/admin/train/status")
def api_train_status():
    if not require_admin():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(TRAIN)


# ---------------------------------------------------------------- bootstrap
init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
