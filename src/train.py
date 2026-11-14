import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_datasets, save_artifacts, MODELS_DIR
from src.model import build_model, HEADS, MODALITY, IMG_SHAPE, TXT_LEN, TAB_DIM, DS_DIM

MODEL_PATH = MODELS_DIR / "school_ai.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"

EPOCHS = {
    "cv": 4,
    "dl": 3,
    "nlp": 8,
    "ml": 15,
    "ds": 30,
}
BATCH = 128


def _loss_fn(head):
    if head == "ds":
        return keras.losses.MeanSquaredError()
    return keras.losses.SparseCategoricalCrossentropy()


def _zeros_y(head, n):
    if head == "ds":
        return np.zeros((n, 1), np.float32)
    units = 10 if head in ("cv", "dl") else 2
    return np.zeros((n, units), np.float32)


def train(data, epochs_map=None, progress_cb=None):
    """Joint multi-task training: cycles batches through each head while the
    other heads' losses are masked out. Returns (model, metrics)."""
    epochs_map = epochs_map or EPOCHS
    model = build_model()
    losses = {h: _loss_fn(h) for h in HEADS}
    opt = keras.optimizers.Adam(1e-3)
    opt.build(model.trainable_variables)
    rng = np.random.RandomState(42)

    total_steps = sum(
        int(np.ceil(len(data[h]["train"][0]) / BATCH)) * epochs_map[h]
        for h in HEADS
    )
    step = 0
    t0 = time.time()

    for epoch in range(max(epochs_map.values())):
        for head in HEADS:
            if epoch >= epochs_map[head]:
                continue
            xs, ys = data[head]["train"]
            order = rng.permutation(len(xs))
            modality = MODALITY[head]
            for start in range(0, len(xs), BATCH):
                idx = order[start:start + BATCH]
                bx, by = xs[idx], ys[idx]
                inputs = {
                    "image": np.zeros((len(bx),) + IMG_SHAPE, np.float32),
                    "text": np.zeros((len(bx), TXT_LEN), np.int32),
                    "tabular": np.zeros((len(bx), TAB_DIM), np.float32),
                    "dsx": np.zeros((len(bx), DS_DIM), np.float32),
                }
                inputs[modality] = bx
                y = by.reshape(-1, 1).astype(np.float32) if head == "ds" else by

                with tf.GradientTape() as tape:
                    preds = model(inputs, training=True)
                    pred = preds[HEADS.index(head)]
                    loss = losses[head](y, pred)
                grads = tape.gradient(loss, model.trainable_variables)
                opt.apply_gradients(
                    (tf.zeros_like(v) if g is None else g, v)
                    for g, v in zip(grads, model.trainable_variables)
                )
                step += 1
                if progress_cb:
                    progress_cb(step, total_steps, head, float(loss))
                if step % 50 == 0:
                    print(f"[{step}/{total_steps}] {head} loss={loss:.4f}")

    metrics = evaluate(model, data)
    train_time = time.time() - t0
    metrics["train_time_sec"] = round(train_time, 1)
    return model, metrics


def evaluate(model, data):
    metrics = {}
    for head in HEADS:
        xs, ys = data[head]["test"]
        modality = MODALITY[head]
        preds_all = []
        for start in range(0, len(xs), 256):
            bx = xs[start:start + 256]
            inputs = {
                "image": np.zeros((len(bx),) + IMG_SHAPE, np.float32),
                "text": np.zeros((len(bx), TXT_LEN), np.int32),
                "tabular": np.zeros((len(bx), TAB_DIM), np.float32),
                "dsx": np.zeros((len(bx), DS_DIM), np.float32),
            }
            inputs[modality] = bx
            out = model(inputs, training=False)
            preds_all.append(out[HEADS.index(head)].numpy())
        pred = np.concatenate(preds_all)
        if head == "ds":
            mae = float(np.mean(np.abs(pred.reshape(-1) - ys)))
            metrics[head] = {"mae": round(mae, 4), "n": int(len(ys))}
        else:
            acc = float((pred.argmax(axis=1) == ys).mean())
            metrics[head] = {"accuracy": round(acc, 4), "n": int(len(ys))}
    return metrics


def save(model, artifacts, metrics):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_artifacts(artifacts)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))


def main():
    print("loading datasets...")
    data, artifacts = load_datasets()
    for h in HEADS:
        print(f"  {h}: train={len(data[h]['train'][0])} test={len(data[h]['test'][0])}")
    print("training...")
    model, metrics = train(data)
    save(model, artifacts, metrics)


if __name__ == "__main__":
    main()
