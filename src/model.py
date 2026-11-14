import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

HEADS = ["cv", "dl", "nlp", "ml", "ds"]
MODALITY = {"cv": "image", "dl": "image", "nlp": "text", "ml": "tabular", "ds": "dsx"}

IMG_SHAPE = (28, 28, 1)
TXT_LEN = 200
VOCAB_SIZE = 5000
TAB_DIM = 30
DS_DIM = 4

CV_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
DL_CLASSES = [str(i) for i in range(10)]
NLP_CLASSES = ["Negative", "Positive"]
ML_CLASSES = ["Benign", "Malignant"]
HEAD_CLASSES = {"cv": CV_CLASSES, "dl": DL_CLASSES, "nlp": NLP_CLASSES, "ml": ML_CLASSES}
HEAD_LABELS = {
    "cv": "Computer Vision (Fashion-MNIST)",
    "dl": "Deep Learning (MNIST digits)",
    "nlp": "NLP (IMDB sentiment)",
    "ml": "Machine Learning (breast cancer)",
    "ds": "Data Science (tips regression)",
}


def build_model():
    img_in = keras.Input(shape=IMG_SHAPE, name="image")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(img_in)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    img_feat = layers.Dense(128, activation="relu", name="img_feat")(x)

    txt_in = keras.Input(shape=(TXT_LEN,), dtype="int32", name="text")
    t = layers.Embedding(VOCAB_SIZE, 32)(txt_in)
    t = layers.GlobalAveragePooling1D()(t)
    txt_feat = layers.Dense(64, activation="relu", name="txt_feat")(t)

    tab_in = keras.Input(shape=(TAB_DIM,), name="tabular")
    b = layers.Dense(64, activation="relu")(tab_in)
    tab_feat = layers.Dense(32, activation="relu", name="tab_feat")(b)

    ds_in = keras.Input(shape=(DS_DIM,), name="dsx")
    s = layers.Dense(32, activation="relu")(ds_in)
    ds_feat = layers.Dense(16, activation="relu", name="ds_feat")(s)

    shared = layers.Concatenate()([img_feat, txt_feat, tab_feat, ds_feat])
    shared = layers.Dense(128, activation="relu", name="shared")(shared)
    shared = layers.Dropout(0.2)(shared)

    def head(name, feat, units):
        h = layers.Concatenate()([shared, feat])
        h = layers.Dense(64, activation="relu")(h)
        if units > 1:
            return layers.Dense(units, activation="softmax", name=name)(h)
        return layers.Dense(1, activation="linear", name=name)(h)

    outputs = [
        head("cv", img_feat, 10),
        head("dl", img_feat, 10),
        head("nlp", txt_feat, 2),
        head("ml", tab_feat, 2),
        head("ds", ds_feat, 1),
    ]

    model = keras.Model(
        inputs=[img_in, txt_in, tab_in, ds_in],
        outputs=outputs,
        name="school_ai_multitask",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss={
            "cv": "sparse_categorical_crossentropy",
            "dl": "sparse_categorical_crossentropy",
            "nlp": "sparse_categorical_crossentropy",
            "ml": "sparse_categorical_crossentropy",
            "ds": "mse",
        },
        metrics={"cv": "accuracy", "dl": "accuracy", "nlp": "accuracy",
                 "ml": "accuracy", "ds": "mae"},
    )
    return model


def empty_inputs(batch=1):
    return {
        "image": tf.zeros((batch,) + IMG_SHAPE, dtype=tf.float32),
        "text": tf.zeros((batch, TXT_LEN), dtype=tf.int32),
        "tabular": tf.zeros((batch, TAB_DIM), dtype=tf.float32),
        "dsx": tf.zeros((batch, DS_DIM), dtype=tf.float32),
    }


def predict_head(model, head, x):
    """Run one input through the full model, return (probs/val, confidence)."""
    import numpy as np
    inputs = empty_inputs(batch=len(x) if hasattr(x, "__len__") else 1)
    inputs[MODALITY[head]] = x
    preds = model.predict(inputs, verbose=0)
    out = preds[HEADS.index(head)]
    if head == "ds":
        return out.astype(float).reshape(-1), None
    idx = out.argmax(axis=1)
    conf = out.max(axis=1)
    return idx, conf
