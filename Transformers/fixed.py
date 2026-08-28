import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Model, ops
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

constituents_df = pd.read_csv("constituents.csv")
tags_df = pd.read_csv("tags.csv")

print("Constituents shape:", constituents_df.shape)
print("Tags shape:", tags_df.shape)

tags_df = (
    tags_df
    .sort_values(["event_no", "jet_no"])
    .groupby("event_no", sort=False)
    .head(2)
    .reset_index(drop=True)
)

merged_df = constituents_df.merge(tags_df, on=["event_no", "jet_no"], how="inner")

print("Tags after filtering:", tags_df.shape)
print("Merged shape:", merged_df.shape)

FEATURE_COLS = ["pt", "eta", "phi", "charge", "energy"]
TOP_N = 50

def build_padded_arrays(merged, label_col, top_n=TOP_N):
    grouped = merged.groupby(["event_no", "jet_no"], sort=True)

    features_list = []
    labels = []

    for _, group in grouped:
        # Sort by pt descending, keep top_n
        group_sorted = group.nlargest(top_n, "pt")
        feats = group_sorted[FEATURE_COLS].values.astype("float32")
        features_list.append(feats)
        labels.append(int(group[label_col].iloc[0]))

    # max_len is at most top_n (jets with fewer constituents will be shorter)
    max_len = max(len(f) for f in features_list)
    N = len(features_list)
    print(f"  max constituents after top-{top_n} cut: {max_len}")

    X = np.zeros((N, max_len, 5), dtype="float32")
    mask = np.zeros((N, max_len),    dtype="float32")

    for i, f in enumerate(features_list):
        X[i, :len(f)] = f
        mask[i, :len(f)] = 1.0

    y = np.array(labels, dtype="float32")
    return X, mask, y


class AttentionMaskLayer(layers.Layer):
    def call(self, mask):
        attn = ops.einsum("bi,bj->bij", mask, mask)
        return ops.cast(attn, "bool")

class MaskedMeanPool(layers.Layer):
    def call(self, x, mask):
        mask_exp = ops.expand_dims(mask, -1)
        summed = ops.sum(x * mask_exp, axis=1)
        counts = ops.sum(mask_exp, axis=1)
        return summed / ops.maximum(counts, 1.0)

def build_transformer(max_len, feature_dim=5, d_model=32, num_heads=4,
                      num_blocks=2, ff_dim=64, dropout=0.1):
    x_in = layers.Input(shape=(max_len, feature_dim), name="constituents")
    mask_in = layers.Input(shape=(max_len,), name="mask")

    x = layers.LayerNormalization()(x_in)
    x = layers.Dense(d_model)(x)

    attn_mask = AttentionMaskLayer()(mask_in)

    for i in range(num_blocks):
        attn_out = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name=f"mha_{i}",
        )(x, x, attention_mask=attn_mask)
        x = layers.LayerNormalization(name=f"norm1_{i}")(x + attn_out)

        ffn_out = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{i}")(x)
        ffn_out = layers.Dense(d_model, name=f"ffn2_{i}")(ffn_out)
        ffn_out = layers.Dropout(dropout, name=f"drop_{i}")(ffn_out)
        x = layers.LayerNormalization(name=f"norm2_{i}")(x + ffn_out)

    pooled = MaskedMeanPool()(x, mask_in)
    output = layers.Dense(1, activation="sigmoid")(pooled)

    model = Model(inputs=[x_in, mask_in], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["AUC", "accuracy"],
    )
    return model

def run_experiment(label_col, merged_df, epochs=10, batch_size=64):
    print(f"\n{'='*55}")
    print(f"  Training on: {label_col}  (top {TOP_N} constituents by pt)")
    print(f"{'='*55}")

    X, mask, y = build_padded_arrays(merged_df, label_col)
    print(f"  X shape   : {X.shape}")

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  Class 1 (quark): {n_pos}   Class 0 (gluon): {n_neg}")

    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, stratify=y.astype(int), random_state=42
    )

    X_train, mask_train, y_train = X[train_idx], mask[train_idx], y[train_idx]
    X_val, mask_val, y_val = X[val_idx], mask[val_idx], y[val_idx]

    n_pos_tr = int(y_train.sum())
    n_neg_tr = len(y_train) - n_pos_tr
    total_tr = len(y_train)
    w0 = total_tr / (2 * n_neg_tr)
    w1 = total_tr / (2 * n_pos_tr)
    class_weight = {0: w0, 1: w1}
    print(f"  Class weights:: 0 (gluon): {w0:.3f}, 1 (quark): {w1:.3f}")

    model = build_transformer(max_len=X.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_AUC", mode="max", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_AUC", mode="max", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        [X_train, mask_train], y_train,
        validation_data=([X_val, mask_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    y_prob  = model.predict([X_val, mask_val], batch_size=batch_size).ravel()
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"\n  Validation ROC AUC: {roc_auc:.4f}")

    y_pred = (y_prob >= 0.5).astype(int)
    print("  Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("  Classification Report:\n",
          classification_report(y_val, y_pred, digits=4))

    hist = history.history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training: {label_col} (top {TOP_N} by pt)")
    for ax, m, vm in zip(
        axes,
        ["loss", "AUC", "accuracy"],
        ["val_loss", "val_AUC", "val_accuracy"]
    ):
        ax.plot(hist[m],  label="train")
        ax.plot(hist[vm], label="val")
        ax.set_title(m); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"training_curves_{label_col}.png")

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC — {label_col} (top {TOP_N} by pt)")
    plt.legend(); plt.grid(alpha=0.3); plt.savefig(f"roc_{label_col}.png")

    return model, history, fpr, tpr, roc_auc

# Train on IFN tag
model_ifn, hist_ifn, fpr_ifn, tpr_ifn, auc_ifn = run_experiment("IFN tag", merged_df)

# Train on ATLAS tag
model_atlas, hist_atlas, fpr_atlas, tpr_atlas, auc_atlas = run_experiment("ATLAS tag", merged_df)

plt.figure(figsize=(6, 5))
plt.plot(fpr_ifn,   tpr_ifn,   label=f"IFN AUC = {auc_ifn:.4f}")
plt.plot(fpr_atlas, tpr_atlas, label=f"ATLAS AUC = {auc_atlas:.4f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title(f"ROC Comparison: IFN vs ATLAS ({TOP_N} highest pt constituents)")
plt.legend(); plt.grid(alpha=0.3); plt.savefig("roc_comparison_ifn_vs_atlas.png")

print(f"\nIFN  tag ROC AUC : {auc_ifn:.4f}")
print(f"ATLAS tag ROC AUC : {auc_atlas:.4f}")
