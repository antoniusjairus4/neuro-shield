import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────
AD_DIR    = 'datasets/converted_nii'
CN_DIR    = 'datasets/converted_nii_cn'
MODEL_DIR = 'models'
IMG_SIZE  = 128
BATCH     = 8
EPOCHS    = 50
os.makedirs(MODEL_DIR, exist_ok=True)

# ── LOAD & PREPROCESS ONE NII FILE ──────────────────────
def load_nii_slice(path, size=IMG_SIZE):
    try:
        img = nib.load(path).get_fdata()
        if img.ndim < 3:
            return None
        mid = img.shape[2] // 2
        slc = img[:, :, mid]
        slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
        slc = tf.image.resize(slc[..., np.newaxis], [size, size]).numpy()
        slc = np.concatenate([slc, slc, slc], axis=-1)
        return slc.astype(np.float32)
    except:
        return None

# ── EXTRACT PATIENT ID FROM FILENAME ────────────────────
def get_patient_id(filename):
    # Format: 003_S_1059_xxx.nii → patient = 003_S_1059
    parts = filename.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return filename

# ── LOAD DATASET ────────────────────────────────────────
print("Loading AD scans...")
X, y, groups = [], [], []

for f in sorted(os.listdir(AD_DIR)):
    if f.endswith('.nii'):
        data = load_nii_slice(os.path.join(AD_DIR, f))
        if data is not None:
            X.append(data)
            y.append(1)
            groups.append(get_patient_id(f))

print(f"  Loaded {sum(1 for label in y if label == 1)} AD scans")

print("Loading CN scans...")
for f in sorted(os.listdir(CN_DIR)):
    if f.endswith('.nii'):
        data = load_nii_slice(os.path.join(CN_DIR, f))
        if data is not None:
            X.append(data)
            y.append(0)
            groups.append(get_patient_id(f))

print(f"  Loaded {sum(1 for label in y if label == 0)} CN scans")

X = np.array(X)
y = np.array(y)
groups = np.array(groups)
X, y, groups = shuffle(X, y, groups, random_state=42)
print(f"\nTotal dataset: {len(X)} scans | Shape: {X.shape}")
print(f"Unique patients: {len(set(groups))}")

# ── PATIENT-LEVEL TRAIN/TEST SPLIT ──────────────────────
# This prevents data leakage — same patient never in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
groups_train = groups[train_idx]

# Validation split from training set
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx2, val_idx = next(gss_val.split(X_train, y_train, groups_train))

X_val = X_train[val_idx]
y_val = y_train[val_idx]
X_train = X_train[train_idx2]
y_train = y_train[train_idx2]

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Train AD: {sum(y_train)} | Train CN: {sum(y_train==0)}")

# ── BUILD MODEL ─────────────────────────────────────────
print("\nBuilding EfficientNetB4 model...")
base = EfficientNetB4(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ── CALLBACKS ───────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        f'{MODEL_DIR}/mri_model_best.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    )
]

# ── TRAIN ───────────────────────────────────────────────
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# ── EVALUATE ────────────────────────────────────────────
print("\nEvaluating on test set...")
loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Test AUC:      {auc:.4f}")

# ── SAVE ────────────────────────────────────────────────
model.save(f'{MODEL_DIR}/mri_model_final.h5')
print(f"\nModel saved to {MODEL_DIR}/mri_model_final.h5")

# ── PLOT ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train Acc')
ax1.plot(history.history['val_accuracy'], label='Val Acc')
ax1.set_title('Accuracy'); ax1.legend()
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss'); ax2.legend()
plt.savefig(f'{MODEL_DIR}/training_curves.png')
print("Training curves saved!")
