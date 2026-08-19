import os

!pip install -q kaggle

from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

if not os.path.exists("data/shipsnet.json"):
    !kaggle datasets download -d hariharanalm/dolphin-vs-whale -p data --unzip
    !kaggle datasets download -d rhammell/ships-in-satellite-imagery -p data --unzip
else:
    print("Data already downloaded, skipping.")

import json, random
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 128

# ---------------------------------------------------------------------------
# Load ALL whale images (no sampling cap)
# ---------------------------------------------------------------------------
all_files = []
for root, dirs, fs in os.walk("data"):
    for f in fs:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_files.append(os.path.join(root, f))

whale_paths = [p for p in all_files if "whale" in p.lower()]
print("Whale images found:", len(whale_paths))
assert len(whale_paths) > 0, "No whale images found — check the 'data' folder structure with os.walk('data')"

whale_images = [
    np.array(Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))
    for p in whale_paths
]

# ---------------------------------------------------------------------------
# Load ALL ship images (no sampling cap)
# ---------------------------------------------------------------------------
with open("data/shipsnet.json") as f:
    ship_json = json.load(f)

ship_images = []
for i in range(len(ship_json["data"])):
    px = np.array(ship_json["data"][i], dtype=np.uint8).reshape(3, 80, 80).transpose(1, 2, 0)
    ship_images.append(np.array(Image.fromarray(px).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)))

print("Ship images found:", len(ship_images))

# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
X = np.array(whale_images + ship_images, dtype="float32") / 255.0
y = np.array([1] * len(whale_images) + [0] * len(ship_images))

idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# ---------------------------------------------------------------------------
# Class weights to handle imbalance (whale vs ship counts likely differ a lot)
# ---------------------------------------------------------------------------
n_whale = int((y_train == 1).sum())
n_ship = int((y_train == 0).sum())
total = n_whale + n_ship
class_weight = {
    0: total / (2 * n_ship) if n_ship > 0 else 1.0,
    1: total / (2 * n_whale) if n_whale > 0 else 1.0,
}
print("Class counts -> whale:", n_whale, " ship:", n_ship)
print("Class weights:", class_weight)

# ---------------------------------------------------------------------------
# Model with light data augmentation baked in
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    X_train, y_train,
    epochs=15,
    validation_split=0.15,
    batch_size=64,
    class_weight=class_weight,
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2%}")

for i in range(4):
    pred = "WHALE ALERT" if model.predict(X_test[i:i+1], verbose=0)[0][0] > 0.5 else "no whale"
    actual = "whale" if y_test[i] == 1 else "no_whale"
    plt.imshow(X_test[i], interpolation="nearest")
    plt.title(f"Predicted: {pred} | Actual: {actual}")
    plt.axis("off")
    plt.show()

model.save("whale_detector.keras")
print("Saved: whale_detector.keras")
