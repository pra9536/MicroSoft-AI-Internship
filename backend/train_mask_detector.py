# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------- CONFIGURATION ----------------
INIT_LR = 1e-4        # Initial learning rate
EPOCHS = 20           # Total training epochs
BS = 32               # Batch size

DIRECTORY = r"F:\Prateek Yadav Internship\AICTE\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# ---------------- DATA LOADING ----------------
print("[INFO] Loading images...")
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Encode labels (one-hot)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split train/test data
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state=42)

# ---------------- DATA AUGMENTATION ----------------
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ---------------- MODEL BUILDING ----------------
print("[INFO] Building model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Show model summary
model.summary()

# ---------------- MODEL COMPILATION ----------------
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# ---------------- TRAINING ----------------
print("[INFO] Training head...")

# Callbacks for better training
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

# ✅ Fixed steps_per_epoch & validation_steps to avoid data shortage warning
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS, shuffle=True),
    # steps_per_epoch=max(1, len(trainX) // BS),
   # validation_data=(testX, testY),
    validation_data=(testX, testY),
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# ---------------- EVALUATION ----------------
print("[INFO] Evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# ---------------- SAVE MODEL ----------------
print("[INFO] Saving mask detector model...")
# After training
model.save("mask_detector.h5")



# ---------------- PLOT RESULTS ----------------
N = len(H.history["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

print("[INFO] Training complete ✅")
