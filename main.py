import random
import os
from google.colab import drive 

drive.mount('/content/drive')

train_zip_path = "/content/drive/MyDrive/ML_HW1_Dataset/train.zip"
test_zip_path = "/content/drive/MyDrive/ML_HW1_Dataset/test.zip"

if os.path.isfile(train_zip_path) and os.path.isfile(test_zip_path):
    os.system(f"unzip -q {train_zip_path} -d /content/")
    os.system(f"unzip -q {test_zip_path} -d /content/")
    print("Files successfully unzipped")
else:
    if not os.path.isfile(train_zip_path):
        print("Cannot find train.zip")
    if not os.path.isfile(test_zip_path):
        print("Cannot find test.zip")

"""## 1. Load Packages"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import random
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping

"""## 2. Load the Dataset

- Let's take a look at artist.csv, which contains important information including:
  - Artist name (name)
  - Style or genre (genre)
  - Number of paintings in the dataset (paintings)

There are a total of 50 artists, meaning there are 50 classes to classify.
"""

train_dir = "./train_resized/"
test_dir = "./test_resized/"
artists = pd.read_csv("./artists.csv")
num_classes = artists.shape[0]
print("Number of artists : ", num_classes)
artists.head()

"""Extract only the artist names and the number of paintings, and join the names using underscores."""

artists = artists.loc[:, ["name", "paintings"]]
artists["name"] = artists["name"].str.split(" ").apply(lambda parts: "_".join(parts))
artists.head()

"""
**Count the number of paintings per artist.**
* The data imbalance across classes can negatively impact model training. 
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.figure(figsize=(10, 6))

barplot = sns.barplot(x=artists.name,  y=artists.paintings)
for item in barplot.get_xticklabels():
    item.set_rotation(90)

print("It is evident that the number of paintings per artist is highly imbalanced, which can affect model training.")
print("Maximum number of paintings:", artists.paintings.max(), " Minimum number of paintings:", artists.paintings.min())

"""Randomly read and display some paintings"""

img_list = os.listdir(train_dir)
total_len = len(img_list)
random_list = random.sample(range(0, total_len), 20)
print("Total number of training paintings:", total_len)

show_imgs = [img_list[rand] for rand in random_list]

plt.figure(figsize=(16, 16))
for index, imgName in enumerate(show_imgs):
    img_path = train_dir + imgName
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.subplot(4, 5, index + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("_".join(imgName.split("_")[:-1]))

"""## Data Preprocessing

**Several important steps need to be handled:**
1. Currently, we only have the file paths of the paintings. For example: /content/train_resized/Claude_Monet_22.jpg. We need to extract the label: Claude_Monet.
2. The labels (artist names) are in English. We need to map them to numeric labels. For example: Van_Gogh → 0; Edvard_Munch → 1.
3. The numeric labels need to be further converted into one-hot vectors of depth 50.
4. We need to read the images from the file paths.
5. The images have different sizes; we must resize them to a consistent size to feed into the model.
6. Normalize the pixel values of each image to the range [0, 1].
7. The file paths are ordered by artist name — too organized! We need to shuffle the dataset.
8. If needed, split the dataset into training data and validation data.
9. Important! Make sure you know your input and output shapes before building the model. Example: input: (256, 256, 3); output: (50,).

"""

# Later during training, we will use class_name[artist_name] to get the label,
# and after prediction, we will use rev_class_name[predicted_index] to map back to the artist name.

def make_author_dict():
    # Get all unique artist names
    artists_names = artists["name"].unique()

    # Create mapping -> {artist_name: index}
    label_map = {name: index for index, name in enumerate(artists_names)}

    # Create reverse mapping -> {index: artist_name}
    rev_label_map = {index: name for name, index in label_map.items()}

    return label_map, rev_label_map

# Create the dictionary that maps artist names to numeric labels (class_name)
# and the dictionary that maps numeric labels back to artist names (rev_class_name)
class_name, rev_class_name = make_author_dict()

print("Mapping from English names to numeric labels:", class_name)
print("Mapping from numeric labels back to English names:", rev_class_name)


def get_label(pic_name, label_map):
    # Extract the label and convert it into a numeric value
    # Example: Claude_Monet_1.jpg -> Claude_Monet -> 1

    pic_name = pic_name.split('.')[0]
    name_parts = pic_name.split('_')

    # author_name = combine first and last names, excluding the index number
    author_name = '_'.join(name_parts[:-1])

    if author_name not in label_map:
        raise ValueError(f"Unknown artist: {author_name}")

    return label_map[author_name]


def get_path(dir, pic_name):
    # Merge the directory path with the picture name
    # Example: ./train_resized/ + Claude_Monet_1.jpg => ./train_resized/Claude_Monet_1.jpg

    path = os.path.join(dir, pic_name)

    return path


def make_paths_label(dir):
    img_list = os.listdir(dir)
    paths = []
    labels = []

    # Preprocess and store the paths and labels using a for loop
    for img_name in img_list:
        path = get_path(dir, img_name)
        label = get_label(img_name, class_name)
        paths.append(path)
        labels.append(label)

    # Determine the number of classes for one-hot encoding
    num_classes = len(class_name)

    # Convert labels to one-hot encoding
    onehot_labels = keras.utils.to_categorical(labels, num_classes=num_classes)

    return paths, onehot_labels

# Let's check it
paths, onehot_labels = make_paths_label(train_dir)
ld = 2
print("paths : ")
for p in paths[:5]:
    print(p)
print("-" * 20)
print("labels : ")
for label in onehot_labels[:5]:
    print(label)


# Define the image dimensions for model input
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

# Shuffle buffer size
SHUFFLE_BUFFER = 1000

def get_image(path):
    # Read image from the given path
    file = tf.io.read_file(path)
    img = tf.io.decode_jpeg(file, channels=3)
    img = tf.cast(img, tf.float32)

    # Resize each image to IMG_HEIGHT x IMG_WIDTH
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])

    # Normalize pixel values to the [0, 1] range
    img = img / 255.0

    return img


# Data Augmentation -> Reduce overfitting
def load_image_with_augmentation(path, label):
    img = get_image(path)

    img = tf.image.random_flip_left_right(img)    # Random horizontal flip
    img = tf.image.random_brightness(img, max_delta=0.1)    # Random brightness adjustment
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)    # Random contrast adjustment
    img = tf.image.random_saturation(img, lower=0.9, upper=1.1)  # Random saturation adjustment
    img = tf.image.random_hue(img, max_delta=0.01)    # Random hue adjustment

    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, label


# Oversampling
def make_balanced_dataset(paths, labels):

    label_counts = Counter(labels)
    threshold = 250

    aug_paths, aug_labels = [], []

    for path, label in zip(paths, labels):

        # Keep the original image
        aug_paths.append(path)
        aug_labels.append(label)

        # If the number of images for a label is less than the threshold, apply oversampling
        if label_counts[label] < threshold:
            repeat_times = (threshold - label_counts[label]) // label_counts[label]
            for _ in range(repeat_times):
                # Repeatedly add the original image path and label
                aug_paths.append(path)
                aug_labels.append(label)

    # Convert labels to one-hot encoding
    num_classes = len(set(aug_labels))
    onehot_labels = keras.utils.to_categorical(aug_labels, num_classes=num_classes)

    path_ds = tf.data.Dataset.from_tensor_slices(aug_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(onehot_labels)

    full_ds = tf.data.Dataset.zip((path_ds, label_ds))

    # Data augmentation
    full_ds = full_ds.map(load_image_with_augmentation)
    
    # Batching and prefetching
    full_ds = full_ds.shuffle(SHUFFLE_BUFFER).batch(32).prefetch(tf.data.AUTOTUNE)

    # Visualize the distribution after oversampling
    balanced_counts = Counter(aug_labels)

    df = pd.DataFrame.from_dict(balanced_counts, orient='index', columns=['paintings'])
    df = df.reset_index().rename(columns={"index": "label"})

    if 'rev_class_name' in globals():
        df["name"] = df["label"].apply(lambda idx: rev_class_name[idx])
    else:
        df["name"] = df["label"]

    plt.figure(figsize=(10, 6))
    sns.barplot(x="name", y="paintings", data=df)
    plt.xticks(rotation=90)
    plt.show()

    # Return the balanced dataset
    return full_ds



# First split into train and validation sets, then apply oversampling and augmentation
# Only the training dataset (train_ds) will be processed; validation dataset (val_ds) remains untouched
def make_dataset(dir):

    paths, onehot_labels = make_paths_label(dir)
    total_len = len(paths)

    # Split into training data (80%) and validation data (20%)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    train_paths = paths[:train_len]
    train_labels = [np.argmax(l) for l in onehot_labels[:train_len]]

    val_paths = paths[train_len:]
    val_labels = onehot_labels[train_len:]

    # Process training data -> oversampling + augmentation
    train_ds = make_balanced_dataset(train_paths, train_labels)

    # Estimate the total size of the training data after oversampling
    label_counts = Counter(train_labels)
    threshold = 250
    est_total = 0
    for count in label_counts.values():
        est_total += count
        if count < threshold:
            # A rough estimate of how many times each image will be duplicated
            repeat_times = (threshold - count) // count
            est_total += count * repeat_times

    # Validation data (no oversampling or augmentation)
    val_path_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    val_label_ds = tf.data.Dataset.from_tensor_slices(val_labels)
    val_img_ds = val_path_ds.map(get_image)
    val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds)).batch(32).prefetch(tf.data.AUTOTUNE)

    print("Train size (before oversampling):", len(train_paths), f" ⮕ after (estimated): {est_total}")
    print("Validation size:", len(val_paths))

    return train_ds, val_ds

# Build the training and validation datasets
train_ds, val_ds = make_dataset(train_dir)


# Build the test dataset
def make_test_dataset(test_dir):

    paths, onehot_labels = make_paths_label(test_dir)

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(onehot_labels)

    # Resize + normalize the images
    img_ds = path_ds.map(lambda p: get_image(p))

    test_ds = tf.data.Dataset.zip((img_ds, label_ds)).batch(50)
    return test_ds

# Visualize a few images from the training set
plt.figure(figsize=(12, 8))
for image_batch, label_batch in train_ds.take(1):
    for index in range(6):
        img = image_batch[index].numpy()
        label = label_batch[index].numpy()
        l = np.argmax(label)

        plt.subplot(2, 3, index + 1)
        plt.imshow(img)
        plt.title("Label number: {} \n Author Name: {}".format(l, rev_class_name[l]))
        plt.axis("off")


# Check the dimensions after adding batch
trainiter = iter(train_ds)
x, y = next(trainiter)
print("Training image batch shape:", x.shape)
print("Training label batch shape:", y.shape)

"""## 4. Build the Model
"""

input_shape = (128, 128, 3)
num_classes = 50

from tensorflow.keras import regularizers

# Define custom model
model = keras.Sequential([
    keras.Input(shape=input_shape),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(1024, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-5)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-5)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation="softmax"),
])

# Show the model architecture
model.summary()

"""## 5. Define the Training Plan

Feed the preprocessed data into the model.
"""

EPOCHS = 50

optimizer = keras.optimizers.Adam(learning_rate=5e-5)

# Compile the model: define learning strategy and loss function
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Stop training early if validation loss does not improve
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Start training
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stop])


"""## 6. Evaluate the Model"""

# Check what metrics are available in history
print(history.history.keys())

# Plot training and validation accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="lower right")
plt.show()

# Plot training and validation loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper right")
plt.show()

# Load the test dataset
test_ds = make_test_dataset(test_dir)

# Evaluate the model
score = model.evaluate(test_ds)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""## 7. Make Predictions
* Use the model to predict authors for previously loaded images.
* Try feeding a real-world painting image into the model and predict its author.
"""

def predict_author(img):
    # A function to predict the author from a single image
    # Input : OpenCV image (height, width, 3)
    # Output : Predicted artist name, e.g., Claude_Monet

    # 1. Expand the image dimensions: (height, width, 3) -> (1, height, width, 3)
    img = np.expand_dims(img, axis=0)

    # 2. Pass through the model using model.predict
    # 3. Extract the softmax output (shape: (50,)) and get the index with the highest probability
    pred_idx = np.argmax(model.predict(img, verbose=0)[0])

    # 4. Map the prediction index back to the artist name
    return rev_class_name.get(pred_idx, "Unknown")

# Visualize predictions
plt.figure(figsize=(16, 16))
for index, imgName in enumerate(show_imgs):
    img_path = train_dir + imgName
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.subplot(4, 5, index + 1)
    plt.axis("off")
    plt.imshow(img)

    # Resize and normalize the image before prediction
    img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0

    true_author = "_".join(imgName.split("_")[:-1])
    pred_author = predict_author(img).replace(" ", "_")

    plt.title(
        "True Author: {}\nPredicted Author: {}".format(
            true_author, pred_author
        ),
        size=11,
    )

"""
* Upload Your Own Image and Test the Model
"""

from google.colab import files

def upload_img():
    uploaded = files.upload()
    img_name = list(uploaded.keys())[0]
    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img

def eval():
    img = upload_img()
    plt.title("Predicted Author: {}".format(predict_author(img).replace(" ", "_")))
    plt.axis("off")
    plt.show()

# Upload your own image to test the model
eval()
