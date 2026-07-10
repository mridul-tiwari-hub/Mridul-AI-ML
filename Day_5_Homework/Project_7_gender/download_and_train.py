import os
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "gender_dataset")
train_male_dir = os.path.join(dataset_dir, "train", "male")
train_female_dir = os.path.join(dataset_dir, "train", "female")

# Recreate folders to clear old images
if os.path.exists(train_male_dir):
    shutil.rmtree(train_male_dir)
if os.path.exists(train_female_dir):
    shutil.rmtree(train_female_dir)

os.makedirs(train_male_dir, exist_ok=True)
os.makedirs(train_female_dir, exist_ok=True)

# 20 Unique Male Face IDs from Unsplash
male_ids = [
    "photo-1507003211169-0a1dd7228f2d",
    "photo-1500648767791-00dcc994a43e",
    "photo-1539571696357-5a69c17a67c6",
    "photo-1506794778202-cad84cf45f1d",
    "photo-1522075469751-3a6694fb2f61",
    "photo-1501196354995-cbb51c65aaea",
    "photo-1492562080023-ab3db95bfbce",
    "photo-1504257400765-1d019b88c658",
    "photo-1519085360753-af0119f7cbe7",
    "photo-1489980508314-941910ded1f4",
    "photo-1506863530036-1efeddceb993",
    "photo-1531427186611-ecfd6d936c79",
    "photo-1503443207922-dff7d543ee0e",
    "photo-1534308983496-4fabb1a015ee",
    "photo-1568602471122-7832951cc4c5",
    "photo-1500048993953-d23a436266cf",
    "photo-1513956589380-bad6acb9b9d4",
    "photo-1507153079406-79046180b3fa",
    "photo-1519345182560-3f2917c472ef",
    "photo-1542156822-6924d1a71aba"
]

# 20 Unique Female Face IDs from Unsplash
female_ids = [
    "photo-1494790108377-be9c29b29330",
    "photo-1438761681033-6461ffad8d80",
    "photo-1534528741775-53994a69daeb",
    "photo-1544005313-94ddf0286df2",
    "photo-1517841905240-472988babdf9",
    "photo-1508214751196-bcfd4ca60f91",
    "photo-1542206395-9feb3edaa68d",
    "photo-1531746020798-e6953c6e8e04",
    "photo-1524504388940-b1c1722653e1",
    "photo-1488426862026-3ee34a7d66df",
    "photo-1521119989659-a83eee488004",
    "photo-1531123897727-8f129e1688ce",
    "photo-1529626455594-4ff0802cfb7e",
    "photo-1506919258185-6078bba55d2a",
    "photo-1530577197743-7adf1009518b",
    "photo-1541647376583-d1e257237598",
    "photo-1509305717901-817478657471",
    "photo-1554151228-14d9def656e4",
    "photo-1484186139897-d5fc6b908812",
    "photo-1485178575877-1a13bf489fea"
]

IMG_SIZE = 150

def download_image(url, filepath):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Downloaded: {filepath}")
    except Exception as e:
        print(f"Failed downloading {url}: {e}")

print("Downloading real human face dataset...")
for i, photo_id in enumerate(male_ids):
    url = f"https://images.unsplash.com/{photo_id}?w={IMG_SIZE}&h={IMG_SIZE}&fit=crop"
    download_image(url, os.path.join(train_male_dir, f"male_{i}.jpg"))

for i, photo_id in enumerate(female_ids):
    url = f"https://images.unsplash.com/{photo_id}?w={IMG_SIZE}&h={IMG_SIZE}&fit=crop"
    download_image(url, os.path.join(train_female_dir, f"female_{i}.jpg"))

print("Dataset generated successfully!")

# Define Transfer Learning CNN Model
def create_transfer_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Set up generators with proper MobileNetV2 preprocessing [-1, 1]
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=4,
    class_mode='binary'
)

# Train the model
model = create_transfer_model()
print("Training transfer learning model with proper preprocessing...")
model.fit(train_generator, epochs=20, steps_per_epoch=len(train_generator))

# Save model locally
local_model_path = os.path.join(base_dir, "gender_cnn_model.h5")
model.save(local_model_path)
print(f"Model saved to {local_model_path}")

# Copy model to other required folders in the workspace
workspace_root = os.path.dirname(os.path.dirname(base_dir))
dest_paths = [
    os.path.join(workspace_root, "models", "gender_cnn_model.h5"),
    os.path.join(workspace_root, "Day_5_Project_7_Gender_Classification", "models", "gender_cnn_model.h5")
]

for dest in dest_paths:
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(local_model_path, dest)
    print(f"Copied model to: {dest}")

print("All tasks completed successfully!")
