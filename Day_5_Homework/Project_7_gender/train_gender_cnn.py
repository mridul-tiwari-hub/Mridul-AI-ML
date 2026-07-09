import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Set paths
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "gender_dataset")
train_male_dir = os.path.join(dataset_dir, "train", "male")
train_female_dir = os.path.join(dataset_dir, "train", "female")

os.makedirs(train_male_dir, exist_ok=True)
os.makedirs(train_female_dir, exist_ok=True)

# Generate synthetic face-like images to have a real dataset to train on
# Male images: vertical lines or darker features
# Female images: circles or brighter features
IMG_SIZE = 150

print("Generating synthetic dataset...")
for i in range(30):
    # Male synthetic image
    male_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Draw vertical grid lines (resembles male pattern)
    cv2.gridInteraction = True
    for x in range(10, IMG_SIZE, 20):
        cv2.line(male_img, (x, 0), (x, IMG_SIZE), (100, 150, 255), 2)
    # Add noise
    male_img = cv2.blur(male_img + np.random.randint(0, 50, male_img.shape, dtype=np.uint8), (5,5))
    cv2.imwrite(os.path.join(train_male_dir, f"male_{i}.jpg"), male_img)
    
    # Female synthetic image
    female_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Draw concentric circles (resembles female pattern)
    for r in range(20, IMG_SIZE // 2, 20):
        cv2.circle(female_img, (IMG_SIZE // 2, IMG_SIZE // 2), r, (255, 150, 200), 2)
    # Add noise
    female_img = cv2.blur(female_img + np.random.randint(0, 50, female_img.shape, dtype=np.uint8), (5,5))
    cv2.imwrite(os.path.join(train_female_dir, f"female_{i}.jpg"), female_img)

print("Dataset generated successfully!")

# Define the CNN architecture as explained in the docx
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Setup generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=10,
    class_mode='binary'
)

# Train CNN model
model = create_cnn_model()
print("Training CNN model...")
model.fit(train_generator, epochs=5, steps_per_epoch=len(train_generator))

# Save model
model_path = os.path.join(base_dir, "gender_cnn_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}")
