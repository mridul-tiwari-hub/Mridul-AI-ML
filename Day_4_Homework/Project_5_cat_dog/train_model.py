import os
import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Dataset path
dataset_path = os.path.dirname(os.path.abspath(__file__))

images = []
labels = []
classes = ['Cat', 'Dog']
IMG_SIZE = 64

print("Loading dataset images...")
for label, folder in enumerate(classes):
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found!")
        continue
    
    count = 0
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.flatten())
            labels.append(label)
            count += 1
        except Exception as e:
            print(f"Error reading image {file}: {e}")
    print(f"Loaded {count} images from {folder}")

X = np.array(images) / 255.0
y = np.array(labels)

print(f"Training data dimensions: X = {X.shape}, y = {y.shape}")

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

model_path = os.path.join(dataset_path, "cat_dog_model.pkl")
joblib.dump(model, model_path)
print(f"Success! Model trained and saved to {model_path}")