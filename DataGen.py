import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(dataset_dir, img_size=(32, 32)):
    X = []
    y = []

    for label in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Hatalı dosya: {file_path}")
                continue

            resized = cv2.resize(img, img_size)
            flattened = resized.flatten()
            X.append(flattened)
            y.append(label)

    return np.array(X), np.array(y)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("acc:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model {path} dosyasına kaydedildi.")

def main():
    dataset_dir = "dataset"  # bu klasörü sen oluşturmalısın
    X, y = load_dataset(dataset_dir)
    model = train_model(X, y)
    save_model(model)

if __name__ == "__main__":
    main()
