import cv2
import numpy as np
import pickle
from collections import deque

def load_model(path="model.pkl"): #modeli çekmek için
    with open(path, "rb") as f:
        return pickle.load(f)

def preprocess(path, threshold=120): #siyah-beyaz halde okuması için
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = np.where(gray > threshold, 0, 1).astype(np.uint8)
    return image, binary

def bfs_label(binary): #label işlemi için matris oluşturuyor
    h, w = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    label_id = 1

    for y in range(h): #labeling olmadıysa bfs başlatılıyor
        for x in range(w):
            if binary[y, x] == 1 and labels[y, x] == 0:
                q = deque()
                q.append((y, x))
                labels[y, x] = label_id
                while q: #BFS ile label yayılımı
                    cy, cx = q.popleft()
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                                    labels[ny, nx] = label_id
                                    q.append((ny, nx))
                label_id += 1
    return labels, label_id

def extract_and_predict(image, binary, labels, label_id, model): #roi çıkarımı ve OCR
    h, w = binary.shape
    results = []

    for label in range(1, label_id): #labeling yapılan pixellerin koordinatları okunması için
        ys, xs = np.where(labels == label)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        width, height = x_max - x_min, y_max - y_min
        if 15 < w < 80 and 20 < h < 100: #gerçek dünyaya uyum için gürültü
            continue

        #32x32 boyutunda sınıflandırma datasetten tahmin etmesi için
        roi = binary[y_min:y_max+1, x_min:x_max+1] * 255
        resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_NEAREST)
        flattened = resized.flatten().reshape(1, -1)
        prediction = model.predict(flattened)[0]
        results.append((x_min, prediction, (x_min, y_min, x_max, y_max)))

    results.sort(key=lambda x: x[0])  # sola yakınlıkla sırala doğru tahmin için
    predicted_text = ''.join([char for _, char, _ in results])

    #Görsel üzerine çizim için
    for _, char, (x_min, y_min, x_max, y_max) in results:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
        cv2.putText(image, char, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return predicted_text, image

def main():
    model = load_model("model.pkl")
    image, binary = preprocess("deneme1.png")
    labels, label_id = bfs_label(binary)
    predicted_text, output_img = extract_and_predict(image.copy(), binary, labels, label_id, model)

    print("Tahmin:", predicted_text)

    output_img = cv2.resize(output_img, (800, 300))
    cv2.imshow("Tespit", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
