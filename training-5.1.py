import cv2
import joblib
import numpy as np

model = joblib.load("model.pkl")

image = cv2.imread("deneme1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


blurred = cv2.GaussianBlur(gray, (5, 5), 0)


thresh = cv2.adaptiveThreshold(
    blurred, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)


kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

results = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if 15 < w < 80 and 20 < h < 100:  #gerçek dünyaya yakın olması için
        roi = thresh[y:y+h, x:x+w]
        resized = cv2.resize(roi, (32, 32)).flatten().reshape(1, -1)
        prediction = model.predict(resized)[0]
        results.append((x, prediction))


results.sort(key=lambda tup: tup[0])
cumle = ''.join([char for (_, char) in results])
print("Tahmin:", cumle)


for x, char in results:
    cv2.putText(image, char, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cv2.imshow("Tahmin", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
