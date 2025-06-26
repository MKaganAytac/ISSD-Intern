import cv2
import numpy as np
from collections import deque

img = cv2.imread("geosekil.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = np.where(gray > 120, 0, 1).astype(np.uint8)

labels = np.zeros_like(binary, dtype=np.int32)
label_id = 1
h, w = binary.shape

for y in range(h):
    for x in range(w):
        if binary[y, x] == 1 and labels[y, x] == 0:
            q = deque()
            q.append((y, x))
            labels[y, x] = label_id

            while q:
                cy, cx = q.popleft()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h) and (0 <= nx < w):
                            if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = label_id
                                q.append((ny, nx))
            label_id += 1

output = img.copy()
min_x = w + 1
sol_yakin_kenar = -1

for label in range(1, label_id):
    ys, xs = np.where(labels == label)

    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)

    if x_min < min_x:
        min_x = x_min
        sol_yakin_kenar = label

for y in range(h):
    for x in range(w):
        if labels[y, x] > 0:
            if labels[y, x] == sol_yakin_kenar:
                output[y, x] = (0, 0, 255)
            else:
                output[y, x] = (0, 255, 0)

pencere = cv2.resize(output, (800, 600))
cv2.imshow("Kutu", pencere)
cv2.waitKey(0)
cv2.destroyAllWindows()
