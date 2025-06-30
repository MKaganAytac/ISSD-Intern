import cv2
import numpy as np
from collections import deque

def gorseldef(path, threshold=120):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = np.where(gray > threshold, 0, 1).astype(np.uint8)
    return img, binary

def bfslabeldef(binary):
    h, w = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    label_id = 1

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
    return labels, label_id

def solyakinkenardef(labels, label_id):
    min_x = labels.shape[1] + 1
    leftmost_label = -1

    for label in range(1, label_id):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        x_min = np.min(xs)
        if x_min < min_x:
            min_x = x_min
            leftmost_label = label
    return leftmost_label

def renklidef(img, labels, leftmost_label):
    output = img.copy()
    h, w = labels.shape

    for y in range(h):
        for x in range(w):
            if labels[y, x] > 0:
                if labels[y, x] == leftmost_label:
                    output[y, x] = (0, 0, 255)
                else:
                    output[y, x] = (0, 255, 0)
    return output

def main():
    img, binary = gorseldef("geosekil.png")
    labels, label_id = bfslabeldef(binary)
    leftmost_label = solyakinkenardef(labels, label_id)
    output = renklidef(img, labels, leftmost_label)

    pencere = cv2.resize(output, (800, 600))
    cv2.imshow("Kutu", pencere)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
