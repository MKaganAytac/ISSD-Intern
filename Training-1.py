import cv2
import numpy as np

img = cv2.imread("geosekil.png")
gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
font= cv2.FONT_HERSHEY_COMPLEX

a, thresh = cv2.threshold(gri, 127, 255, cv2.THRESH_BINARY)
kontur, b = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in kontur:
    e = 0.01*cv2.arcLength(i, True)
    approx =cv2.approxPolyDP(i, e, True)
    cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    print("köşeler", len(approx))
    print(approx)

resized_img = cv2.resize(img, (800, 600))
cv2.imshow("pencere", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
