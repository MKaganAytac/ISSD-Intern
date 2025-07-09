import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("deneme1.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 3)

text = pytesseract.image_to_string(gray, lang='eng')

print("Çıktı:")
print(text)

cv2.imshow("deneme1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
