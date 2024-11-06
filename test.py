import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Importing the image 
image = cv2.imread('image.jpeg')
if image is None:
    print("无法读取指定的图片，请检查文件路径和格式。")
else:
    cv2.imshow('Original Image', image)  
    cv2.waitKey(0)
