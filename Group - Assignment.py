import cv2
import numpy as np

# Real-time camera acquisition
# def capture_image():
#     # Turn on the camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Unable to turn on the camera")
#         return None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Unable to capture image")
#             break

#         # Show raw image
#         cv2.imshow('Original Image', frame)

#         # Press 'ESC' to exit
#         if cv2.waitKey(1) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return frame  # Returns the last frame of the image

# Image preprocessing
def preprocess_image(image):
    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # The denoising process is complete
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarization processing
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    
    # Edge detection
    edges = cv2.Canny(blurred, 100, 200) # Canny edge detection
    
    # Morphological processing: closed operation
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find the outline
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Delineate
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Resize
    resized_image = cv2.resize(image, (400, 400))

    return closing, contours, resized_image


if __name__ == "__main__":
    # # Capture image    
    # image = capture_image()
    image=cv2.imread('CV\Group - Assignment\Strawberry.bmp')
    if image is not None:
        # Image preprocessing
        processed_image, contours, resized_image = preprocess_image(image)

        # Display processing result
        cv2.imshow('Processed Image', processed_image)
        cv2.imshow('Resized Image', resized_image)  

        # Press 'ESC' to exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Unable to capture image")