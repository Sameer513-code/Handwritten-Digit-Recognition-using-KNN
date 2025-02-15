import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import cv2

class DigitRecognizerApp:

    def preprocess_image(self, img):
        # Convert to grayscale (L mode ensures 8-bit pixels)
        img = img.convert("L")  

        # Resize to 28x28 pixels (same as MNIST dataset)
        img = img.resize((28, 28))

        # Convert image to NumPy array
        img = np.array(img)

        # Apply binary thresholding
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

        # Find bounding box of the digit
        coords = cv2.findNonZero(img)  # Find non-zero pixels
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box

        # Crop the region containing the digit
        img = img[y:y+h, x:x+w]

        # Resize back to 28x28 (preserving aspect ratio)
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

        # Create a blank 28x28 image and paste the resized digit in the center
        blank_image = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - img.shape[1]) // 2
        y_offset = (28 - img.shape[0]) // 2
        blank_image[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

        # Normalize pixel values (MNIST uses values between 0 and 1)
        blank_image = blank_image / 255.0  

        # Flatten image to match KNN input format (1D array of 784 values)
        blank_image = blank_image.reshape(1, -1)  

        return blank_image