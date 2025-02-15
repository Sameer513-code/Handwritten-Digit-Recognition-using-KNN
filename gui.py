import numpy as np
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw, ImageOps
import cv2
import matplotlib.pyplot as plt


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.button.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+10, y+10, fill="black", width=10)
        self.draw.ellipse([x, y, x+10, y+10], fill=0)

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

    def predict_digit(self):
        # Convert canvas to image
        img = self.image

        # Preprocess image before prediction
        img = self.preprocess_image(img)

        # Debugging: Show the processed image
        plt.imshow(img.reshape(28, 28), cmap="gray")
        plt.show()

        print(f"Processed Image Shape: {img.shape}")

        # Predict digit using KNN
        #pred = knn.predict(img)

        # Display prediction
        tk.Label(self.root, text=f"Prediction: {pred[0]}", font=("Arial", 20)).pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
