<h1>Handwritten Digit Recognition using KNN</h1>

<h3>📌 Overview</h3>
This project is a simple handwritten digit recognition system using K-Nearest Neighbors (KNN) classifier. Users can draw digits on a Tkinter GUI, and the model predicts the digit based on the MNIST dataset. The project involves image processing with OpenCV and NumPy to match the input format of the dataset.
<br>
<br>
<h3>🚀 Features</h3>

- Draw a digit on a Tkinter GUI

- Preprocessing: Grayscale conversion, thresholding, cropping, resizing

- Prediction using KNN trained on the MNIST dataset (Achieved Model Accuracy of 97.4%)

- Real-time visualization of the processed image
<br>
<h3>📂 Project Structure</h3>

├── main.py              # Main script to run the application<br>
├── model.py             # KNN model training and prediction<br>
├── preprocess.py        # Image processing functions<br>
├── gui.py               # Tkinter GUI for drawing digits<br>
├── README.md            # Project documentation<br>
<br>
<br>
<h3>🛠️ Tools & Technologies Used</h3>

Python (Primary language)

OpenCV (cv2): Image processing

NumPy (np): Array manipulations

Tkinter: GUI for user input

Scikit-learn (KNN): Machine learning model
<br>
<br>
<h3>⚠️Issues I Encountered</h3>

- Incorrect Predictions for Certain Digits
- Numbers 6, 7, 8, 9 were often misclassified.
- 8 was detected as 3, and 9 as 4.
<br>
<br>
<h3>Potential Causes:</h3>

- Thresholding Issue → Binarization might be incorrect.
- Cropping Issue → Incorrect bounding box adjustments.
- Normalization Issue → Pixel values may not align with MNIST.
- KNN Sensitivity → Distorted digits can affect performance.
<br>
<br>
<h3>🔧 Solutions & Improvements</h3>

- Refine Image Preprocessing: Ensure digit format matches MNIST.
- Fix Thresholding & Cropping: Avoid unnecessary distortions.
- Experiment with Other Models: Consider Neural Networks for better accuracy.
<br>
<br>
<h3>🏁 Want to Run the Project ?</h3>

Clone the Repository

```
git clone https://github.com/yourusername/handwritten-digit-recognition.git cd handwritten-digit-recognition
```
<br>
<br>
<h3>📌 Future Enhancements</h3>

- Train and test a Neural Network (e.g., CNN)
- Improve GUI usability and user experience
- Implement real-time drawing smoothing
- Add a dataset expansion for better accuracy
