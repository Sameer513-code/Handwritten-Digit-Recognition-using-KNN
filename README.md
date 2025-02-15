# Handwritten Digit Recognition using KNN

## 📌 Overview
This project is a simple handwritten digit recognition system using K-Nearest Neighbors (KNN) classifier. Users can draw digits on a Tkinter GUI, and the model predicts the digit based on the MNIST dataset. The project involves image processing with OpenCV and NumPy to match the input format of the dataset.
<br>
<br>
## 🚀 Features

- Draw a digit on a Tkinter GUI

- Preprocessing: Grayscale conversion, thresholding, cropping, resizing

- Prediction using KNN trained on the MNIST dataset (Achieved Model Accuracy of 97.4%)

- Real-time visualization of the processed image<br>
<br>

## 📂 Project Structure
├── main.py ------------> Main script to run the application<br>
├── model.py------------> KNN model training and prediction<br>
├── preprocess.py-------> Image processing functions<br>
├── gui.py--------------> Tkinter GUI for drawing digits<br>
├── README.md-----------> Project documentation<br>
<br>
## 🛠️ Tools & Technologies Used

- Python (Primary language)

- OpenCV (cv2): Image processing

- NumPy (np): Array manipulations

- Tkinter: GUI for user input

- Scikit-learn (KNN): Machine learning model
<br>

## ⚠️Issues I Encountered
- Incorrect Predictions for Certain Digits
- Numbers 6, 7, 8, 9 were often misclassified.
- 8 was detected as 3, and 9 as 4.
<br>

## 🤔Potential Causes:
- Thresholding Issue → Binarization might be incorrect.
- Cropping Issue → Incorrect bounding box adjustments.
- Normalization Issue → Pixel values may not align with MNIST.
- KNN Sensitivity → Distorted digits can affect performance.
<br>

## 🔧 Solutions & Improvements
- Refine Image Preprocessing: Ensure digit format matches MNIST.
- Fix Thresholding & Cropping: Avoid unnecessary distortions.
- Experiment with Other Models: Consider Neural Networks for better accuracy.
<br>

<h3>🏁 Want to Run the Project ?</h3>

Clone the Repository

```
git clone https://github.com/yourusername/handwritten-digit-recognition.git cd handwritten-digit-recognition
```

## 📌 Future Enhancements

- Train and test a Neural Network (e.g., CNN)
- Improve GUI usability and user experience
- Implement real-time drawing smoothing
- Add a dataset expansion for better accuracy
