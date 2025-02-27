# Waste Classification with PyTorch & Node.js

This project is a **web application** built using **Node.js, Express, and EJS**, where users can upload an image, and the backend will process it through a **PyTorch ONNX model** to classify the image.

## Features

- Upload an image through a web interface.
- Convert the image to a standardized format (3, 224, 224).
- Pass the image through a **PyTorch ONNX model** for classification.
- Return and display the predicted label to the user.

## Installation Guide

### 1️⃣ **Clone the Repository**

```sh
git clone https://github.com/Tushnik1/WasteClassifier.git
cd yourproject
```

### 2️⃣ **Install Dependencies**

#### Backend (Node.js)

```sh
npm install express multer ejs sharp onnxruntime-node
```

#### Python (For Preprocessing & ONNX Model Creation)

```sh
pip install torch torchvision pillow onnx onnxruntime
```

### 3️⃣ **Project Structure**

```
project_root/
│── models/                 # ONNX model storage
│   ├── model.onnx          # Trained PyTorch ONNX model
│
│── uploads/                # Uploaded images (temporary storage)
│
│── public/                 # Static assets (CSS, JS, images)
│   ├── styles.css          # Stylesheet for frontend
│
│── views/                  # Frontend templates (EJS)
│   ├── index.ejs           # Upload page
│   ├── result.ejs          # Result display page
│
│── server.js               # Main Express server file
│── dataset_loader.py       # Python script to preprocess dataset
│── convert_model.py        # Python script to convert PyTorch to ONNX
│── README.md               # Documentation
```

## Usage

### **1️⃣ Start the Server**

```sh
node server.js
```

### **2️⃣ Open in Browser**

Visit: `http://localhost:3000`

### **3️⃣ Upload an Image**

- The app will automatically process the image and return a label.
- Only **JPEG** and **PNG** files are allowed.

## Dataset Processing (PyTorch)

To prepare the dataset:
- Convert them to `(3,224,224)` tensors
- Assigns labels (`0` for 'R', `1` for 'O')

## Convert PyTorch Model to ONNX

If you have a `.pth` model, convert it using:

```python
import torch

# Load your trained PyTorch model
model = torch.load("your_model.pth")
model.eval()

# Dummy input tensor (adjust shape as needed)
dummy_input = torch.randn(1, 3, 224, 224)

# Convert to ONNX format
torch.onnx.export(model, dummy_input, "model.onnx", export_params=True, opset_version=11, do_constant_folding=True, input_names=['input'], output_names=['output'])

```

This will save an `onnx` model inside the `models/` folder.


### Dataset
<br>
Link to dataset: <a href="https://www.kaggle.com/datasets/techsash/waste-classification-data/data">Click here</a>
DOI citation: --


