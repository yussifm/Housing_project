Below is an updated **README.md** file that incorporates the full list of dependencies, explains the various models used, and provides setup instructions for both the ML model and the Flutter app.

```markdown
# Housing Price Prediction Project

This project consists of two main components:

1. **ML Model**: A Python-based machine learning solution for predicting housing prices in Bengaluru using multiple regression models.
2. **Flutter App**: A mobile application that loads the exported ML model (in ONNX format) to perform predictions locally.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [ML Model Setup](#ml-model-setup)
  - [Prerequisites](#prerequisites)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Converting the Model to ONNX](#converting-the-model-to-onnx)
- [Flutter App Setup](#flutter-app-setup)
  - [Prerequisites](#prerequisites-1)
  - [Integrating the ONNX Model](#integrating-the-onnx-model)
  - [Running the App](#running-the-app)
- [Data Link](#data-link)
- [License](#license)

---

## Overview

The **ML Model** component trains multiple regression models on housing data (using features such as `total_sqft`, `bath`, `balcony`, and `size`) to predict house prices. The models include traditional regression methods, tree-based ensembles, gradient boosting frameworks, and a customizable neural network using TensorFlow/Keras. Once trained, the model is saved using `joblib` and can be converted to the ONNX format to enable cross-platform deployment.

The **Flutter App** loads the ONNX model from its assets and uses the `onnxruntime` package to perform predictions directly on the device.

---

## Project Structure

```
HousingPricePrediction/
├── ml_model/
│   ├── data/
│   │   └── Bengaluru_House_Data.csv   # Dataset
│   ├── train_model.py                  # Script to train and evaluate multiple ML models
│   ├── convert_to_onnx.py              # Converts the trained model to ONNX format
│   ├── house_price_model.pkl           # Saved scikit-learn model [generated]
│   ├── scaler.pkl                      # Saved scaler for preprocessing [generated]
│   ├── house_price_model.onnx          # ONNX model file [generated]
│   ├── requirements.txt                # Python dependencies for ml_model
│
├── flutter_app/
│   ├── assets/
│   │   └── house_price_model.onnx      # Copy of the ONNX model file
│   ├── lib/
│   │   ├── main.dart                   # Main entry point for the Flutter app
│   │   ├── predict.dart                # Contains logic to load and run the ONNX model
│   │   └── ui/
│   │       └── home_screen.dart        # UI components for the app
│   ├── pubspec.yaml                    # Flutter configuration file (includes assets)
│
└── README.md                           # This file
```

---

## Models Used

The project includes several regression models:

1. **Linear Regression (Baseline Model)**  
   *Assumes a linear relationship between features and target; simple, interpretable, and serves as a baseline.*

2. **Decision Tree Regressor**  
   *Builds a tree-like structure that splits data based on feature values; effective for non-linear relationships but may overfit if not tuned.*

3. **Random Forest Regressor**  
   *An ensemble of decision trees that averages predictions; improves robustness and reduces overfitting compared to a single tree.*

4. **XGBoost Regressor**  
   *A gradient boosting method that builds models sequentially to correct previous errors; known for its performance, speed, and regularization techniques.*

5. **LightGBM Regressor**  
   *A gradient boosting framework optimized for speed and efficiency, especially on large datasets; utilizes histogram-based algorithms for faster computation.*

6. **Custom Neural Network (TensorFlow/Keras)**  
   *A deep learning model capable of capturing complex non-linear relationships; highly customizable in terms of architecture and hyperparameters.*

---

## ML Model Setup

### Prerequisites

- Python 3.x
- Install required packages using:
  ```bash
  pip install -r ml_model/requirements.txt
  ```
  
The **`requirements.txt`** in the `ml_model` folder includes:
```
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
skl2onnx
onnx
tensorflow
onnx-tf
xgboost
lightgbm
```

### Data Preparation

- Place your dataset (`Bengaluru_House_Data.csv`) in the `ml_model/data/` folder.
- Ensure the dataset includes columns such as `total_sqft`, `bath`, `balcony`, `size`, and `price`.
- The training script handles data cleaning, such as removing missing values and converting textual data to numeric values.

### Training the Model

- Run the training script from the `ml_model` folder:
  ```bash
  python train_model.py
  ```
- This script:
  - Loads and cleans the dataset.
  - Trains multiple models (Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and a Custom Neural Network).
  - Evaluates each model (calculating MAE and RMSE).
  - Saves the selected model (`house_price_model.pkl`) and scaler (`scaler.pkl`).
  - Generates visualization charts for data distribution, feature relationships, and prediction performance.

### Converting the Model to ONNX

- After training, convert the model to ONNX format by running:
  ```bash
  python convert_to_onnx.py
  ```
- This will create the file `house_price_model.onnx` used by the Flutter app.
- Copy the `house_price_model.onnx` file into the `flutter_app/assets/` folder.

---

## Flutter App Setup

### Prerequisites

- Flutter SDK installed.
- Ensure the following dependency is included in your `pubspec.yaml`:
  ```yaml
  dependencies:
    flutter:
      sdk: flutter
    onnxruntime: ^0.3.0  # or latest version available
  ```

### Integrating the ONNX Model

- Place the ONNX model file in `flutter_app/assets/house_price_model.onnx`.
- The Flutter app loads the model using the `onnxruntime` package.
- In your Dart code (e.g., in `predict.dart`):
  - Load the model from assets.
  - Prepare input data (a list of floats corresponding to the model's features).
  - Run inference using ONNX Runtime to obtain the predicted price.

### Running the App

- Navigate to the `flutter_app` folder:
  ```bash
  cd flutter_app
  ```
- Run the Flutter app:
  ```bash
  flutter run
  ```

---

## Data Link

You can access the dataset via Kaggle:  
[Housing Data on Kaggle](https://www.kaggle.com/code/mfaisalqureshi/banglore-house-price-prediction/input)

---

## License

This project is licensed under the MIT License.
```

