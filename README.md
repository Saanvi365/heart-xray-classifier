# Heart X-Ray Classifier

A machine learning model that classifies chest X-rays as NORMAL or PNEUMONIA, built to turn messy medical imaging data into something a doctor could act on.

## What it does

- Loads and preprocesses the Chest X-Ray Pneumonia dataset from Kaggle (5,000+ labeled images)
- Explores the data: sample image visualization, image dimension and format analysis, class distribution
- Applies data augmentation (rotation, shifts, zoom, horizontal flips) to balance and expand the training set
- Trains a convolutional neural network, then experiments with transfer learning using VGG16 and ResNet50

## Tech stack

Python, TensorFlow/Keras, OpenCV, NumPy, scikit-learn, Matplotlib

## Running it

Open `heart_xray_classifier.ipynb` in Google Colab (GPU runtime recommended). The notebook downloads the dataset automatically via `kagglehub`.

Dataset: [Chest X-Ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
