# Deep Audio Classification using FSC22 Dataset

## Overview

This project implements an end-to-end **audio classification pipeline** for environmental sound recognition using the FSC22 dataset. The system follows a research-based methodology including:

- Audio data augmentation
- Feature extraction using spectrogram representations
- Machine learning classification using Support Vector Machine (SVM)
- Deep learning classification using Convolutional Neural Networks (CNN)

The objective is to improve classification performance by applying feature engineering and modern machine learning techniques to audio data.

---

## Project Pipeline

The implemented system follows this workflow:

```
Raw Audio → Data Augmentation → Feature Extraction →
Feature Processing → Classification (SVM / CNN)
```

---

## Dataset

The FSC22 dataset contains labeled environmental audio recordings stored as `.wav` files along with metadata describing class labels.

### Dataset Structure

```
FSC22 Dataset
│
├── Audio Wise V1.0/
│   ├── *.wav files
│
├── Metadata/
│   ├── Metadata V1.0 FSC22.csv
```

### Metadata File

The metadata CSV provides:

- Source file name
- Dataset file name
- Class ID
- Class label

This file is used to map audio samples to their corresponding class labels.

---

## Data Augmentation

Deep learning models require large training datasets. To increase dataset size and reduce overfitting, pitch shifting augmentation is applied.

### Augmentation Method

For each audio file:

- Original sample
- Positive pitch shift (+2 steps)
- Negative pitch shift (−2 steps)

This increases the dataset size by generating additional training samples.

### Implementation

- Implemented using `librosa.effects.pitch_shift`
- Sampling rate preserved during augmentation

### Result

```
1 audio sample → 3 samples (original + augmented)
```

---

## Feature Extraction

Audio signals are transformed into time–frequency representations before classification.

### 1. Mel Spectrogram

Represents the power spectrum of audio mapped to the mel scale, which approximates human hearing.

#### Steps:
- Frame audio into overlapping windows
- Apply Fourier transform
- Map frequencies to mel scale
- Convert power values to decibel scale

#### Output:

```
(number of mel bands × time frames)
```

---

### 2. MFCC (Mel Frequency Cepstral Coefficients)

MFCCs describe the spectral characteristics of sound and are widely used in audio recognition.

#### Steps:
- Compute mel spectrogram
- Apply logarithmic scaling
- Perform discrete cosine transform

#### Output:

```
(number of coefficients × time frames)
```

---

## Feature Processing

Machine learning models require fixed-size feature vectors.

### Dimensionality Reduction

For each spectrogram:

```
Mean across time frames → 1D vector
```

Final feature vector:

```
MFCC mean + Mel mean → Combined feature vector
```

This converts 2D spectrograms into 1D features suitable for ML algorithms.

---

## Machine Learning Model — Support Vector Machine (SVM)

A Support Vector Machine classifier is used for ML-based classification.

### Why SVM?

- Handles nonlinear feature relationships
- Works well with small datasets
- Effective for audio classification tasks
- Uses kernel functions for complex decision boundaries

### Training Procedure

- 80% training data
- 20% validation data
- RBF kernel used
- Feature scaling applied using standardization

---

## Deep Learning Model — Convolutional Neural Network (CNN)

A 9-layer CNN architecture is implemented for spectrogram-based classification.

### CNN Architecture

- Multiple convolution layers
- Max pooling layers
- Fully connected dense layers
- Dropout for regularization
- Softmax output layer

### Input Representation

To create image-like inputs:

- Multiple spectrograms generated using different window sizes
- Combined into RGB-like representation

### Training Configuration

- 50 epochs
- Early stopping to prevent overfitting
- 80/20 train–validation split

---

## Train–Validation Split

The dataset is divided using the Pareto principle:

```
80% → Training
20% → Validation
```

This ensures sufficient data for learning while preserving unseen data for evaluation.

---

## Technologies Used

- Python
- NumPy
- Librosa (audio processing)
- Scikit-learn (SVM classifier)
- TensorFlow / Keras (CNN model)
- Kaggle Notebook environment

---

## Project Structure

```
project/
│
├── notebooks/
│   ├── data_augmentation.ipynb
│   ├── feature_extraction.ipynb
│   ├── svm_training.ipynb
│   ├── cnn_training.ipynb
│
├── README.md
```

---

## Results

The system demonstrates the effectiveness of:

- Data augmentation
- Spectrogram-based feature extraction
- Machine learning and deep learning classification methods

Performance depends on model configuration and dataset size.

---

## Future Improvements

- Hyperparameter tuning for SVM
- Additional data augmentation (time stretching, noise addition)
- Transfer learning using pretrained audio models
- Cross-validation experiments
- Model comparison (SVM vs CNN vs XGBoost)

---

## How to Run

1. Load dataset in Kaggle environment.
2. Run data augmentation pipeline.
3. Extract MFCC and Mel features.
4. Train SVM or CNN model.
5. Evaluate model performance.
