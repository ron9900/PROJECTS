# PROJECTS
# 🧬 Embryo Quality Prediction using Deep Learning

This project aims to automate the classification of human embryo images based on their developmental stage and quality grade using deep learning. It assists embryologists and fertility clinics in selecting the most viable embryos during the IVF process.

## 🚀 Overview

- **Goal:** Predict embryo quality and stage (e.g., 8-cell, Morula, Blastocyst) and assign grades (A, B, C).
- **Model:** Fine-tuned DenseNet-201 for high-accuracy image classification.
- **Deployment:** Web app built using Streamlit for real-time predictions.
- **Dataset:** Custom annotated embryo image dataset.

## 🧠 Key Features

- 📸 **Image Preprocessing:** Resizing, normalization, and augmentation.
- 🧩 **Transfer Learning:** DenseNet-201 with fine-tuning for domain-specific performance.
- ⚖️ **Class Imbalance Handling:** Data augmentation and class weighting.
- 📊 **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- 🌐 **Deployment:** Streamlit app with interactive UI.

## 📁 Project Structure

embryo-quality-prediction/
│
├── data/ # Dataset and annotation files &#8595
├── notebooks/ # Jupyter notebooks for EDA and model training
├── models/ # Saved model weights
├── app/ # Streamlit app code
├── utils/ # Helper scripts
├── requirements.txt # Required Python packages
├── README.md # Project documentation
└── main.py # Entrypoint for model training/testing



## 🔧 Tech Stack

- **Languages:** Python
- **Libraries:** TensorFlow, Keras, OpenCV, scikit-learn, Streamlit, NumPy, Matplotlib
- **Model:** DenseNet-201 (Transfer Learning)
- **Tools:** Google Colab, Roboflow (for annotation), Git, Streamlit, Docker (optional)

## 🖥️ Streamlit App Preview

<p align="center">
  <img src="app_preview.gif" width="600"/>
</p>

