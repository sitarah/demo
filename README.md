# Clasifyr - Prompt Classifier

## Overview

**Clasifyr** is a Streamlit-based application designed to classify text prompts into predefined categories such as "Monitoring" and "Orchestration". The app allows users to upload a CSV file containing prompts and their corresponding categories, train a machine learning model on this data, and then make predictions on new prompts. The application also supports saving the trained model for future predictions without needing to retrain.

## Features

- **CSV Upload & Data Preview**: Upload a CSV file containing text prompts and their respective categories. The app previews the data for quick inspection.
- **Model Training**: Train a Naive Bayes classifier on the uploaded dataset. The app saves the trained model and vectorizer for future use.
- **Prediction**: Enter a prompt to classify it as "Monitoring" or "Orchestration". If the input is uninformative, the app guides the user to provide a more meaningful prompt.
- **Model Persistence**: The trained model is saved locally in `.joblib` format, allowing for predictions without retraining.
- **Performance Visualization**: Visualize the model's performance through a confusion matrix.

## Installation

To run the Clasifyr application locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sitarah/demo.git
   cd demo
