import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to load data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Function for feature engineering
def feature_engineering(df):
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
    df = df.drop(columns=['sepal width (cm)'])
    return df

# Function to train model and return metrics
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return model, scaler, accuracy, precision, recall, f1, y_pred, y_test

# Set up the Streamlit app
st.set_page_config(page_title="Machine Learning Pipeline Demo", layout="wide")

# CSS to set background color and button styles
page_bg_color = '''
<style>
[data-testid="stAppViewContainer"] {
    background-color: #BFDDDF;  /* Change this to your desired background color */
    color: black;
}
.stButton > button {
    background-color: #70BDC2;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    cursor: pointer;
    width: 200px;
    margin: 10px auto;
    display: block;
}
.stButton > button:hover {
    background-color: #5aa1a4;
}
</style>
'''

st.markdown(page_bg_color, unsafe_allow_html=True)

# App title and description
st.markdown(
    """
    <div style="background-color: #70BDC2; padding: 5px;">
        <h1 style="text-align: center; color: white;">PROJECT DEMO</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state if not already
if 'df' not in st.session_state:
    st.session_state.df = None
if 'original_metrics' not in st.session_state:
    st.session_state.original_metrics = None
if 'engineered_metrics' not in st.session_state:
    st.session_state.engineered_metrics = None
if 'confusion_matrix' not in st.session_state:
    st.session_state.confusion_matrix = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Vertical buttons
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("Load Data"):
        st.session_state.df = load_data()
        st.write("### Iris Dataset")
        st.write(st.session_state.df.head())

    if st.button("Feature Engg"):
        if st.session_state.df is not None:
            st.session_state.df = feature_engineering(st.session_state.df)
            st.write("### Dataset after Feature Engineering")
            st.write(st.session_state.df.head())
        else:
            st.write("Please load the data first.")

    if st.button("ML Training"):
        if st.session_state.df is not None:
            # Train with original features
            original_df = load_data()
            model, scaler, orig_accuracy, orig_precision, orig_recall, orig_f1, _, _ = train_model(original_df)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.original_metrics = (orig_accuracy, orig_precision, orig_recall, orig_f1)

            # Train with engineered features
            engineered_df = feature_engineering(load_data())
            model, scaler, eng_accuracy, eng_precision, eng_recall, eng_f1, _, _ = train_model(engineered_df)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.engineered_metrics = (eng_accuracy, eng_precision, eng_recall, eng_f1)

            st.write("## Model Performance with Original Features")
            st.write(f"Accuracy: {st.session_state.original_metrics[0]:.2f}")
            st.write(f"Precision: {st.session_state.original_metrics[1]:.2f}")
            st.write(f"Recall: {st.session_state.original_metrics[2]:.2f}")
            st.write(f"F1 Score: {st.session_state.original_metrics[3]:.2f}")

            st.write("## Model Performance with Engineered Features")
            st.write(f"Accuracy: {st.session_state.engineered_metrics[0]:.2f}")
            st.write(f"Precision: {st.session_state.engineered_metrics[1]:.2f}")
            st.write(f"Recall: {st.session_state.engineered_metrics[2]:.2f}")
            st.write(f"F1 Score: {st.session_state.engineered_metrics[3]:.2f}")
        else:
            st.write("Please load the data and perform feature engineering first.")

    if st.button("Inference"):
        uploaded_file = st.file_uploader("Upload a CSV file for inference", type=["csv"])
        if uploaded_file is not None:
            st.write("File uploaded successfully.")  # Debug statement
            unseen_data = pd.read_csv(uploaded_file)
            st.write("### Unseen Data")
            st.write(unseen_data.head())

            # Ensure the unseen data has the same feature engineering applied
            if 'petal width (cm)' in unseen_data.columns and 'petal length (cm)' in unseen_data.columns:
                unseen_data['petal_ratio'] = unseen_data['petal length (cm)'] / unseen_data['petal width (cm)']
                if 'sepal width (cm)' in unseen_data.columns:
                    unseen_data = unseen_data.drop(columns=['sepal width (cm)'])
                st.write("Feature engineering applied to unseen data.")  # Debug statement

                # Scale the unseen data
                X_unseen = st.session_state.scaler.transform(unseen_data)
                predictions = st.session_state.model.predict(X_unseen)

                # Map predictions to species names
                species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
                predicted_species = [species_mapping[pred] for pred in predictions]

                st.write("### Predictions")
                st.write(predicted_species)
            else:
                st.write("The uploaded data does not have the required columns.")

    if st.button("UI Rendering or Monitoring"):
        if st.session_state.df is not None:
            model, scaler, accuracy, precision, recall, f1, y_pred, y_test = train_model(st.session_state.df)
            st.session_state.confusion_matrix = confusion_matrix(y_test, y_pred)
            
            st.write("## Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(st.session_state.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        else:
            st.write("Please load the data and train the model first.")
