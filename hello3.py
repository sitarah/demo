import streamlit as st
import pandas as pd
import uuid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.colors as mcolors
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Clasifyr", layout="wide")

# File paths for saving and loading the model and vectorizer
MODEL_PATH = 'model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

# Ensure session state is initialized
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'vectorizer' not in st.session_state:
    st.session_state['vectorizer'] = None

if 'search_input' not in st.session_state:
    st.session_state['search_input'] = ''

if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None

if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

if 'y_pred' not in st.session_state:
    st.session_state['y_pred'] = None

# Function to load and display the CSV file
def load_and_display_csv(file):
    if file:
        df = pd.read_csv(file)
        st.write("Data Preview:")
        st.write(df.head())
        return df

# Function to train a machine learning model
def train_machine_learning_model(df):
    if df is not None:
        st.write("""
            <div style="display: flex; justify-content: space-around;">
                <div>
                    <h3>Training model on the following data:</h3>
                    {table}
                </div>
                <div>
                    <h3>Columns in the uploaded CSV file:</h3>
                    {columns}
                </div>
            </div>
        """.format(table=df.head().to_html(index=False), columns=pd.DataFrame(df.columns, columns=["Columns"]).to_html(index=False)), unsafe_allow_html=True)

        # Ensure the column names are correct
        if 'prompt' not in df.columns or 'genre' not in df.columns:
            st.error("The CSV file must contain 'prompt' and 'genre' columns.")
            return

        # Extract features and labels
        X = df['prompt']
        y = df['genre']

        # Convert text data into numerical data
        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        # Train a Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        report_df = pd.DataFrame(report).transpose()

        # Save the trained model and vectorizer to files
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        # Creating a table with the classification report data
        report_table = report_df.style.format("{:.2f}").set_table_styles([
            {'selector': 'th', 'props': [('font-size', '16px'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('font-size', '14px'), ('text-align', 'center')]},
            {'selector': 'thead th.col_heading', 'props': 'text-align: center;'}
        ]).set_properties(**{'background-color': '#f0f4f9', 'color': 'black', 'border-color': '#e3e7ec'}).to_html()

        # Displaying the table in the gradient div
        st.write(f"""
            <div style="background: linear-gradient(to bottom, #e3f2fd, #f8bbd0); border-radius: 10px; padding: 20px; margin-top: 20px; text-align: center; width: 80%; margin-left: auto; margin-right: auto;">
                <h2 style="text-align: center; font-family: Arial, sans-serif;">Classification Report:</h2>
                <div style="font-size: 16px; padding: 10px; background: #ffffff; border-radius: 10px; margin-top: 10px; text-align: centre; display: inline-block; width: 100%;">
                    {report_table}
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.session_state['model'] = model
        st.session_state['vectorizer'] = vectorizer
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred

# Function to make predictions
def make_prediction(model, vectorizer, prompt):
    # Check if the prompt is likely informative
    if len(prompt.strip()) < 3 or not any(char.isalpha() for char in prompt):
        return "Please provide an informative prompt."
    
    prompt_vectorized = vectorizer.transform([prompt])
    prediction = model.predict(prompt_vectorized)
    return prediction[0]

# Load the model and vectorizer if they exist
def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    return None, None

# Custom CSS for the rest of the page
st.markdown("""
    <style>
    .title {
        font-size: 50px;
        font-weight: 700;
        text-align: center;
        margin-top: 50px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .title .hello {
        background: linear-gradient(to right, #4F82EE, #C4ADE1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-right: 10px;
    }
    .title .there {
        background: linear-gradient(to right, #B56CA4, #D96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 24px;
        color: #999;
        text-align: center;
        margin-bottom: 50px;
    }
    .header {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        margin-bottom: 0px;
        padding-left: 10px;
        margin-top: 2px;
    }
    .logo {
        height: 50px;
        margin-right: 20px;
    }
    .thin-line {
        border: none;
        height: 0.5px;
        background-color: #E3E3E3;
        margin: 5px 0;
    }
    .app-name {
        font-size: 22px;
        font-weight: 700;
        color: #c41e3a;
    }
    .button {
        width: 100%;
        height: 150px;
        background-color: #f0f4f9; /* Background color */
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        padding: 20px;
        margin: 10px;
        cursor: pointer; /* Make the cards clickable */
        border: none;
        color: inherit;
    }
    .button .card-icon {
        font-size: 48px; /* Increase the size of the icon */
        margin-bottom: 10px;
        color: #6c757d;
    }
    .button .card-text {
        font-size: 18px; /* Increase the size of the text */
        font-weight: bold; /* Make the text bold */
    }
    .card-button:hover {
        background-color: #e3e7ec; /* Background color on hover */
    }
    .prediction-result {
        background: linear-gradient(to bottom, #e3f2fd, #f8bbd0);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        font-family: Arial, sans-serif;
        font-size: 16px;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }
    .spacer {
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header with Logo and App Name
st.markdown('<div class="header">'
            '<img class="logo" src="https://img.icons8.com/clouds/100/000000/database.png" alt="Logo">'  # Using an external icon for demonstration
            '<div class="app-name">Clasifyr</div>'
            '</div>', unsafe_allow_html=True)

# Add a thin line
st.markdown('<hr class="thin-line">', unsafe_allow_html=True)

# Title and subtitle with gradient
st.markdown('<div class="title"><span class="hello">Prompt</span><span class="there"> Classifier</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering Your Data: Classify and Predict with Precision</div>', unsafe_allow_html=True)

# Display prediction section when 'Make predictions on new data' is clicked
key = st.session_state.get('key', '')

if key == 'make_predictions':
    if st.session_state['model'] is None or st.session_state['vectorizer'] is None:
        # Load the model and vectorizer if available
        model, vectorizer = load_model_and_vectorizer()
        if model is not None and vectorizer is not None:
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vectorizer
        else:
            st.error("You need to train the model first.")
    if st.session_state['model'] is not None and st.session_state['vectorizer'] is not None:
        search_input = st.text_input("Enter a prompt for Prediction:", key='search_input_prediction')
        if st.button('Check Prediction', key='check_prediction_button'):
            prediction = make_prediction(st.session_state['model'], st.session_state['vectorizer'], search_input)
            st.markdown(f"""
                <div class="prediction-result">
                    {f'The predicted genre for the prompt "{search_input}" is "<b>{prediction}</b>"' if prediction != "Please provide an informative prompt." else prediction}
                </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)  # Spacer between output and buttons

# Cards in a row with space
cols = st.columns(4)

# Render the card buttons and handle button clicks
if cols[0].button('📂\nLoad the dataset', key='load_data', use_container_width=True):
    st.session_state['key'] = 'load_data'
if cols[1].button('📊\nTrain Model on dataset', key='train_model', use_container_width=True):
    st.session_state['key'] = 'train_model'
if cols[2].button('🔍\nMake predictions on new data', key='make_predictions', use_container_width=True):
    st.session_state['key'] = 'make_predictions'
if cols[3].button('📈\nVisualize the model\'s performance', key='visualize_performance', use_container_width=True):
    st.session_state['key'] = 'visualize_performance'

# Handle file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="file_uploader")

df = None

if key == 'load_data':
    df = load_and_display_csv(uploaded_file)
elif key == 'train_model':
    df = load_and_display_csv(uploaded_file)  # Ensure the data is loaded
    if df is not None:
        train_machine_learning_model(df)

elif key == 'visualize_performance':
    if st.session_state['y_test'] is None or st.session_state['y_pred'] is None:
        st.error("You need to train the model first.")
    else:
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        # Custom color map using the provided colors
        colors = ["#D96570", "#F4C6D9"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        
        # Adjusting the figure size to be smaller
        fig, ax = plt.subplots(figsize=(6, 4))  # Reduce the figure size
        
        # Customizing the heatmap with the custom color map
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False, 
                    annot_kws={"size": 10}, linewidths=0.2, linecolor='grey')
        
        ax.set_xlabel('Predicted Labels', fontsize=10)
        ax.set_ylabel('True Labels', fontsize=10)
        ax.set_title('Confusion Matrix', fontsize=12)
        
        # Adjust layout to prevent clipping of labels or title
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Center-align the image on the page with a fixed width
        st.markdown(f"<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(buf, width=750)  # Force the image to be 750px wide
        st.markdown(f"</div>", unsafe_allow_html=True)

# Custom style for file uploader area
st.markdown("""
    <style>
    .css-1f6kzdf {
        height: 300px;
        border: 1px dashed #ddd;
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer; /* Make the cards clickable */
        transition: all 0.3s ease;
    }
    .css-1f6kzdf:hover {
        border-color: #c41e3a;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)
