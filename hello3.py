import streamlit as st
import pandas as pd
import uuid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Clasifyr", layout="wide")

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

        report_html = """
            <div style="background: linear-gradient(to bottom, #e3f2fd, #f8bbd0); border-radius: 10px; padding: 20px; margin-top: 40px; text-align: center;">
                <h2 style="text-align: center; font-family: Arial, sans-serif;">Classification Report:</h2>
                <div style="font-size: 16px; padding: 10px; background: #ffffff; border-radius: 10px; margin-top: 10px; text-align: left; display: inline-block;">
                    <pre style="font-family: Arial, sans-serif;">
<b>Precision    Recall  F1-Score   Support</b>
<br>
Monitoring  {monitoring_precision:.2f}     {monitoring_recall:.2f}  {monitoring_f1:.2f}  {monitoring_support}
<br>
Orchestration  {orchestration_precision:.2f}     {orchestration_recall:.2f}  {orchestration_f1:.2f}  {orchestration_support}
<br>
<b>Accuracy</b>  {accuracy:.2f}
<b>Macro Avg</b>  {macro_avg_precision:.2f}     {macro_avg_recall:.2f}  {macro_avg_f1:.2f}  
                    </pre>
                </div>
            </div>
        """.format(
            monitoring_precision=report['Monitoring']['precision'],
            monitoring_recall=report['Monitoring']['recall'],
            monitoring_f1=report['Monitoring']['f1-score'],
            monitoring_support=report['Monitoring']['support'],
            orchestration_precision=report['Orchestration']['precision'],
            orchestration_recall=report['Orchestration']['recall'],
            orchestration_f1=report['Orchestration']['f1-score'],
            orchestration_support=report['Orchestration']['support'],
            accuracy=accuracy,
            macro_avg_precision=report['macro avg']['precision'],
            macro_avg_recall=report['macro avg']['recall'],
            macro_avg_f1=report['macro avg']['f1-score']
        )

        st.write(report_html, unsafe_allow_html=True)

        st.session_state['model'] = model
        st.session_state['vectorizer'] = vectorizer
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred

# Function to make predictions
def make_prediction(model, vectorizer, prompt):
    prompt_vectorized = vectorizer.transform([prompt])
    prediction = model.predict(prompt_vectorized)
    return prediction[0]

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
    .search-bar-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        width: 100%;
        color: #F0F4F9;
    }
    .search-input-container {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        color: #F0F4F9;
    }
    .search-input {
        width: 100%;
        height: 60px; /* Increased height */
        border: 1px solid #ddd;
        border-radius: 5px 0 0 5px; /* Rounded corners on the left side */
        padding: 0 10px;
        color: #F0F4F9;
    }
    .search-button {
        background-color: #c41e3a;
        color: white;
        border: none;
        padding: 0 20px;
        height: 60px; /* Increased height */
        border-radius: 0 5px 5px 0; /* Rounded corners on the right side */
        cursor: pointer;
        white-space: nowrap;
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
        font-size: 14px;
        font-weight: bold;
        padding: 20px;
        margin: 10px;
        cursor: pointer; /* Make the cards clickable */
        border: none;
        color: inherit;
    }
    .card-button:hover {
        background-color: #e3e7ec; /* Background color on hover */
    }
    .card-icon {
        font-size: 30px;
        margin-bottom: 10px;
        color: #6c757d;
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
st.markdown('<div class="title"><span class="hello">Hello</span><span class="there"> There</span></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Empowering Your Data: Classify and Predict with Precision</div>', unsafe_allow_html=True)

# Search bar and button in the same row
st.markdown('<div class="search-bar-container">', unsafe_allow_html=True)
search_col1, search_col2 = st.columns([5, 1])
with search_col1:
    search_input = st.text_input("", key='search_input', value=st.session_state['search_input'], placeholder="Enter a prompt here", label_visibility="collapsed")
with search_col2:
    if st.button('Check', key='check_button'):
        if st.session_state['model'] is None or st.session_state['vectorizer'] is None:
            st.error("You need to train the model first.")
        else:
            prediction = make_prediction(st.session_state['model'], st.session_state['vectorizer'], search_input)
            st.write(f"The predicted genre for the prompt '{search_input}' is '{prediction}'")
st.markdown('</div>', unsafe_allow_html=True)

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

# Handle card clicks on the backend
key = st.session_state.get('key', '')

df = None

if key == 'load_data':
    df = load_and_display_csv(uploaded_file)
elif key == 'train_model':
    df = load_and_display_csv(uploaded_file)  # Ensure the data is loaded
    if df is not None:
        train_machine_learning_model(df)
elif key == 'make_predictions':
    if st.session_state['model'] is None or st.session_state['vectorizer'] is None:
        st.error("You need to train the model first.")
    else:
        search_input = st.text_input("Enter a prompt for prediction:", key='search_input_prediction')
        if st.button('Check Prediction', key='check_prediction_button'):
            prediction = make_prediction(st.session_state['model'], st.session_state['vectorizer'], search_input)
            st.write(f"The predicted genre for the prompt '{search_input}' is '{prediction}'")
elif key == 'visualize_performance':
    if st.session_state['y_test'] is None or st.session_state['y_pred'] is None:
        st.error("You need to train the model first.")
    else:
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig, ax = plt.subplots(figsize=(3, 2))  # Further adjust the figure size here
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)  # Remove color bar for compactness
        ax.set_xlabel('Predicted Labels', fontsize=10)
        ax.set_ylabel('True Labels', fontsize=10)
        ax.set_title('Confusion Matrix', fontsize=12)
        st.pyplot(fig)

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
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .css-1f6kzdf:hover {
        border-color: #c41e3a;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)
