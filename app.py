import streamlit as st
import scikit-learn
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
import string

# Load and prepare your data here
data = pd.read_csv('breast-cancer.csv')  # Modify path as necessary
X = data.iloc[:, 2:]  # Assuming feature columns start from index 2
y = data.iloc[:, 1]   # Assuming label column is at index 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "SVC": SVC(probability=True),
    "K-Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gaussian NB": GaussianNB(),
    "Logistic Regression": LogisticRegression()
}

model_accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracies[name] = accuracy_score(y_test, y_pred)

API_KEY = 'AIzaSyCrOIklqgXwxh_3Y6zgQlj5RYlKqS_W6ag'
CSE_ID = 'e465b812daba544b5'

def main():
    st.set_page_config(layout="wide", page_title="Breast Health Analysis", page_icon="🏥")

    st.sidebar.title("Navigation")
    menu = ["Login", "Home", "Predict", "Model Comparison", "Interactive Analysis", "Medical Suggestion"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Handle user session
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if choice == "Login":
        login()

    if not st.session_state.logged_in:
        st.warning("Please login to access other functionalities.")
        return

    if choice == "Home":
        home()

    elif choice == "Predict":
        predict()

    elif choice == "Model Comparison":
        model_comparison()

    elif choice == "Interactive Analysis":
        train_own_model()

    elif choice == "Medical Suggestion":
        medical_suggestion()

def login():
    st.header("User Login")
    with st.form(key='user_login_form'):
        phone = st.text_input("Enter your mobile number")
        name = st.text_input("Enter your name")
        submit = st.form_submit_button("Submit")

    if submit:
        if phone and name:
          st.session_state.logged_in = True
          st.success("Login Successful")
        else:
            st.error("Please fill all fields")

def home():
    st.header("Welcome to the Breast Cancer Detection App")
    st.write("This application predicts breast cancer based on various input features and provides other health insights. Use the sidebar to navigate through different functionalities.")
    st.image("https://imgs.search.brave.com/3b8OeaPx9PWZQ8oghbj5lQDjPDEyUIwe1qwZaAQ2Fgc/rs:fit:500:0:0/g:ce/aHR0cHM6Ly9pbWcu/ZnJlZXBpay5jb20v/cHJlbWl1bS1waG90/by9oZWFsdGhjYXJl/LWJhY2tncm91bmQt/YmVhdXRpZnVsLWNv/bG9yZnVsLTNkLXBp/Y3R1cmUtZ2VuZXJh/dGl2ZS1haV8xNDY2/NzEtNzU1MTIuanBn/P3NpemU9NjI2JmV4/dD1qcGc", caption="Feel free to explore!")

def predict():
    st.header("Breast Cancer Prediction")
    features = []
    for col_name in X.columns:
        feature = st.number_input(f"Enter {col_name}", min_value=float(X[col_name].min()),
                                  max_value=float(X[col_name].max()), value=float(X[col_name].mean()))
        features.append(feature)

    selected_model = st.selectbox("Choose a model", list(models.keys()))
    model = models[selected_model]

    if st.button('Predict'):
        features_df = pd.DataFrame([features], columns=X.columns)
        prediction = model.predict(features_df)
        prediction_proba = model.predict_proba(features_df)

        st.subheader('Prediction:')
        result = "Malignant" if prediction[0] == 1 else "Benign"

        if (prediction_proba[0][1]*100.0) == (prediction_proba[0][0]*100.0) :
            st.write(result)
        elif (prediction_proba[0][1]*100.0) > (prediction_proba[0][0]*100.0) :
            result = "Malignant"
            st.write(result)
        else:
            result = "Benign"
            st.write(result)

        st.subheader('Prediction Probability:')
        st.write(f"Malignant: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Benign: {prediction_proba[0][0]*100:.2f}%")

def model_comparison():
    st.header("Model Accuracy Comparison")
    for model_name, accuracy in model_accuracies.items():
        st.write(f"{model_name}: {accuracy:.2f}")

    plt.figure(figsize=(10, 5))
    plt.bar(model_accuracies.keys(), model_accuracies.values(), color='blue')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies')
    st.pyplot(plt)

from sklearn.preprocessing import LabelEncoder

def train_own_model():
    st.title("Model Analysis / Tester")
    uploaded_file = st.file_uploader("Choose a CSV file to upload.")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        if st.checkbox("Remove Null Values"):
            data.dropna(inplace=True)
            st.write("Null values removed.")
            st.write(data.head())

        # Determine which columns need encoding
        if st.checkbox("Encode Categorical Data"):
            encodable_columns = [col for col in data.columns if data[col].dtype == 'object']
            encoder = LabelEncoder()
            for col in encodable_columns:
                data[col] = encoder.fit_transform(data[col])
            st.write("Categorical data encoded.")
            st.write(data.head())

        # Data slicing inputs
        strow = st.number_input("Row Start From:", min_value=0, step=1, format="%d")
        edrow = st.number_input("Row Ending At:", min_value=0, step=1, format="%d", value=len(data) - 1)
        stcol = st.number_input("Column Start From:", min_value=0, step=1, format="%d")
        edcol = st.number_input("Column Ending At:", min_value=0, step=1, format="%d", value=data.shape[1] - 1)
        edcoly = st.number_input("Prediction Column Index:", min_value=0, step=1, format="%d")

        X = data.iloc[strow:edrow + 1, stcol:edcol]
        y = data.iloc[strow:edrow + 1, edcoly]

        test_size = st.number_input("Test Size Data:", min_value=0.01, max_value=0.99, value=0.25, step=0.01, format="%.2f")
        rand_state = st.number_input("Random State:", min_value=0, step=1, format="%d")

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)
            st.write("Training and Testing datasets are ready.")

            # if st.checkbox("Apply Preprocessing"):
            #     from sklearn.preprocessing import StandardScaler
            #     sc = StandardScaler()
            #     X_train = sc.fit_transform(X_train)
            #     X_test = sc.transform(X_test)  # Use transform here to prevent data leakage

            # Model training and evaluation
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.metrics import mean_squared_error

            svc = SVC()
            dtr = DecisionTreeRegressor()
            svc.fit(X_train, y_train)
            dtr.fit(X_train, y_train)

            y_pred_svc = svc.predict(X_test)
            y_pred_dtr = dtr.predict(X_test)

            st.write("SVC MSE:", mean_squared_error(y_test, y_pred_svc))
            st.write("DTR MSE:", mean_squared_error(y_test, y_pred_dtr))

            st.write("The mean squared error (MSE) is a common way to measure the quality of predictions made by a machine learning model.\nA lower MSE generally indicates a better fit between the model's predictions and the actual outcomes.")
            st.write("Here's a general rule of thumb for interpreting\n\t\tMSE: 0.0 - 0.1: Excellent\n\t\t0.1 - 0.3: Good\n\t\t0.3 - 0.5: Acceptablen\n\t\tAbove 0.5: Poor")

            if mean_squared_error(y_test, y_pred_svc) == mean_squared_error(y_test, y_pred_dtr) :
              st.write("Both are Best Choice...")
            elif mean_squared_error(y_test, y_pred_svc) > mean_squared_error(y_test, y_pred_dtr) :
                st.write("The DTR {Decision Tree Regressor} Is Best Chocie For Training Your Model...")
            else:
                st.write("The SVC {Support Vector Classifire} Is Best Chocie To Training Your Model...")

def medical_suggestion():
    st.header("Medical Information / Suggestions")
    cancer_type = st.text_input("Enter Cancer Type ('m' for malignant or 'b' for benign)", "")
    if cancer_type:
        fetch_and_display_results(cancer_type)

def fetch_and_display_results(query):
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query + "cancer details"
    }
    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
    results = response.json()

    if "items" in results:
        for item in results['items']:
            st.write("###", item['title'])
            st.write(item['snippet'])
            if 'pagemap' in item and 'cse_image' in item['pagemap']:
                st.image(item['pagemap']['cse_image'][0]['src'], caption=item['title'])

if __name__ == "__main__":
    main()
