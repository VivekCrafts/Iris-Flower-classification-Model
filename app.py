import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score


# Load and preprocess data
data = pd.read_csv('D:\\Machine Learning\\Iris.csv')  # Replace with your dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("ML Model Deployment with Streamlit")
st.write("This app lets you make predictions using the trained Logistic Regression model.")

# Sidebar for user input
st.sidebar.header("Input Features")
input_data = {}
for col, name in enumerate(data.columns[1:-1]):
    input_data[name] = st.sidebar.number_input(f"{name}", value=0.0)

# Predict button
if st.button("Predict"):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)
    st.write(f"Prediction: {prediction[0]}")
# Model performance
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy_score(model.predict(X_test),y_test):.2f}")

# Confusion Matrix
if st.checkbox("Show Confusion Matrix"):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    st.pyplot(disp.figure_)
