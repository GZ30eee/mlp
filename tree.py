import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Function to load random dataset (Iris for classification, California housing for regression)
def load_random_data():
    dataset_choice = st.radio("Choose a Dataset", ["Iris (Classification)", "California Housing (Regression)", "Upload Your Own CSV"])
    
    if dataset_choice == "Iris (Classification)":
        # Load Iris dataset (classification)
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = pd.Categorical.from_codes(data.target, data.target_names)
        return df, 'target', 'classification'

    elif dataset_choice == "California Housing (Regression)":
        # Load California housing dataset (regression)
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression'
    
    elif dataset_choice == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            # Read uploaded file into a pandas DataFrame
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            target_column = st.selectbox("Select target column", df.columns)
            task_type = st.radio("Choose Task Type", ['classification', 'regression'])
            return df, target_column, task_type
        else:
            st.write("Please upload a CSV file.")
            return None, None, None

# Streamlit UI
st.title("Decision Tree Model with Streamlit")
st.write("This application allows you to train a Decision Tree model on either a classification or regression dataset.")

# Load Data
df, target_column, task_type = load_random_data()

if df is not None:
    st.write("Dataset loaded successfully!")
    st.write(df.head())

    # Select features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter settings
    max_depth = st.slider("Max Depth of the Tree", min_value=1, max_value=10, value=3)

    # Model Selection based on task type (Classification or Regression)
    if task_type == 'classification':
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Accuracy Metric for Classification
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of the Decision Tree Classifier: {accuracy * 100:.2f}%")

        # Feature Importance Plot
        feature_importance = clf.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)

        st.write("Feature Importance:")
        st.write(feature_df)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax)
        st.pyplot(fig)

        # Visualize the decision tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=feature_names, class_names=df[target_column].unique(), ax=ax)
        st.pyplot(fig)

    elif task_type == 'regression':
        clf = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Mean Squared Error Metric for Regression
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error of the Decision Tree Regressor: {mse:.2f}")

        # Feature Importance Plot for Regression
        feature_importance = clf.feature_importances_
        feature_names = X.columns
        feature_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        feature_df = feature_df.sort_values(by="Importance", ascending=False)

        st.write("Feature Importance:")
        st.write(feature_df)

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax)
        st.pyplot(fig)

        # Visualize the decision tree
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(clf, filled=True, feature_names=feature_names, ax=ax)
        st.pyplot(fig)
