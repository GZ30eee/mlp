import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.datasets import load_iris, fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from io import StringIO
import graphviz
from sklearn.tree import export_graphviz

def load_random_data():
    dataset_choice = st.sidebar.radio("Choose a Dataset", ["Iris (Classification)", "Wine (Classification)", "California Housing (Regression)", "Upload Your Own CSV"])
    
    if dataset_choice == "Iris (Classification)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'classification'
    
    elif dataset_choice == "Wine (Classification)":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'classification'
    
    elif dataset_choice == "California Housing (Regression)":
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, 'target', 'regression'
    
    elif dataset_choice == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            target_column = st.sidebar.selectbox("Select Target Column", df.columns)
            task_type = st.sidebar.radio("Choose Task Type", ['classification', 'regression'])
            return df, target_column, task_type
        else:
            return None, None, None

st.title("Decision Tree Model Explorer")
st.write("Train & visualize a decision tree model on different datasets.")

df, target_column, task_type = load_random_data()
if df is not None:
    st.write("### Dataset Preview")
    st.write(df.head())

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    st.sidebar.write("### Model Parameters")
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
    criterion = st.sidebar.radio("Criterion", ['gini', 'entropy'] if task_type == 'classification' else ['squared_error', 'friedman_mse'])
    
    if task_type == 'classification':
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    st.write("### Model Performance")
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        
        st.write("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        
        st.write("#### Classification Report")
        st.text(classification_report(y_test, y_pred))
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
        
        st.write("#### Residual Plot")
        fig, ax = plt.subplots()
        sns.histplot(y_test - y_pred, kde=True, ax=ax)
        st.pyplot(fig)
    
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    st.pyplot(fig)
    
    st.write("### Decision Tree Visualization")
    dot_data = export_graphviz(model, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
    st.graphviz_chart(dot_data)
    
    st.write("### Download Predictions")
    output_df = X_test.copy()
    output_df['Actual'] = y_test
    output_df['Predicted'] = y_pred
    st.download_button("Download Predictions", output_df.to_csv(index=False), "predictions.csv", "text/csv")
